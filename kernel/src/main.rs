//! BMDB kernel main. Bootloader hands control to `kernel_main`.

#![no_std]
#![no_main]
#![feature(abi_x86_interrupt)]

mod acpi;
mod apic;
#[cfg(feature = "bench")]
mod bench;
mod gdt;
mod interrupts;
mod memory;
mod percpu;
#[cfg(feature = "silo-bench")]
mod silo_bench;
mod smp;

use bmdb_core::lba_alloc;
use bmdb_serial::serial_println;
use bootloader_api::config::Mapping;
use bootloader_api::{BootInfo, BootloaderConfig, entry_point};
use core::panic::PanicInfo;
use x86_64::{VirtAddr, registers::control::Cr3, structures::paging::Translate};

/// Boot contract with bootloader 0.11. Physical memory must be mapped
/// wholesale (the mapping spans at least 4 GiB, so LAPIC and PCI BAR
/// MMIO stay reachable through the offset), and the kernel stack must
/// hold the bench sample buffers — the 80 KiB default is too small.
static BOOTLOADER_CONFIG: BootloaderConfig = {
    let mut config = BootloaderConfig::new_default();
    config.mappings.physical_memory = Some(Mapping::Dynamic);
    config.kernel_stack_size = 512 * 1024;
    config
};

entry_point!(kernel_main, config = &BOOTLOADER_CONFIG);

fn kernel_main(boot_info: &'static mut BootInfo) -> ! {
    // All boot-info access below is read-only; drop the &mut so the
    // memory map can be lent to the frame allocator as a 'static borrow.
    let boot_info: &'static BootInfo = boot_info;

    init();

    // Attach the framebuffer console before the first line. On the
    // ThinkCentre Tiny the video port is the only usable console: the box
    // has no serial header, and AMT Serial-over-LAN drops the moment this
    // driverless OS takes over the shared NIC — so the KT UART sink is
    // deliberately left unattached.
    init_fb_console(boot_info);

    serial_println!("Hello, BMDB");

    // Sanity-check the IDT by triggering a breakpoint.
    x86_64::instructions::interrupts::int3();

    let phys_mem_offset = VirtAddr::new(
        boot_info
            .physical_memory_offset
            .into_option()
            .expect("bootloader did not map physical memory"),
    );
    let mut mapper = unsafe { memory::init(phys_mem_offset) };
    let mut frame_allocator =
        unsafe { memory::BootInfoFrameAllocator::new(&boot_info.memory_regions) };

    let (l4_frame, _) = Cr3::read();
    serial_println!(
        "physical memory offset: {:?}, L4 page table at: {:?}",
        phys_mem_offset,
        l4_frame.start_address()
    );

    // SMP-e: install the BSP's per-CPU slot and point GS base at it.
    // BSP is always bring-up index 0.
    unsafe { percpu::init(0) };
    let bsp_cpu = unsafe { percpu::current() };
    serial_println!("PERCPU: BSP cpu_index={}", bsp_cpu.cpu_index);

    // SMP-b: the LAPIC is needed to send IPIs for AP wake-up (SMP-d).
    // Safety: bootloader's `map_physical_memory` feature maps all physical
    // memory at `phys_mem_offset`, and `kernel_main` runs exactly once.
    unsafe { apic::init(phys_mem_offset.as_u64()) };

    // SMP-c: enumerate APs via ACPI MADT so SMP-d knows what to wake.
    // UEFI firmware places the RSDP outside the legacy BIOS ROM window,
    // so the bootloader-reported address is the primary source; the
    // legacy scan inside acpi::init covers BIOS boots.
    let rsdp_addr = boot_info.rsdp_addr.into_option();
    unsafe { acpi::init(phys_mem_offset.as_u64(), rsdp_addr) };

    // SMP-d.1: copy the real-mode trampoline and probe each AP with
    // INIT + SIPI. Marker-byte validation only; full mode transition
    // and Rust AP entry come in SMP-d.2. The memory map guards the
    // low-memory pages the trampoline claims.
    unsafe { smp::init(phys_mem_offset.as_u64(), &boot_info.memory_regions) };

    serial_println!("PCI devices:");
    bmdb_pci::scan_all();

    let mut nvme = bmdb_nvme::init(phys_mem_offset, &mut mapper, &mut frame_allocator)
        .expect("NVMe init failed");

    // Zero-copy WAL path: resolve the physical address of the raw-block
    // pool so `append_no_flush` can hand it straight to the NVMe
    // controller as a PRP, skipping the driver's bounce buffer. (Earlier
    // this looked broken on real hardware, but the real cause was the WAL
    // colliding with the disk's GPT — now fixed by DEVICE_BASE — so the
    // zero-copy path is back on and re-verified against the drive.)
    unsafe {
        bmdb_core::wal::init_raw_pool(|vptr| {
            mapper
                .translate_addr(VirtAddr::new(vptr as u64))
                .expect("WAL raw pool not mapped")
                .as_u64()
        });
    }

    serial_println!(
        "LBA layout: superblock@{}, wal@{}..={} ({} blocks), data@{}..",
        lba_alloc::SUPERBLOCK_LBA,
        lba_alloc::WAL_START,
        lba_alloc::wal_end(),
        lba_alloc::WAL_LEN,
        lba_alloc::DATA_START,
    );

    #[cfg(feature = "bench")]
    bench::run_bench(&mut nvme);
    #[cfg(feature = "silo-bench")]
    silo_bench::run(&mut nvme, smp::online_aps());
    #[cfg(not(any(feature = "bench", feature = "silo-bench")))]
    run_engine(&mut nvme);

    serial_println!("It did not crash!");
    finish();
}

/// Default workload: the durable transaction gate. With the `hw-loop`
/// feature it is bracketed by unattended real-hardware diagnostics — a
/// boot counter and raw-block / WAL durability probes — that report over
/// the video console during a self-resetting PXE test cycle.
#[cfg(not(any(feature = "bench", feature = "silo-bench")))]
fn run_engine(nvme: &mut bmdb_nvme::Controller) {
    #[cfg(feature = "hw-loop")]
    boot_counter(nvme);
    #[cfg(feature = "hw-loop")]
    nvme_selftest(nvme);
    engine_durable_gate(nvme);
    #[cfg(feature = "hw-loop")]
    wal_readback(nvme);
}

/// End of boot. The unattended hardware-test build self-resets to run the
/// next PXE image (AMT cannot reset the box while this driverless OS holds
/// the shared NIC); every other build simply halts.
#[cfg(all(feature = "hw-loop", not(any(feature = "bench", feature = "silo-bench"))))]
fn finish() -> ! {
    delay_then_reset()
}
#[cfg(not(all(feature = "hw-loop", not(any(feature = "bench", feature = "silo-bench")))))]
fn finish() -> ! {
    hlt_loop()
}

/// Read, bump, and persist a boot counter in a dedicated raw block via the
/// bounce-buffer write path (the one real hardware honors), so it survives
/// the self-reset cycle even while the WAL path is under repair. A rising
/// number across screenshots means the box is still cycling; a stuck
/// number means a boot wedged before this point.
#[cfg(all(feature = "hw-loop", not(any(feature = "bench", feature = "silo-bench"))))]
fn boot_counter(nvme: &mut bmdb_nvme::Controller) {
    use bmdb_core::lba_alloc::{BLOCK_SIZE, DATA_START};

    let lba = DATA_START + 1;
    let mut blk = [0u8; BLOCK_SIZE];
    let prev = if nvme.read_block(lba, &mut blk).is_ok() {
        u64::from_le_bytes([
            blk[0], blk[1], blk[2], blk[3], blk[4], blk[5], blk[6], blk[7],
        ])
    } else {
        0
    };
    // A never-written block reads back as drive garbage; treat an
    // implausible value as a fresh start so the count stays legible.
    let prev = if prev > 1_000_000 { 0 } else { prev };
    let next = prev + 1;
    blk = [0u8; BLOCK_SIZE];
    blk[..8].copy_from_slice(&next.to_le_bytes());
    let _ = nvme.write_block(lba, &blk);
    let _ = nvme.flush();
    serial_println!("BMDB: boot #{}", next);
}

/// Hold the boot log on screen for a capture window, then reset the
/// machine so the next PXE image runs. Warm reset via the Intel PCH
/// reset-control register (0xCF9), with the legacy 8042 pulse as a
/// fallback — either re-enters POST and PXE.
#[cfg(all(feature = "hw-loop", not(any(feature = "bench", feature = "silo-bench"))))]
fn delay_then_reset() -> ! {
    use x86_64::instructions::port::Port;

    serial_println!("BMDB: self-reset in ~30s for the next PXE cycle");
    // TSC has no known frequency here, so this is a rough wall-clock wait
    // (~20-45s across plausible core clocks) — ample to capture a frame.
    let start = unsafe { core::arch::x86_64::_rdtsc() };
    while unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(start) < 90_000_000_000 {
        core::hint::spin_loop();
    }
    unsafe {
        let mut rst: Port<u8> = Port::new(0xCF9);
        rst.write(0x02); // set RST_CPU arming bit
        rst.write(0x06); // RST_CPU | SYS_RST -> warm reset
        // If 0xCF9 did nothing, pulse the 8042 controller reset line.
        let mut kbd: Port<u8> = Port::new(0x64);
        kbd.write(0xFE);
    }
    hlt_loop()
}

/// Read the first WAL blocks straight back right after the engine gate
/// committed to them, in the same boot. If these are BAD while the
/// scratch-block roundtrip above is OK, writes to the low WAL LBAs are
/// being dropped by the drive even though writes to the data region land
/// — a region/LBA problem, not the write path. Also prints a build tag so
/// the running image is unambiguous across the reboot loop.
#[cfg(all(feature = "hw-loop", not(any(feature = "bench", feature = "silo-bench"))))]
fn wal_readback(nvme: &mut bmdb_nvme::Controller) {
    use bmdb_core::lba_alloc::{BLOCK_SIZE, WAL_START};

    serial_println!("WAL-READBACK build tag: bounce-path-v2");
    for off in 0..3u64 {
        let lba = WAL_START + off;
        let mut b = [0u8; BLOCK_SIZE];
        if nvme.read_block(lba, &mut b).is_ok() {
            let magic = u64::from_le_bytes([
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
            ]);
            serial_println!(
                "WAL-READBACK: lba={} magic={} first8=[{:#04x} {:#04x} {:#04x} {:#04x} {:#04x} {:#04x} {:#04x} {:#04x}]",
                lba,
                if magic == bmdb_core::wal::WAL_MAGIC { "OK" } else { "BAD" },
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
            );
        }
    }
}

/// Raw-block durability probe, independent of the WAL and engine.
///
/// Reads a scratch block first (so a sentinel written by the previous
/// boot proves the drive persisted it across the power cycle), then
/// writes a fresh sentinel + pattern, flushes, and reads it straight
/// back (proving the in-boot write/read path itself). On real hardware
/// this separates "the write never reaches media" from "recovery logic
/// drops it".
#[cfg(all(feature = "hw-loop", not(any(feature = "bench", feature = "silo-bench"))))]
fn nvme_selftest(nvme: &mut bmdb_nvme::Controller) {
    use bmdb_core::lba_alloc::{BLOCK_SIZE, DATA_START};

    let lba = DATA_START;

    let mut before = [0u8; BLOCK_SIZE];
    match nvme.read_block(lba, &mut before) {
        Ok(()) => serial_println!(
            "NVMe self-test: pre-read lba={} first=[{:#04x} {:#04x} {:#04x} {:#04x}] (sentinel db b0 0f if it survived reboot)",
            lba, before[0], before[1], before[2], before[3],
        ),
        Err(_) => serial_println!("NVMe self-test: pre-read FAILED"),
    }

    let mut buf = [0u8; BLOCK_SIZE];
    for (i, b) in buf.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(7) ^ 0x5A;
    }
    buf[0] = 0xDB;
    buf[1] = 0xB0;
    buf[2] = 0x0F;

    if nvme.write_block(lba, &buf).is_err() {
        serial_println!("NVMe self-test: WRITE FAILED");
        return;
    }
    if nvme.flush().is_err() {
        serial_println!("NVMe self-test: FLUSH FAILED");
    }

    let mut after = [0u8; BLOCK_SIZE];
    if nvme.read_block(lba, &mut after).is_err() {
        serial_println!("NVMe self-test: read-back FAILED");
        return;
    }
    serial_println!(
        "NVMe self-test: in-boot roundtrip {} (read-back first=[{:#04x} {:#04x} {:#04x}])",
        if after == buf { "OK" } else { "MISMATCH" },
        after[0],
        after[1],
        after[2],
    );

    // Probe the WAL's first block as left by the PREVIOUS boot. The engine
    // writes the WAL via the zero-copy path (write_block_from_phys), unlike
    // the bounce-buffer writes above. Runs before this boot's engine gate,
    // so a valid magic here means a prior boot's zero-copy write reached
    // media; a BAD magic while the scratch sentinel survived pins the bug
    // on the zero-copy path rather than on NVMe persistence.
    let mut wal = [0u8; BLOCK_SIZE];
    if nvme.read_block(bmdb_core::lba_alloc::WAL_START, &mut wal).is_ok() {
        let magic = u64::from_le_bytes([
            wal[0], wal[1], wal[2], wal[3], wal[4], wal[5], wal[6], wal[7],
        ]);
        serial_println!(
            "NVMe self-test: wal[{}] magic={} first8=[{:#04x} {:#04x} {:#04x} {:#04x} {:#04x} {:#04x} {:#04x} {:#04x}]",
            bmdb_core::lba_alloc::WAL_START,
            if magic == bmdb_core::wal::WAL_MAGIC { "OK" } else { "BAD" },
            wal[0], wal[1], wal[2], wal[3], wal[4], wal[5], wal[6], wal[7],
        );
    }
}

/// Transaction-engine crash-recovery gate.
///
/// Rebuilds the durable store from the WAL, verifies every key a prior
/// boot committed is still present, then durably commits one more key.
/// Runs on every boot; the recovered count grows by one per run, proving
/// transactional durability across `timeout` / kill / restart cycles.
/// Also exercises a durable multi-key transaction with a delete.
#[cfg(not(any(feature = "bench", feature = "silo-bench")))]
fn engine_durable_gate(nvme: &mut bmdb_nvme::Controller) {
    use bmdb_core::engine::Engine;

    // 512 records × 64 bytes = 32 KiB; fine as a kernel static.
    static ENGINE: Engine = Engine::new();

    ENGINE.recover(nvme).expect("engine recover failed");

    // Prior boots committed keys 1..=N durably (value = key * 10), one new
    // key per boot. Count the survivors and verify each value.
    let mut recovered = 0u64;
    loop {
        let key = (recovered + 1).to_be_bytes();
        match ENGINE.transaction(8, |t| t.get(key)).flatten() {
            Some(v) => {
                assert_eq!(v, (recovered + 1) * 10, "recovered value mismatch");
                recovered += 1;
            }
            None => break,
        }
    }
    serial_println!("ENGINE: recovered {} durable key(s)", recovered);

    // Durably commit the next key; it must survive the next boot.
    let next = recovered + 1;
    ENGINE
        .transaction_durable(nvme, 8, |t| t.put(next.to_be_bytes(), next * 10))
        .expect("durable commit I/O error")
        .expect("durable commit aborted");
    serial_println!("ENGINE: durably committed key {} (total {} key(s))", next, next);

    // Durable multi-key transaction with a delete, on a high key range so
    // it does not disturb the per-boot counter above.
    ENGINE
        .transaction_durable(nvme, 8, |t| {
            t.put(1001u64.to_be_bytes(), 11)?;
            t.put(1002u64.to_be_bytes(), 22)?;
            t.delete(1001u64.to_be_bytes())?;
            Ok(())
        })
        .expect("durable demo I/O error")
        .expect("durable demo aborted");
    let state = ENGINE
        .transaction(8, |t| {
            Ok((t.get(1001u64.to_be_bytes())?, t.get(1002u64.to_be_bytes())?))
        })
        .expect("post-demo read must commit");
    assert_eq!(state, (None, Some(22)), "put+delete demo left wrong state");
    serial_println!("ENGINE: multi-key durable txn + delete OK (1001 deleted, 1002=22)");
}

fn init() {
    gdt::init();
    interrupts::init_idt();
}

/// Attach the bootloader's linear framebuffer as a console sink so the
/// boot log appears on the video output. This is the only console that
/// survives on the target box (no serial header; AMT SoL drops under a
/// driverless OS), read back with a capture card on the video port.
fn init_fb_console(boot_info: &'static BootInfo) {
    use bootloader_api::info::PixelFormat;

    let Some(fb) = boot_info.framebuffer.as_ref() else {
        serial_println!("FB: no framebuffer in boot_info");
        return;
    };
    let info = fb.info();
    serial_println!(
        "FB: {}x{} stride={} bpp={} fmt={:?} buf={:p} len={}",
        info.width,
        info.height,
        info.stride,
        info.bytes_per_pixel,
        info.pixel_format,
        fb.buffer().as_ptr(),
        fb.buffer().len(),
    );
    let kind = match info.pixel_format {
        PixelFormat::Rgb => bmdb_serial::PixelKind::Rgb,
        PixelFormat::Bgr => bmdb_serial::PixelKind::Bgr,
        PixelFormat::U8 => bmdb_serial::PixelKind::Gray,
        // Unknown byte order: assume the common 32-bit BGRx and let the
        // capture reveal whether the channels need swapping.
        _ => bmdb_serial::PixelKind::Bgr,
    };
    // Scale the 8x8 font up on high-resolution panels, but keep the cell
    // small enough (~16px, ~68 rows) that the whole boot log fits without
    // scrolling: a scroll memmoves the entire framebuffer over the slow
    // video aperture, which stalls output line by line on real hardware.
    // The capture side can zoom in for a closer read.
    let scale = (info.height / 540).max(1);
    let buffer = fb.buffer();
    // Safety: the bootloader maps the framebuffer for the kernel's
    // lifetime and memory::init keeps that mapping (no CR3 switch); the
    // reported geometry matches the buffer.
    unsafe {
        bmdb_serial::install_fb(
            buffer.as_ptr() as *mut u8,
            buffer.len(),
            info.width,
            info.height,
            info.stride,
            info.bytes_per_pixel,
            kind,
            scale,
        );
    }
}

/// Attach the Intel AMT SoL "KT" UART (a 16550-compatible PCI serial
/// function) as a second console sink. The I/O-BAR path needs no paging
/// setup, so this can run before memory init and catch every line.
///
/// Unused on the ThinkCentre Tiny: poking the KT UART takes the
/// management engine's out-of-band network down. Kept for boxes where
/// SoL is the only console.
#[allow(dead_code)]
fn init_sol_console() {
    let Some(addr) = bmdb_pci::find_device(0x07, 0x00) else {
        return;
    };
    bmdb_pci::enable_device(&addr);
    let found = match bmdb_pci::read_bar(&addr, 0) {
        // Safety: the BAR belongs to a function advertising the 16550-
        // compatible serial class, and I/O decoding was just enabled.
        bmdb_pci::Bar::Io { port } => unsafe { bmdb_serial::install_kt_pio(port) },
        // A memory-BAR KT would need an MMU mapping this early in boot;
        // no target machine has surfaced one, so just report it.
        bmdb_pci::Bar::Mmio { .. } => false,
    };
    if found {
        serial_println!(
            "console: AMT SoL KT UART at {:02x}:{:02x}.{}",
            addr.bus,
            addr.device,
            addr.function,
        );
    }
}

/// Halt the CPU until the next interrupt. Used in idle loops to avoid burning
/// power on a tight spin.
fn hlt_loop() -> ! {
    loop {
        x86_64::instructions::hlt();
    }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    serial_println!("panic: {}", info);
    hlt_loop();
}
