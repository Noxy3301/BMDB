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

    // Mirror the console to the AMT SoL KT UART before the first line,
    // so machines without a legacy COM1 (headless vPro boxes) get the
    // whole boot log over Serial-over-LAN.
    init_sol_console();

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

    // Zero-copy WAL path: resolve the physical address of the raw-
    // block pool so `append_no_flush` can hand it straight to the
    // NVMe controller as a PRP, skipping the driver's bounce buffer.
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
    engine_durable_gate(&mut nvme);

    serial_println!("It did not crash!");
    hlt_loop();
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

/// Attach the Intel AMT SoL "KT" UART (a 16550-compatible PCI serial
/// function) as a second console sink. The I/O-BAR path needs no paging
/// setup, so this can run before memory init and catch every line.
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
