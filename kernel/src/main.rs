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

#[cfg(not(any(feature = "bench", feature = "silo-bench")))]
use bmdb_core::kv::Kv;
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
    kv_gate_test(&mut nvme);

    serial_println!("It did not crash!");
    hlt_loop();
}

/// Phase 3 crash-recovery gate.
///
/// Recovers the KV by replaying the WAL, inserts one new record keyed by the
/// next LSN, and verifies that every previously-recovered record is still
/// readable. Runs on every boot; the recovered count grows by one per run,
/// proving durability across `timeout` / kill / restart cycles.
#[cfg(not(any(feature = "bench", feature = "silo-bench")))]
fn kv_gate_test(nvme: &mut bmdb_nvme::Controller) {
    let mut kv = Kv::recover(nvme).expect("KV recover failed");

    let lsn_at_start = kv.next_lsn();
    let recovered = lsn_at_start.saturating_sub(1);
    let (nodes, height) = kv.tree_stats();
    serial_println!(
        "KV: recovered {} record(s), next_lsn={}, tree nodes={}, height={}",
        recovered,
        lsn_at_start,
        nodes,
        height,
    );

    // Every prior boot wrote key = lsn.to_be_bytes(), value = (lsn * 10)
    // .to_be_bytes(). Confirm all of those are still in the tree.
    for lsn in 1..lsn_at_start {
        let key = lsn.to_be_bytes();
        let expected = (lsn * 10).to_be_bytes();
        let got = kv.get(key).expect("recovered key missing from tree");
        assert_eq!(got, expected, "recovered value mismatch for lsn={}", lsn);
    }

    // Append one more record tagged with the next LSN.
    let new_lsn = kv.next_lsn();
    let new_key = new_lsn.to_be_bytes();
    let new_value = (new_lsn * 10).to_be_bytes();
    let prior = kv.put(nvme, new_key, new_value).expect("KV put failed");
    assert!(prior.is_none(), "fresh LSN should have no prior value");

    // Immediate read-back.
    let echo = kv.get(new_key).expect("KV get after put returned None");
    assert_eq!(echo, new_value);

    serial_println!("KV: put+get OK (new lsn={}), total keys={}", new_lsn, new_lsn);
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
