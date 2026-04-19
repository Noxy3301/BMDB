//! BMDB kernel main. Bootloader hands control to `kernel_main`.

#![no_std]
#![no_main]
#![feature(abi_x86_interrupt)]

mod gdt;
mod interrupts;
mod memory;

use bmdb_core::kv::Kv;
use bmdb_core::lba_alloc;
use bmdb_serial::serial_println;
use bootloader::{BootInfo, entry_point};
use core::panic::PanicInfo;
use x86_64::{VirtAddr, registers::control::Cr3};

entry_point!(kernel_main);

fn kernel_main(boot_info: &'static BootInfo) -> ! {
    serial_println!("Hello, BMDB");

    init();

    // Sanity-check the IDT by triggering a breakpoint.
    x86_64::instructions::interrupts::int3();

    let phys_mem_offset = VirtAddr::new(boot_info.physical_memory_offset);
    let mapper = unsafe { memory::init(phys_mem_offset) };

    let (l4_frame, _) = Cr3::read();
    serial_println!(
        "physical memory offset: {:?}, L4 page table at: {:?}",
        phys_mem_offset,
        l4_frame.start_address()
    );

    serial_println!("PCI devices on bus 0:");
    bmdb_pci::scan_bus(0);

    let mut nvme = bmdb_nvme::init(phys_mem_offset, &mapper).expect("NVMe init failed");

    serial_println!(
        "LBA layout: superblock@{}, wal@{}..={} ({} blocks), data@{}..",
        lba_alloc::SUPERBLOCK_LBA,
        lba_alloc::WAL_START,
        lba_alloc::wal_end(),
        lba_alloc::WAL_LEN,
        lba_alloc::DATA_START,
    );

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
