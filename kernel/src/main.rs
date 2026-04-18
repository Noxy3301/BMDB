//! BMDB kernel main. Bootloader hands control to `kernel_main`.

#![no_std]
#![no_main]
#![feature(abi_x86_interrupt)]

mod gdt;
mod interrupts;
mod memory;

use bmdb_core::bptree::{BpTree, InsertError};
use bmdb_core::lba_alloc;
use bmdb_core::wal::{Op, Wal};
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

    wal_recovery_test(&mut nvme);
    bptree_smoke_test();

    serial_println!("It did not crash!");
    hlt_loop();
}

/// Insert 50 keys in reverse order to force multiple splits, then look them
/// all up. Also exercises duplicate rejection and miss-on-absent-key. Lives
/// entirely in memory — no WAL, no NVMe.
fn bptree_smoke_test() {
    let mut tree = BpTree::new();

    for i in (1u64..=50).rev() {
        let key = i.to_be_bytes();
        let value = (i * 100).to_be_bytes();
        tree.insert(key, value).expect("bptree insert failed");
    }

    for i in 1u64..=50 {
        let key = i.to_be_bytes();
        let expected = (i * 100).to_be_bytes();
        let got = tree.lookup(key).expect("bptree lookup missing");
        assert_eq!(got, expected, "value mismatch for key {}", i);
    }

    assert!(tree.lookup(999u64.to_be_bytes()).is_none());
    assert_eq!(
        tree.insert(1u64.to_be_bytes(), [0; 8]),
        Err(InsertError::DuplicateKey),
    );

    serial_println!(
        "B+tree: 50 inserts + lookups OK, nodes={}, height={}",
        tree.num_nodes(),
        tree.height(),
    );
}

/// Recover the WAL, list existing records, and append one new record every
/// boot. If persistence works, the total record count grows by one per
/// QEMU run.
fn wal_recovery_test(nvme: &mut bmdb_nvme::Controller) {
    let mut wal = Wal::recover(nvme).expect("WAL recover failed");
    let existing = wal.next_lsn().saturating_sub(1);
    serial_println!(
        "WAL: recovered {} record(s), next_lba={}, next_lsn={}",
        existing,
        wal.next_lba(),
        wal.next_lsn(),
    );

    for lba in lba_alloc::WAL_START..wal.next_lba() {
        let rec = Wal::read_at(nvme, lba)
            .expect("WAL read failed")
            .expect("WAL record missing during dump");
        serial_println!(
            "  lba={} lsn={} op={:?} key={:?}",
            lba,
            rec.lsn,
            rec.op(),
            rec.key
        );
    }

    let lba = wal.next_lba();
    let lsn = wal
        .append(nvme, Op::Put, 0, *b"boot\0\0\0\0", [0xBE, 0xEF, 0, 0, 0, 0, 0, 0])
        .expect("WAL append failed");
    serial_println!("WAL: appended lsn={} at lba={} (durable)", lsn, lba);
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
