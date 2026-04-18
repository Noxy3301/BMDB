//! BMDB kernel main. Bootloader hands control to `kernel_main`.

#![no_std]
#![no_main]
#![feature(abi_x86_interrupt)]

mod gdt;
mod interrupts;
mod memory;

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

    wal_smoke_test(&mut nvme);

    serial_println!("It did not crash!");
    hlt_loop();
}

/// Append a few records and read them back. Until recovery is wired up, this
/// only verifies that encoding, NVMe I/O, and checksums agree in one session.
fn wal_smoke_test(nvme: &mut bmdb_nvme::Controller) {
    let mut wal = Wal::new();
    let sample: [(Op, [u8; 8], [u8; 8]); 3] = [
        (Op::Put, *b"alpha\0\0\0", *b"A\0\0\0\0\0\0\0"),
        (Op::Put, *b"bravo\0\0\0", *b"B\0\0\0\0\0\0\0"),
        (Op::Delete, *b"alpha\0\0\0", [0; 8]),
    ];

    let mut logged: [(u64, bmdb_core::lba_alloc::Lba); 3] =
        [(0, 0), (0, 0), (0, 0)];
    for (i, (op, k, v)) in sample.iter().enumerate() {
        let lba = wal.next_lba();
        let lsn = wal.append(nvme, *op, 0, *k, *v).expect("WAL append failed");
        logged[i] = (lsn, lba);
        serial_println!("WAL: append lsn={} at lba={} op={:?}", lsn, lba, op);
    }

    for (i, (expected_lsn, lba)) in logged.iter().enumerate() {
        let rec = Wal::read_at(nvme, *lba)
            .expect("WAL read failed")
            .expect("WAL record missing");
        assert_eq!(rec.lsn, *expected_lsn);
        assert_eq!(rec.key, sample[i].1);
        assert_eq!(rec.op(), Some(sample[i].0));
        if sample[i].0 == Op::Put {
            assert_eq!(rec.value, sample[i].2);
        }
    }
    serial_println!("WAL: 3 records round-tripped through NVMe");
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
