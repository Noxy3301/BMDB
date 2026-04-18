//! BMDB kernel main. Bootloader hands control to `kernel_main`.

#![no_std]
#![no_main]
#![feature(abi_x86_interrupt)]

mod gdt;
mod interrupts;
mod memory;

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

    bmdb_nvme::init(phys_mem_offset, &mapper);

    serial_println!("It did not crash!");
    hlt_loop();
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
