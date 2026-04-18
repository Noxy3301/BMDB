//! BMDB kernel main. Bootloader hands control to `kernel_main`.

#![no_std]
#![no_main]

use bmdb::{hlt_loop, memory, nvme, pci, serial_println};
use bootloader::{BootInfo, entry_point};
use core::panic::PanicInfo;
use x86_64::{VirtAddr, registers::control::Cr3};

entry_point!(kernel_main);

fn kernel_main(boot_info: &'static BootInfo) -> ! {
    serial_println!("Hello, BMDB");

    bmdb::init();

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
    pci::scan_bus(0);

    nvme::init(phys_mem_offset, &mapper);

    serial_println!("It did not crash!");
    hlt_loop();
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    serial_println!("panic: {}", info);
    hlt_loop();
}
