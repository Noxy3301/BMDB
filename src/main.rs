//! BMDB kernel main. Bootloader hands control to `kernel_main`.

#![no_std]
#![no_main]

use bmdb::serial_println;
use bootloader::{BootInfo, entry_point};
use core::panic::PanicInfo;

entry_point!(kernel_main);

fn kernel_main(_boot_info: &'static BootInfo) -> ! {
    serial_println!("Hello, BMDB");

    bmdb::init();

    // Sanity-check the IDT by triggering a breakpoint.
    x86_64::instructions::interrupts::int3();

    serial_println!("It did not crash!");
    loop {}
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    serial_println!("panic: {}", info);
    loop {}
}
