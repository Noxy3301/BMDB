//! BMDB kernel main. Bootloader hands control to `kernel_main`.

#![no_std]
#![no_main]

use bmdb::serial_println;
use bootloader::{BootInfo, entry_point};
use core::panic::PanicInfo;

entry_point!(kernel_main);

fn kernel_main(_boot_info: &'static BootInfo) -> ! {
    serial_println!("Hello, BMDB");
    loop {}
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    serial_println!("panic: {}", info);
    loop {}
}
