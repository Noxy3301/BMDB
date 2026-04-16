//! BMDB entry point.

#![no_std]
#![no_main]

use core::panic::PanicInfo;

/* Bootloader entry point. */
#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    loop {}
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}
