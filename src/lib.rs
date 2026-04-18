//! BMDB library crate. Shared modules used by the binary and future tests.

#![no_std]
#![feature(abi_x86_interrupt)]

pub mod gdt;
pub mod interrupts;
pub mod memory;
pub mod nvme;
pub mod pci;
pub mod serial;

pub fn init() {
    gdt::init();
    interrupts::init_idt();
}

/// Halt the CPU until the next interrupt. Used in idle loops to avoid burning
/// power on a tight spin.
pub fn hlt_loop() -> ! {
    loop {
        x86_64::instructions::hlt();
    }
}
