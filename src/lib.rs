//! BMDB library crate. Shared modules used by the binary and future tests.

#![no_std]
#![feature(abi_x86_interrupt)]

pub mod gdt;
pub mod interrupts;
pub mod serial;

pub fn init() {
    gdt::init();
    interrupts::init_idt();
}
