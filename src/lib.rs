//! BMDB library crate. Shared modules used by the binary and future tests.

#![no_std]
#![feature(abi_x86_interrupt)]

pub mod interrupts;
pub mod serial;

pub fn init() {
    interrupts::init_idt();
}
