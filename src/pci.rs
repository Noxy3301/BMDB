//! PCI configuration space access via legacy port I/O (0xCF8 / 0xCFC).
//!
//! Works on all x86 PCs without extra setup. PCIe extended config (beyond the
//! first 256 bytes) is not reachable this way; use ECAM for that.

use crate::serial_println;
use x86_64::instructions::port::Port;

const CONFIG_ADDRESS: u16 = 0xCF8;
const CONFIG_DATA: u16 = 0xCFC;

/// Read a 32-bit word from a device's config space.
/// `offset` is byte offset, must be 4-byte aligned.
fn read_config(bus: u8, device: u8, function: u8, offset: u8) -> u32 {
    // Format: [31]=enable [23:16]=bus [15:11]=device [10:8]=function [7:2]=offset/4
    let address = (1u32 << 31)
        | ((bus as u32) << 16)
        | ((device as u32) << 11)
        | ((function as u32) << 8)
        | ((offset as u32) & 0xFC);

    let mut addr_port: Port<u32> = Port::new(CONFIG_ADDRESS);
    let mut data_port: Port<u32> = Port::new(CONFIG_DATA);

    unsafe {
        addr_port.write(address);
        data_port.read()
    }
}

/// Walk all devices on `bus` and print what we find.
pub fn scan_bus(bus: u8) {
    for device in 0..32 {
        // An empty slot returns 0xFFFF as vendor ID (pull-ups on the bus).
        let vendor = (read_config(bus, device, 0, 0x00) & 0xFFFF) as u16;
        if vendor == 0xFFFF {
            continue;
        }

        // Header type bit 7 indicates multi-function device.
        let header_type = (read_config(bus, device, 0, 0x0C) >> 16) as u8;
        let max_function = if header_type & 0x80 != 0 { 8 } else { 1 };

        for function in 0..max_function {
            let vendor_device = read_config(bus, device, function, 0x00);
            let vendor = (vendor_device & 0xFFFF) as u16;
            if vendor == 0xFFFF {
                continue;
            }
            let device_id = (vendor_device >> 16) as u16;
            let class_rev = read_config(bus, device, function, 0x08);
            let class = (class_rev >> 24) as u8;
            let subclass = (class_rev >> 16) as u8;
            let prog_if = (class_rev >> 8) as u8;
            serial_println!(
                "{:02x}:{:02x}.{} vendor={:#06x} device={:#06x} class={:02x}:{:02x}:{:02x}",
                bus, device, function, vendor, device_id, class, subclass, prog_if,
            );
        }
    }
}
