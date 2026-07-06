//! PCI configuration space access via legacy port I/O (0xCF8 / 0xCFC).
//!
//! Works on all x86 PCs without extra setup. PCIe extended config (beyond the
//! first 256 bytes) is not reachable this way; use ECAM for that.

#![no_std]

use bmdb_serial::serial_println;
use x86_64::instructions::port::Port;

const CONFIG_ADDRESS: u16 = 0xCF8;
const CONFIG_DATA: u16 = 0xCFC;

/// Location of a PCI function within the host's PCI hierarchy.
#[derive(Debug, Clone, Copy)]
pub struct PciAddress {
    pub bus: u8,
    pub device: u8,
    pub function: u8,
}

/// A decoded Base Address Register. The low bits of the raw value select
/// the address space; callers need to know which one they got, since an
/// I/O BAR is reached with `in`/`out` and a memory BAR through the MMU.
#[derive(Debug, Clone, Copy)]
pub enum Bar {
    Io { port: u16 },
    Mmio { addr: u64 },
}

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

fn write_config(bus: u8, device: u8, function: u8, offset: u8, value: u32) {
    let address = (1u32 << 31)
        | ((bus as u32) << 16)
        | ((device as u32) << 11)
        | ((function as u32) << 8)
        | ((offset as u32) & 0xFC);

    let mut addr_port: Port<u32> = Port::new(CONFIG_ADDRESS);
    let mut data_port: Port<u32> = Port::new(CONFIG_DATA);

    unsafe {
        addr_port.write(address);
        data_port.write(value);
    }
}

/// Tracks buses already walked. Broken firmware can wire bridge
/// secondary-bus numbers into a cycle; 256 bits of state is cheaper
/// than trusting it not to.
struct BusVisited([u64; 4]);

impl BusVisited {
    fn new() -> Self {
        BusVisited([0; 4])
    }

    /// Returns `true` the first time `bus` is claimed.
    fn claim(&mut self, bus: u8) -> bool {
        let word = (bus >> 6) as usize;
        let bit = 1u64 << (bus & 63);
        let fresh = self.0[word] & bit == 0;
        self.0[word] |= bit;
        fresh
    }
}

/// Walk one bus, invoking `f` for every function, and recurse into
/// PCI-PCI bridges. Real machines put NVMe (and most everything else)
/// behind root ports on secondary buses, so a flat bus-0 scan only
/// works on QEMU's default machine.
fn walk_bus(bus: u8, visited: &mut BusVisited, f: &mut impl FnMut(PciAddress, u8, u8, u8)) {
    if !visited.claim(bus) {
        return;
    }
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
            let vendor = (read_config(bus, device, function, 0x00) & 0xFFFF) as u16;
            if vendor == 0xFFFF {
                continue;
            }
            let class_rev = read_config(bus, device, function, 0x08);
            let class = (class_rev >> 24) as u8;
            let subclass = (class_rev >> 16) as u8;
            let prog_if = (class_rev >> 8) as u8;

            f(
                PciAddress { bus, device, function },
                class,
                subclass,
                prog_if,
            );

            // Header type 0x01 = PCI-PCI bridge; its downstream bus
            // number lives in config offset 0x19.
            let fn_header = (read_config(bus, device, function, 0x0C) >> 16) as u8;
            if fn_header & 0x7F == 0x01 {
                let secondary = ((read_config(bus, device, function, 0x18) >> 8) & 0xFF) as u8;
                if secondary != 0 {
                    walk_bus(secondary, visited, f);
                }
            }
        }
    }
}

/// Walk the whole hierarchy from bus 0 and print what we find.
pub fn scan_all() {
    let mut visited = BusVisited::new();
    walk_bus(0, &mut visited, &mut |addr, class, subclass, prog_if| {
        let vendor_device = read_config(addr.bus, addr.device, addr.function, 0x00);
        serial_println!(
            "{:02x}:{:02x}.{} vendor={:#06x} device={:#06x} class={:02x}:{:02x}:{:02x}",
            addr.bus,
            addr.device,
            addr.function,
            (vendor_device & 0xFFFF) as u16,
            (vendor_device >> 16) as u16,
            class,
            subclass,
            prog_if,
        );
    });
}

/// Find the first function anywhere in the hierarchy matching the given
/// class / subclass.
pub fn find_device(class: u8, subclass: u8) -> Option<PciAddress> {
    let mut found = None;
    let mut visited = BusVisited::new();
    walk_bus(0, &mut visited, &mut |addr, c, s, _| {
        if c == class && s == subclass && found.is_none() {
            found = Some(addr);
        }
    });
    found
}

/// Decode a BAR. I/O BARs (bit 0 set) hold a port number; memory BARs
/// hold a 32-bit or 64-bit physical address (bits [2:1] = 10 selects
/// 64-bit, with the upper half in the next dword).
pub fn read_bar(addr: &PciAddress, bar_index: u8) -> Bar {
    let offset = 0x10 + bar_index * 4;
    let lower = read_config(addr.bus, addr.device, addr.function, offset);

    if lower & 1 != 0 {
        return Bar::Io {
            port: (lower & 0xFFFF_FFFC) as u16,
        };
    }

    let is_64 = (lower & 0b110) == 0b100;
    let base_low = (lower & 0xFFFF_FFF0) as u64;
    if is_64 {
        let upper = read_config(addr.bus, addr.device, addr.function, offset + 4);
        Bar::Mmio {
            addr: ((upper as u64) << 32) | base_low,
        }
    } else {
        Bar::Mmio { addr: base_low }
    }
}

/// Enable I/O Space (bit 0), Memory Space (bit 1) and Bus Master (bit 2)
/// in the command register. I/O and Memory let the CPU reach the
/// device's BARs of either kind; Bus Master lets the device initiate DMA
/// back into system RAM. Setting a space bit the device has no BARs for
/// is a no-op.
pub fn enable_device(addr: &PciAddress) {
    // Command and status share one dword; preserve status by masking.
    let dword = read_config(addr.bus, addr.device, addr.function, 0x04);
    let command = (dword & 0xFFFF) | (1 << 0) | (1 << 1) | (1 << 2);
    write_config(
        addr.bus,
        addr.device,
        addr.function,
        0x04,
        (dword & 0xFFFF_0000) | command,
    );
}
