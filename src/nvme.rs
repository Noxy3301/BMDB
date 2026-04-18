//! NVMe controller identification via BAR0 MMIO access.

use crate::{pci, serial_println};
use x86_64::VirtAddr;

// Register offsets from the NVMe spec, Section 3 (Controller Registers).
const REG_CAP: usize = 0x00; // Controller Capabilities (64-bit)
const REG_VS: usize = 0x08; // Version (32-bit)
const REG_CSTS: usize = 0x1C; // Controller Status (32-bit)

/// Find the NVMe controller on bus 0, map its BAR into virtual memory, and
/// print the identifying registers.
pub fn probe(phys_mem_offset: VirtAddr) {
    let Some(addr) = pci::find_device(0x01, 0x08) else {
        serial_println!("NVMe: no controller found");
        return;
    };
    serial_println!(
        "NVMe: found at {:02x}:{:02x}.{}",
        addr.bus,
        addr.device,
        addr.function
    );

    pci::enable_mmio(&addr);

    let bar0 = pci::read_bar(&addr, 0);
    serial_println!("NVMe: BAR0 physical = {:#x}", bar0);

    // The bootloader's map_physical_memory covers the full physical address
    // range, so BAR regions are reachable through the same offset.
    let regs = (phys_mem_offset.as_u64() + bar0) as *mut u8;
    unsafe {
        let cap = core::ptr::read_volatile(regs.add(REG_CAP) as *const u64);
        let vs = core::ptr::read_volatile(regs.add(REG_VS) as *const u32);
        let csts = core::ptr::read_volatile(regs.add(REG_CSTS) as *const u32);
        serial_println!("NVMe: CAP  = {:#018x}", cap);
        serial_println!(
            "NVMe: VS   = {}.{}.{}",
            (vs >> 16) & 0xFFFF,
            (vs >> 8) & 0xFF,
            vs & 0xFF
        );
        serial_println!("NVMe: CSTS = {:#010x}", csts);
    }
}
