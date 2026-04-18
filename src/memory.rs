//! Paging helpers.
//!
//! The bootloader is configured (via the `map_physical_memory` feature) to map
//! all physical RAM at a fixed virtual offset. This lets us reach page tables
//! and hardware registers through the MMU without creating ad-hoc mappings.

use x86_64::{
    VirtAddr,
    structures::paging::{OffsetPageTable, PageTable},
};

/// Build an `OffsetPageTable` view over the currently active page tables.
///
/// Caller must guarantee:
/// - All physical memory is mapped at `physical_memory_offset`.
/// - This is called once, so the returned `&'static mut` is unique.
pub unsafe fn init(physical_memory_offset: VirtAddr) -> OffsetPageTable<'static> {
    let level_4_table = unsafe { active_level_4_table(physical_memory_offset) };
    unsafe { OffsetPageTable::new(level_4_table, physical_memory_offset) }
}

/// Return a mutable reference to the active level-4 page table.
///
/// CR3 stores the L4 table as a physical frame number; the MMU can only read
/// virtual addresses, so we translate through the bootloader's offset mapping.
unsafe fn active_level_4_table(physical_memory_offset: VirtAddr) -> &'static mut PageTable {
    use x86_64::registers::control::Cr3;

    let (frame, _) = Cr3::read();
    let virt = physical_memory_offset + frame.start_address().as_u64();
    unsafe { &mut *virt.as_mut_ptr() }
}
