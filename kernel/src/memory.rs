//! Paging helpers.
//!
//! The bootloader is configured (via `BootloaderConfig.mappings.physical_memory`)
//! to map all physical memory — at least the first 4 GiB, so MMIO included —
//! at a dynamically chosen virtual offset. This lets us reach page tables
//! and hardware registers through the MMU without creating ad-hoc mappings.

use bootloader_api::info::{MemoryRegionKind, MemoryRegions};
use x86_64::{
    PhysAddr, VirtAddr,
    structures::paging::{FrameAllocator, OffsetPageTable, PageTable, PhysFrame, Size4KiB},
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

/// Bump allocator handing out usable RAM frames for new page-table
/// levels — needed when we map device MMIO that the bootloader's
/// physical-memory window does not already cover (e.g. a PCI BAR the
/// UEFI firmware placed above 4 GiB). Never frees; kernel mappings are
/// permanent.
pub struct BootInfoFrameAllocator {
    regions: &'static MemoryRegions,
    next: usize,
}

impl BootInfoFrameAllocator {
    /// # Safety
    /// `regions` must accurately describe memory; the frames it reports
    /// as usable must not already be in use.
    pub unsafe fn new(regions: &'static MemoryRegions) -> Self {
        Self { regions, next: 0 }
    }

    /// Usable 4 KiB frames at or above 1 MiB. The sub-1 MiB frames are
    /// left for the SMP real-mode trampoline and its scratch page tables
    /// (see `smp`), which claim fixed low addresses. Region bounds are
    /// aligned inward so every yielded frame lies fully inside usable
    /// RAM even if the firmware reported an unaligned region.
    fn usable_frames(&self) -> impl Iterator<Item = PhysFrame> + '_ {
        self.regions
            .iter()
            .filter(|r| r.kind == MemoryRegionKind::Usable)
            .map(|r| {
                let start = (r.start.max(0x10_0000) + 0xFFF) & !0xFFF;
                let end = r.end & !0xFFF;
                (start, end)
            })
            .filter(|&(start, end)| start < end)
            .flat_map(|(start, end)| (start..end).step_by(4096))
            .map(|addr| PhysFrame::containing_address(PhysAddr::new(addr)))
    }
}

unsafe impl FrameAllocator<Size4KiB> for BootInfoFrameAllocator {
    fn allocate_frame(&mut self) -> Option<PhysFrame> {
        let frame = self.usable_frames().nth(self.next);
        self.next += 1;
        frame
    }
}
