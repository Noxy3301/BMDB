//! Fixed partitioning of the block device into superblock, WAL, and data.
//!
//! Phase 3 uses a static layout: every region's start and length is a
//! compile-time constant. A runtime free-block allocator for the data region
//! will be added once workloads can actually fill it.

/// Block size in bytes. Must match the device's logical block size.
pub const BLOCK_SIZE: usize = 512;

/// Logical block address on the backing device.
pub type Lba = u64;

/// Block that holds the on-disk superblock. One block, always at LBA 0.
pub const SUPERBLOCK_LBA: Lba = 0;

/// First WAL block. Records are appended starting here and wrap when the
/// region fills; recovery replays from the last checkpoint.
pub const WAL_START: Lba = 1;

/// Number of blocks reserved for the WAL (~4 MB at 512 B).
pub const WAL_LEN: u64 = 8_191;

/// First data block. B+tree nodes and tuple pages live at or above this LBA.
pub const DATA_START: Lba = WAL_START + WAL_LEN;

/// Last block of the WAL region, inclusive.
#[inline]
pub const fn wal_end() -> Lba {
    WAL_START + WAL_LEN - 1
}

/// Classification of an LBA for invariants and debug asserts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    Superblock,
    Wal,
    Data,
}

/// Which region `lba` falls into. `Data` is returned for any LBA past the WAL
/// tail; callers that need to bound-check against the device capacity must do
/// so themselves.
pub const fn region_of(lba: Lba) -> Region {
    if lba == SUPERBLOCK_LBA {
        Region::Superblock
    } else if lba <= wal_end() {
        Region::Wal
    } else {
        Region::Data
    }
}
