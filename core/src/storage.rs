//! Block-oriented storage interface used by the DB core.
//!
//! Any backend (NVMe on bare metal, a RAM-backed fake for tests, a host-side
//! file for offline verification) implements this trait. The core never names
//! a concrete driver, so storage-dependent logic stays testable in isolation.

use crate::lba_alloc::{BLOCK_SIZE, Lba};

pub trait BlockStorage {
    type Error: core::fmt::Debug;

    fn read_block(&mut self, lba: Lba, out: &mut [u8; BLOCK_SIZE]) -> Result<(), Self::Error>;
    fn write_block(&mut self, lba: Lba, data: &[u8; BLOCK_SIZE]) -> Result<(), Self::Error>;

    /// Zero-copy write. The caller hands the backend a buffer whose
    /// physical address has already been resolved (pinned `static`,
    /// identity-mapped region, etc.), and the backend DMAs directly
    /// out of it. The default implementation bounces through
    /// `write_block` so callers are portable; DMA-capable backends
    /// (NVMe, SATA) override it to skip the driver-side memcpy.
    ///
    /// # Safety
    /// - `vptr` must be a valid, readable `[u8; BLOCK_SIZE]` for the
    ///   duration of the call.
    /// - `phys` must be the physical address corresponding to `vptr`
    ///   under the mapper the backend was initialized with.
    /// - The buffer must be quiescent (no concurrent mutation) across
    ///   the call; the backend may DMA from it asynchronously up to
    ///   but not past the next `flush`.
    unsafe fn write_block_from_phys(
        &mut self,
        lba: Lba,
        vptr: *const [u8; BLOCK_SIZE],
        phys: u64,
    ) -> Result<(), Self::Error> {
        // Default: ignore `phys`, treat the buffer like any other
        // source slice. Backends that can DMA should override.
        let _ = phys;
        let slice: &[u8; BLOCK_SIZE] = unsafe { &*vptr };
        self.write_block(lba, slice)
    }

    /// Block until every previously acknowledged write is durable on the
    /// underlying medium. Required for crash recovery to make any guarantee
    /// stronger than "eventually".
    fn flush(&mut self) -> Result<(), Self::Error>;
}
