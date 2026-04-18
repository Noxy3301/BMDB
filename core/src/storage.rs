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

    /// Block until every previously acknowledged write is durable on the
    /// underlying medium. Required for crash recovery to make any guarantee
    /// stronger than "eventually".
    fn flush(&mut self) -> Result<(), Self::Error>;
}
