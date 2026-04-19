//! BMDB core: hardware-independent database logic.
//!
//! Modules in this crate must not depend on any specific driver. They see
//! storage through small traits or plain block-oriented APIs, so the same
//! logic can run against QEMU, real NVMe, or a host-side fake for tests.

#![no_std]

// Pull in `std` only when building tests; the production crate stays `no_std`.
#[cfg(test)]
extern crate std;

pub mod bench;
pub mod bptree;
pub mod kv;
pub mod lba_alloc;
pub mod storage;
pub mod sync;
pub mod wal;

#[cfg(test)]
mod mem_storage;
