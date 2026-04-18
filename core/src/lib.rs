//! BMDB core: hardware-independent database logic.
//!
//! Modules in this crate must not depend on any specific driver. They see
//! storage through small traits or plain block-oriented APIs, so the same
//! logic can run against QEMU, real NVMe, or a host-side fake for tests.

#![no_std]

pub mod lba_alloc;
