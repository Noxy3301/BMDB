//! Silo OCC data structures — step 1: TID + Record format.
//!
//! TID layout (Silo SOSP 2013 §4.2, Fig 4):
//!
//! ```text
//! bit:  63 .. 32  31 .. 3    2      1       0
//!       [epoch  ][sequence][lock][latest][absent]
//!         32 b     29 b      1     1        1
//! ```
//!
//! The three status bits live in the low word so that masked `u64`
//! compares order TIDs correctly (Silo's commit protocol compares
//! masked values). `LATEST` is reserved for the future multi-version
//! extension and is unused here; `LOCK` and `ABSENT` are active.
//!
//! `Record` stores a single 8-byte value alongside the atomic TID.
//! Both words are true Rust atomics: Silo's read protocol (snapshot
//! TID → read value → re-snapshot TID) needs the value access to be a
//! concurrent-safe atomic load, because a non-atomic reader racing an
//! in-flight writer would be undefined behavior in Rust's memory
//! model even if the TID lock bit would rescue the observed bytes.
//! For 8-byte values that is a natural fit: `AtomicU64` on x86_64 is a
//! single `mov`. Values larger than 8 bytes are out of scope for now.

use core::sync::atomic::{AtomicU64, Ordering};

/// Transaction identifier, packed in a single `u64`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tid(u64);

impl Tid {
    pub const LOCK: u64 = 1 << 2;
    pub const LATEST: u64 = 1 << 1;
    pub const ABSENT: u64 = 1 << 0;
    pub const STATUS_MASK: u64 = Self::LOCK | Self::LATEST | Self::ABSENT;
    pub const VERSION_MASK: u64 = !Self::STATUS_MASK;

    const EPOCH_SHIFT: u32 = 32;
    const EPOCH_BITS: u32 = 32;
    const SEQ_SHIFT: u32 = 3;
    const SEQ_BITS: u32 = 29;
    const EPOCH_MASK: u64 = ((1u64 << Self::EPOCH_BITS) - 1) << Self::EPOCH_SHIFT;
    const SEQ_MASK: u64 = ((1u64 << Self::SEQ_BITS) - 1) << Self::SEQ_SHIFT;

    /// Construct a TID with clear status bits (lock/latest/absent all
    /// zero). `sequence` must fit in 29 bits.
    pub const fn new(epoch: u32, sequence: u32) -> Self {
        debug_assert!((sequence as u64) < (1u64 << Self::SEQ_BITS));
        let epoch_part = (epoch as u64) << Self::EPOCH_SHIFT;
        let seq_part = ((sequence as u64) << Self::SEQ_SHIFT) & Self::SEQ_MASK;
        Tid(epoch_part | seq_part)
    }

    pub const fn from_raw(raw: u64) -> Self {
        Tid(raw)
    }

    pub const fn raw(self) -> u64 {
        self.0
    }

    pub const fn epoch(self) -> u32 {
        ((self.0 & Self::EPOCH_MASK) >> Self::EPOCH_SHIFT) as u32
    }

    pub const fn sequence(self) -> u32 {
        ((self.0 & Self::SEQ_MASK) >> Self::SEQ_SHIFT) as u32
    }

    pub const fn is_locked(self) -> bool {
        (self.0 & Self::LOCK) != 0
    }

    pub const fn is_absent(self) -> bool {
        (self.0 & Self::ABSENT) != 0
    }

    /// Comparison form: status bits stripped so that two TIDs with the
    /// same epoch+sequence compare equal regardless of a lock bit set
    /// by an in-progress writer.
    pub const fn version(self) -> u64 {
        self.0 & Self::VERSION_MASK
    }
}

/// Silo atomic tuple: `tid` and `value` are both true atomics so any
/// concurrent reader/writer pair is well-defined under Rust's memory
/// model. The value is a bare `u64` and the caller is responsible for
/// its interpretation (the KV layer uses `u64::from_be_bytes` / `to_be_bytes`
/// to move between `u64` and `[u8; 8]`).
#[repr(C, align(64))]
pub struct Record {
    tid: AtomicU64,
    value: AtomicU64,
}

impl Record {
    pub const fn new(initial: Tid, value: u64) -> Self {
        Self {
            tid: AtomicU64::new(initial.raw()),
            value: AtomicU64::new(value),
        }
    }

    /// Snapshot the TID with Acquire ordering.
    pub fn load_tid(&self) -> Tid {
        Tid::from_raw(self.tid.load(Ordering::Acquire))
    }

    /// Silo read protocol:
    /// 1. snapshot TID (fail fast if locked),
    /// 2. atomic-load the value,
    /// 3. re-snapshot TID and require the same version.
    ///
    /// Returns `None` on transient inconsistency; the calling
    /// transaction is expected to abort and retry rather than loop
    /// here.
    pub fn read_snapshot(&self) -> Option<(Tid, u64)> {
        let pre = self.load_tid();
        if pre.is_locked() {
            return None;
        }
        let value = self.value.load(Ordering::Acquire);
        let post = self.load_tid();
        if post.raw() != pre.raw() {
            return None;
        }
        Some((pre, value))
    }

    /// Try to acquire the lock bit via CAS. Returns the observed
    /// pre-lock TID on success — the caller will later either install
    /// a new version (bumping TID) or unlock to restore the old one.
    pub fn try_lock(&self) -> Option<Tid> {
        let mut cur = self.tid.load(Ordering::Relaxed);
        loop {
            if cur & Tid::LOCK != 0 {
                return None;
            }
            match self.tid.compare_exchange_weak(
                cur,
                cur | Tid::LOCK,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(Tid::from_raw(cur)),
                Err(observed) => cur = observed,
            }
        }
    }

    /// Publish a new value under the new TID. Value store happens
    /// before the TID release, so any reader seeing `new_tid` via its
    /// Acquire load will also see `new_value`.
    ///
    /// # Safety
    /// Caller must currently hold the lock (i.e. a preceding
    /// successful `try_lock`).
    pub unsafe fn install(&self, new_tid: Tid, new_value: u64) {
        debug_assert!(!new_tid.is_locked());
        self.value.store(new_value, Ordering::Release);
        self.tid.store(new_tid.raw(), Ordering::Release);
    }

    /// Release the lock without changing the value. Restores
    /// `prior_tid` verbatim — used on abort during validation.
    ///
    /// # Safety
    /// Caller must hold the lock, and `prior_tid` must be the TID
    /// observed from the matching `try_lock`.
    pub unsafe fn unlock(&self, prior_tid: Tid) {
        debug_assert!(!prior_tid.is_locked());
        self.tid.store(prior_tid.raw(), Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tid_epoch_and_sequence_round_trip() {
        let t = Tid::new(0x1234_5678, 0x1ABC_DEF);
        assert_eq!(t.epoch(), 0x1234_5678);
        assert_eq!(t.sequence(), 0x1ABC_DEF);
        assert!(!t.is_locked());
        assert!(!t.is_absent());
    }

    #[test]
    fn tid_status_bits_are_independent() {
        let raw = (0x4200u64 << Tid::EPOCH_SHIFT)
            | (0x1234 << Tid::SEQ_SHIFT)
            | Tid::LOCK
            | Tid::ABSENT;
        let t = Tid::from_raw(raw);
        assert_eq!(t.epoch(), 0x4200);
        assert_eq!(t.sequence(), 0x1234);
        assert!(t.is_locked());
        assert!(t.is_absent());
    }

    #[test]
    fn tid_version_strips_status() {
        let raw = (1u64 << Tid::EPOCH_SHIFT) | Tid::LOCK | Tid::ABSENT | Tid::LATEST;
        let v = Tid::from_raw(raw).version();
        assert_eq!(v, 1u64 << Tid::EPOCH_SHIFT);
    }

    #[test]
    fn record_read_snapshot_returns_value_when_unlocked() {
        let r = Record::new(Tid::new(1, 0), 42);
        let (tid, val) = r.read_snapshot().unwrap();
        assert_eq!(val, 42);
        assert_eq!(tid.epoch(), 1);
    }

    #[test]
    fn record_read_snapshot_fails_while_locked() {
        let r = Record::new(Tid::new(1, 0), 42);
        let _prior = r.try_lock().unwrap();
        assert!(r.read_snapshot().is_none());
    }

    #[test]
    fn record_try_lock_is_exclusive() {
        let r = Record::new(Tid::new(1, 0), 42);
        assert!(r.try_lock().is_some());
        assert!(r.try_lock().is_none());
    }

    #[test]
    fn record_install_publishes_new_value_and_releases_lock() {
        let r = Record::new(Tid::new(1, 0), 42);
        let prior = r.try_lock().unwrap();
        unsafe { r.install(Tid::new(prior.epoch(), prior.sequence() + 1), 99) };
        let (tid, val) = r.read_snapshot().unwrap();
        assert_eq!(val, 99);
        assert_eq!(tid.sequence(), prior.sequence() + 1);
        assert!(!tid.is_locked());
    }

    #[test]
    fn record_unlock_leaves_value_intact() {
        let r = Record::new(Tid::new(1, 5), 42);
        let prior = r.try_lock().unwrap();
        unsafe { r.unlock(prior) };
        let (tid, val) = r.read_snapshot().unwrap();
        assert_eq!(val, 42);
        assert_eq!(tid, prior);
    }

    #[test]
    fn record_concurrent_reads_are_never_torn() {
        // Writer bumps sequence and writes value = sequence. Readers
        // must see (seq == value) on every successful snapshot — a
        // broken protocol would let them observe (new_seq, old_value)
        // or (old_seq, new_value).
        use std::sync::Arc;
        use std::thread;

        let r = Arc::new(Record::new(Tid::new(1, 0), 0));
        let writer_r = Arc::clone(&r);
        let writer = thread::spawn(move || {
            for i in 1..=5_000u64 {
                let prior = loop {
                    if let Some(t) = writer_r.try_lock() {
                        break t;
                    }
                };
                let seq = i as u32;
                unsafe { writer_r.install(Tid::new(prior.epoch(), seq), i) };
            }
        });

        let mut torn = 0u64;
        for _ in 0..1_000_000 {
            if let Some((tid, val)) = r.read_snapshot() {
                if tid.sequence() > 0 && tid.sequence() as u64 != val {
                    torn += 1;
                }
            }
        }
        writer.join().unwrap();
        assert_eq!(torn, 0, "read protocol observed a torn (TID, value) pair");
    }
}
