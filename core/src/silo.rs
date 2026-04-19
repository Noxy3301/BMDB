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
//! A single global `GLOBAL_EPOCH` counter supplies the epoch field in
//! every TID a committing transaction stamps. Its advance mechanism is
//! decoupled: any loop — timer interrupt, idle CPU, a dedicated epoch
//! thread — can call [`advance_epoch`] to bump the counter. The
//! minimum frequency required for group-commit latency targets is on
//! the order of Silo's 40 ms, but correctness does not depend on it.
//!
//! `Record` stores a single 8-byte value alongside the atomic TID.
//! Both words are true Rust atomics: Silo's read protocol (snapshot
//! TID → read value → re-snapshot TID) needs the value access to be a
//! concurrent-safe atomic load, because a non-atomic reader racing an
//! in-flight writer would be undefined behavior in Rust's memory
//! model even if the TID lock bit would rescue the observed bytes.
//! For 8-byte values that is a natural fit: `AtomicU64` on x86_64 is a
//! single `mov`. Values larger than 8 bytes are out of scope for now.

use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Global epoch counter. Starts at 1 so that TID epoch 0 is reserved
/// for "never committed" records and never returned by
/// [`advance_epoch`] during normal operation.
///
/// Wrap-around: at a 40 ms advance cadence (Silo's default) the 32-bit
/// counter lasts ~5.4 years of continuous uptime and then wraps back
/// to `0`, which collides with the reserved "never committed" sentinel.
/// This feasibility-phase tradeoff is called out deliberately; the
/// eventual Verus-verified variant should widen the field to `u64` or
/// clamp explicitly.
static GLOBAL_EPOCH: AtomicU32 = AtomicU32::new(1);

/// Snapshot of the current global epoch. Transactions call this at
/// start (lower bound on the epoch their TID may be stamped with) and
/// at commit (so their TID rises to at least the current value).
#[inline]
pub fn current_epoch() -> u32 {
    GLOBAL_EPOCH.load(Ordering::Acquire)
}

/// Advance the global epoch by one and return the new value.
/// `AcqRel` ordering so a dedicated bumper published writes are
/// observable to later epoch readers, and vice versa — covers the
/// future case where a timer-driven bumper rides on the same
/// synchronization edge as worker commits.
///
/// Driver expectations: call from a single source (a timer handler or
/// a dedicated idle loop) at roughly constant intervals. Multiple
/// concurrent callers are safe but weaken the wall-time relationship.
#[inline]
pub fn advance_epoch() -> u32 {
    // `fetch_add` returns the prior value; we want the new one.
    GLOBAL_EPOCH.fetch_add(1, Ordering::AcqRel).wrapping_add(1)
}

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

// -------------------------------------------------------------------
// Silo-3: per-transaction read and write sets.
// -------------------------------------------------------------------

/// Maximum distinct records a single transaction may touch. Bounds the
/// stack-resident `TxnState` footprint and is a typical Silo parameter
/// (Masstree ships the same constant). Transactions that exceed this
/// abort via [`TxnError::ReadSetOverflow`] / [`TxnError::WriteSetOverflow`].
pub const MAX_RW_SET: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnError {
    ReadSetOverflow,
    WriteSetOverflow,
}

/// One entry of a transaction's read set: the identity of the record
/// plus the TID observed when the value was read. Validation re-loads
/// the record's TID and aborts the transaction if the version moved
/// or the lock bit appeared.
#[derive(Debug, Clone, Copy)]
pub struct ReadEntry {
    /// Record identity as a plain address; the caller guarantees
    /// pointer validity (records live in a long-lived pool / tree).
    pub record_addr: usize,
    pub tid_at_read: Tid,
}

/// One entry of a transaction's write set: which record to update and
/// the value to install on commit. The lock/install sequence in the
/// commit protocol walks this list.
#[derive(Debug, Clone, Copy)]
pub struct WriteEntry {
    pub record_addr: usize,
    pub new_value: u64,
}

impl ReadEntry {
    const EMPTY: Self = Self {
        record_addr: 0,
        tid_at_read: Tid::from_raw(0),
    };
}

impl WriteEntry {
    const EMPTY: Self = Self {
        record_addr: 0,
        new_value: 0,
    };
}

/// Per-transaction scratch state. Sized for the worst case and kept
/// on the owning CPU (no heap, no cross-CPU sharing); a transaction
/// owns one `TxnState` throughout its lifetime, then `reset`s it for
/// the next one.
pub struct TxnState {
    start_epoch: u32,
    read_count: u32,
    read_set: [ReadEntry; MAX_RW_SET],
    write_count: u32,
    write_set: [WriteEntry; MAX_RW_SET],
}

impl TxnState {
    pub const fn new(start_epoch: u32) -> Self {
        Self {
            start_epoch,
            read_count: 0,
            read_set: [ReadEntry::EMPTY; MAX_RW_SET],
            write_count: 0,
            write_set: [WriteEntry::EMPTY; MAX_RW_SET],
        }
    }

    pub fn start_epoch(&self) -> u32 {
        self.start_epoch
    }

    pub fn read_entries(&self) -> &[ReadEntry] {
        &self.read_set[..self.read_count as usize]
    }

    pub fn write_entries(&self) -> &[WriteEntry] {
        &self.write_set[..self.write_count as usize]
    }

    /// Record a read-set entry. Returns `Err(ReadSetOverflow)` when
    /// the fixed-size buffer is full; the state is left unchanged on
    /// that path so the caller can abort cleanly.
    pub fn add_read(&mut self, record: &Record, tid_at_read: Tid) -> Result<(), TxnError> {
        if (self.read_count as usize) >= MAX_RW_SET {
            return Err(TxnError::ReadSetOverflow);
        }
        let idx = self.read_count as usize;
        self.read_set[idx] = ReadEntry {
            record_addr: record as *const Record as usize,
            tid_at_read,
        };
        self.read_count += 1;
        Ok(())
    }

    /// Buffer a write. Actual value publication happens in commit
    /// (Silo-4's validate + install phase).
    pub fn add_write(&mut self, record: &Record, new_value: u64) -> Result<(), TxnError> {
        if (self.write_count as usize) >= MAX_RW_SET {
            return Err(TxnError::WriteSetOverflow);
        }
        let idx = self.write_count as usize;
        self.write_set[idx] = WriteEntry {
            record_addr: record as *const Record as usize,
            new_value,
        };
        self.write_count += 1;
        Ok(())
    }

    /// Clear both sets and stamp a new start epoch so the instance can
    /// be reused for the next transaction without reallocating.
    pub fn reset(&mut self, start_epoch: u32) {
        self.start_epoch = start_epoch;
        self.read_count = 0;
        self.write_count = 0;
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
    fn txn_state_initial_is_empty() {
        let t = TxnState::new(7);
        assert_eq!(t.start_epoch(), 7);
        assert_eq!(t.read_entries().len(), 0);
        assert_eq!(t.write_entries().len(), 0);
    }

    #[test]
    fn txn_state_records_read_snapshots_in_order() {
        let r0 = Record::new(Tid::new(1, 0), 10);
        let r1 = Record::new(Tid::new(2, 0), 20);
        let mut t = TxnState::new(0);
        t.add_read(&r0, Tid::new(1, 0)).unwrap();
        t.add_read(&r1, Tid::new(2, 0)).unwrap();
        let entries = t.read_entries();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].tid_at_read, Tid::new(1, 0));
        assert_eq!(entries[1].tid_at_read, Tid::new(2, 0));
        assert_eq!(entries[0].record_addr, &r0 as *const Record as usize);
        assert_eq!(entries[1].record_addr, &r1 as *const Record as usize);
    }

    #[test]
    fn txn_state_buffers_writes_independently_of_reads() {
        let r0 = Record::new(Tid::new(1, 0), 10);
        let mut t = TxnState::new(0);
        t.add_read(&r0, Tid::new(1, 0)).unwrap();
        t.add_write(&r0, 99).unwrap();
        assert_eq!(t.read_entries().len(), 1);
        assert_eq!(t.write_entries().len(), 1);
        assert_eq!(t.write_entries()[0].new_value, 99);
    }

    #[test]
    fn txn_state_read_overflow_returns_error_and_leaves_state_intact() {
        let r = Record::new(Tid::new(1, 0), 0);
        let mut t = TxnState::new(0);
        for _ in 0..MAX_RW_SET {
            t.add_read(&r, Tid::new(1, 0)).unwrap();
        }
        assert_eq!(
            t.add_read(&r, Tid::new(1, 0)),
            Err(TxnError::ReadSetOverflow),
        );
        // The rejected entry must not have been counted.
        assert_eq!(t.read_entries().len(), MAX_RW_SET);
    }

    #[test]
    fn txn_state_write_overflow_returns_error_and_leaves_state_intact() {
        let r = Record::new(Tid::new(1, 0), 0);
        let mut t = TxnState::new(0);
        for _ in 0..MAX_RW_SET {
            t.add_write(&r, 0).unwrap();
        }
        assert_eq!(t.add_write(&r, 0), Err(TxnError::WriteSetOverflow));
        assert_eq!(t.write_entries().len(), MAX_RW_SET);
    }

    #[test]
    fn txn_state_reset_clears_both_sets_and_updates_start_epoch() {
        let r = Record::new(Tid::new(1, 0), 0);
        let mut t = TxnState::new(0);
        t.add_read(&r, Tid::new(1, 0)).unwrap();
        t.add_write(&r, 99).unwrap();
        t.reset(42);
        assert_eq!(t.start_epoch(), 42);
        assert_eq!(t.read_entries().len(), 0);
        assert_eq!(t.write_entries().len(), 0);
    }

    #[test]
    fn advance_epoch_is_monotonic_and_returns_new_value() {
        // The static counter is shared between tests, so only the
        // delta is meaningful.
        let before = current_epoch();
        let after1 = advance_epoch();
        let after2 = advance_epoch();
        assert!(after1 > before);
        assert_eq!(after2, after1 + 1);
        assert_eq!(current_epoch(), after2);
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
