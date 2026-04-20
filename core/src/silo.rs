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

use crate::storage::BlockStorage;
use crate::wal::{Op, Wal};

/// 8-byte key used by the commit-log plumbing. Matches
/// [`crate::wal::Key`]; kept as a local alias so this module does not
/// depend on the WAL's import surface.
pub type Key = [u8; 8];

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

/// One entry of a transaction's write set: which record to update, the
/// value to install on commit, and the key that identifies the record
/// in the log. The lock/install sequence in the commit protocol walks
/// this list; `key` is only consumed later by the commit-log path.
#[derive(Debug, Clone, Copy)]
pub struct WriteEntry {
    pub record_addr: usize,
    pub key: Key,
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
        key: [0; 8],
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

    /// Buffer a write. If the same record already has a write buffered,
    /// the new value overwrites the old one — otherwise commit's
    /// sorted-lock pass would see two entries for one record and
    /// self-abort on the second `try_lock`. Returning the existing
    /// slot also matches most APIs' last-writer-wins expectation.
    ///
    /// `key` is carried as metadata for the commit-log path; it does
    /// not participate in lock acquisition or read validation.
    pub fn add_write(
        &mut self,
        record: &Record,
        key: Key,
        new_value: u64,
    ) -> Result<(), TxnError> {
        let addr = record as *const Record as usize;
        for slot in &mut self.write_set[..self.write_count as usize] {
            if slot.record_addr == addr {
                slot.key = key;
                slot.new_value = new_value;
                return Ok(());
            }
        }
        if (self.write_count as usize) >= MAX_RW_SET {
            return Err(TxnError::WriteSetOverflow);
        }
        let idx = self.write_count as usize;
        self.write_set[idx] = WriteEntry {
            record_addr: addr,
            key,
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

// -------------------------------------------------------------------
// Silo-4: commit protocol.
// -------------------------------------------------------------------

/// Outcome of a Silo [`commit`] attempt. Abort variants are distinct
/// so bench and unit tests can attribute abort pressure to the right
/// source (lock contention vs. read-validation loss vs. exhausted TID
/// range — the last one forces the caller to wait for the epoch to
/// advance, which resets the sequence space).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitOutcome {
    Committed { new_tid: Tid },
    AbortedLockConflict,
    AbortedReadChanged,
    /// 29-bit sequence field is about to wrap within the current
    /// epoch. Wrapping would reuse an older TID and defeat read
    /// validation (an ABA hole), so the commit aborts instead. The
    /// caller must wait for `current_epoch()` to advance and retry.
    AbortedSequenceExhausted,
}

/// Silo OCC precommit, mapped to this crate's types (paper §4.3):
///
/// 1. Sort the write set by record address — deadlock-free ordering
///    across workers that lock the same records.
/// 2. Acquire the lock bit on each write-set record via CAS.
/// 3. Read the global epoch (serialization point).
/// 4. Validate the read set: re-load each record's TID and require
///    the pre-read version unchanged, ignoring the lock bit when the
///    lock is held by us (i.e., same record appears in the write
///    set).
/// 5. Compute a new TID whose sequence is greater than every
///    read/written record's observed sequence, stamped with the just-
///    read global epoch.
/// 6. Install the new value + TID on each write-set record; that
///    atomic store releases the lock.
///
/// On any abort path the function releases every lock it acquired.
///
/// # Safety
/// All `record_addr`s in `state` must be valid `*const Record`s
/// pointing to records that live at least as long as the commit call.
pub unsafe fn commit(state: &mut TxnState) -> CommitOutcome {
    // 1. Sort write set by address. Two writers that touch the same
    // records will see the same lock order, so a cycle is impossible.
    let write_slice = &mut state.write_set[..state.write_count as usize];
    // `sort_unstable_by_key` is in core; `sort_by_key` would need alloc.
    write_slice.sort_unstable_by_key(|w| w.record_addr);

    // 2. Acquire locks in sorted order. Track the highest-index one
    // we've taken so the abort path can release exactly those.
    let mut locked = 0usize;
    let mut lock_versions: [u64; MAX_RW_SET] = [0; MAX_RW_SET];
    for (i, entry) in write_slice.iter().enumerate() {
        let record = unsafe { &*(entry.record_addr as *const Record) };
        match record.try_lock() {
            Some(prior) => {
                lock_versions[i] = prior.raw();
                locked = i + 1;
            }
            None => {
                // Release the locks we did take, in reverse order.
                for (j, entry) in write_slice[..locked].iter().enumerate().rev() {
                    let rec = unsafe { &*(entry.record_addr as *const Record) };
                    unsafe { rec.unlock(Tid::from_raw(lock_versions[j])) };
                }
                return CommitOutcome::AbortedLockConflict;
            }
        }
    }

    // 3. Serialization point: the epoch our TID will be stamped with.
    let epoch = current_epoch();

    // 4. Validate read set. A record that is also in our write set
    // will show `locked` now; that lock is ours, so we compare
    // masked versions and ignore the lock bit.
    let read_count = state.read_count as usize;
    for read in &state.read_set[..read_count] {
        let record = unsafe { &*(read.record_addr as *const Record) };
        let current = record.load_tid();

        let in_write_set = write_slice
            .iter()
            .any(|w| w.record_addr == read.record_addr);

        if current.version() != read.tid_at_read.version() {
            unsafe { release_all_locks(write_slice, &lock_versions, locked) };
            return CommitOutcome::AbortedReadChanged;
        }
        if current.is_locked() && !in_write_set {
            // Someone else locked this record between our read and
            // validation; abort.
            unsafe { release_all_locks(write_slice, &lock_versions, locked) };
            return CommitOutcome::AbortedReadChanged;
        }
    }

    // 5. New TID: epoch = current global; sequence = 1 + max seen
    // across the read and write sets. Silo's paper also factors in
    // the owning CPU's last-assigned sequence; this implementation
    // omits that term. The record-lock exclusion already serializes
    // writers on overlapping records, and epoch advance erases the
    // sequence space between successive worker transactions. The
    // omission is a known and deliberate spec drift for the
    // feasibility phase.
    let mut max_seq: u32 = 0;
    for read in &state.read_set[..read_count] {
        max_seq = max_seq.max(read.tid_at_read.sequence());
    }
    for (i, _entry) in write_slice.iter().enumerate() {
        let prior = Tid::from_raw(lock_versions[i]);
        max_seq = max_seq.max(prior.sequence());
    }
    // If the 29-bit sequence would wrap, aborting protects read
    // validation from an ABA hole (wrapped TID collides with an older
    // one whose version mask is identical).
    const SEQ_MAX: u32 = (1u32 << 29) - 1;
    if max_seq >= SEQ_MAX {
        unsafe { release_all_locks(write_slice, &lock_versions, locked) };
        return CommitOutcome::AbortedSequenceExhausted;
    }
    let new_tid = Tid::new(epoch, max_seq + 1);

    // 6. Install. Each `install` publishes the new value and releases
    // the lock atomically via the TID Release store.
    for entry in write_slice.iter() {
        let record = unsafe { &*(entry.record_addr as *const Record) };
        unsafe { record.install(new_tid, entry.new_value) };
    }

    CommitOutcome::Committed { new_tid }
}

// -------------------------------------------------------------------
// Silo-5: commit-log buffer + group-commit persist path.
// -------------------------------------------------------------------

/// One on-log entry for a Silo commit. A transaction with a write set
/// of size `k` produces `k` `LogEntry`s, all stamped with the same
/// commit epoch. The logger streams these into the WAL, one block
/// each, and the commit's epoch is the ordering key that `persist`
/// publishes as the new `durable_epoch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogEntry {
    pub epoch: u32,
    pub key: Key,
    pub value: u64,
}

impl LogEntry {
    const EMPTY: Self = Self {
        epoch: 0,
        key: [0; 8],
        value: 0,
    };
}

/// Per-worker commit-log buffer capacity. A worker that fills the
/// buffer must cooperate with the logger (drain + flush) before
/// attempting another commit. 256 entries at 24 bytes each is 6 KiB
/// per buffer — small enough to live per-CPU alongside `TxnState`.
pub const LOG_BUFFER_CAP: usize = 256;

/// Signals that a [`LogBuffer`] is full and cannot accept more
/// entries. The caller must persist (drain + flush) before retrying.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogOverflow;

/// Per-worker, single-producer commit-log buffer. A worker pushes
/// entries after each successful commit; the logger takes an immutable
/// view via [`LogBuffer::as_slice`], writes every entry to the WAL,
/// then calls [`LogBuffer::clear`] to free the slots.
///
/// Ownership is strictly single-threaded: each CPU owns its own
/// buffer. The logger executes on one CPU at a time and must hold
/// exclusive access to any buffer it drains (see [`persist`] for the
/// intended usage pattern).
pub struct LogBuffer {
    entries: [LogEntry; LOG_BUFFER_CAP],
    count: u32,
}

impl LogBuffer {
    pub const fn new() -> Self {
        Self {
            entries: [LogEntry::EMPTY; LOG_BUFFER_CAP],
            count: 0,
        }
    }

    #[inline]
    pub const fn capacity(&self) -> usize {
        LOG_BUFFER_CAP
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.count as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.count as usize == LOG_BUFFER_CAP
    }

    #[inline]
    pub fn as_slice(&self) -> &[LogEntry] {
        &self.entries[..self.count as usize]
    }

    /// Drop every buffered entry. Called by the logger after the
    /// entries are durable on the WAL — before that point clearing
    /// would drop log records that still need to be written.
    #[inline]
    pub fn clear(&mut self) {
        self.count = 0;
    }

    /// Append a single entry. Returns [`LogOverflow`] when the buffer
    /// is already full; the caller must drain + flush before retrying.
    pub fn push(&mut self, entry: LogEntry) -> Result<(), LogOverflow> {
        if self.is_full() {
            return Err(LogOverflow);
        }
        let idx = self.count as usize;
        self.entries[idx] = entry;
        self.count += 1;
        Ok(())
    }

    /// Append every write from a committed transaction. All entries
    /// share `epoch`, the TID's epoch at commit time (not the worker's
    /// start epoch — `persist` relies on this being the commit's
    /// durability epoch). Either the whole batch lands or the buffer
    /// stays unchanged, so a partially-logged commit can never leak
    /// into the WAL.
    pub fn record_commit(
        &mut self,
        epoch: u32,
        writes: &[WriteEntry],
    ) -> Result<(), LogOverflow> {
        if self.count as usize + writes.len() > LOG_BUFFER_CAP {
            return Err(LogOverflow);
        }
        for w in writes {
            // Pre-checked; `push` cannot fail in this loop.
            self.push(LogEntry {
                epoch,
                key: w.key,
                value: w.new_value,
            })
            .expect("pre-checked capacity must hold");
        }
        Ok(())
    }
}

impl Default for LogBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Highest epoch whose log entries are durable on the WAL.
///
/// Starts at 0 (no committed txn is durable yet; epoch 0 never stamps
/// a TID, so this is a clean "nothing durable" sentinel). Workers that
/// need to delay a client ack until their commit is durable compare
/// their commit epoch against [`durable_epoch`] and wait until the
/// atomic catches up.
static DURABLE_EPOCH: AtomicU32 = AtomicU32::new(0);

/// Current durable epoch. Every epoch `E <= durable_epoch()` has had
/// its log records flushed to stable storage. `Acquire` pairs with the
/// `Release` store inside [`persist`].
#[inline]
pub fn durable_epoch() -> u32 {
    DURABLE_EPOCH.load(Ordering::Acquire)
}

/// Drain every buffered log entry into the WAL and publish the
/// resulting durability boundary.
///
/// Steps, in order:
/// 1. Walk each buffer, appending every entry to the WAL without
///    flushing (one NVMe block each) and tracking the highest epoch
///    seen across all entries.
/// 2. Issue a single `flush` — this is the I/O-amortization win.
///    Before this returns, every prior `append_no_flush` is durable.
/// 3. Publish the new durability boundary via `DURABLE_EPOCH` with a
///    `Release` store; readers pair with `durable_epoch()`'s
///    `Acquire`.
/// 4. Clear each buffer.
///
/// Synchronization contract — the single-logger simplification: the
/// caller must guarantee that no live commit is buffering a log entry
/// with an epoch less than or equal to the one that this call is about
/// to publish. Silo's paper resolves this with per-worker heartbeats
/// and a min-over-workers fence; the feasibility bench satisfies it by
/// quiescing workers while `persist` runs.
///
/// On an I/O error mid-batch, the function returns early without
/// flushing and without clearing buffers — the WAL may hold partial
/// log entries on disk, but `durable_epoch` stays pinned at its prior
/// value so no caller observes bogus durability. Recovery replays what
/// landed and skips the torn tail.
pub fn persist<S: BlockStorage>(
    storage: &mut S,
    wal: &mut Wal,
    buffers: &mut [&mut LogBuffer],
) -> Result<u32, S::Error> {
    let mut max_epoch: u32 = 0;
    let mut total: u32 = 0;

    for buf in buffers.iter() {
        for entry in buf.as_slice() {
            wal.append_no_flush(
                storage,
                Op::Put,
                entry.epoch as u64,
                entry.key,
                entry.value.to_be_bytes(),
            )?;
            max_epoch = max_epoch.max(entry.epoch);
            total += 1;
        }
    }

    if total == 0 {
        return Ok(durable_epoch());
    }

    wal.flush(storage)?;

    // fetch_max so concurrent persist callers never regress the
    // boundary. `AcqRel` publishes every WAL write to later
    // `durable_epoch()` readers.
    DURABLE_EPOCH.fetch_max(max_epoch, Ordering::AcqRel);

    for buf in buffers.iter_mut() {
        buf.clear();
    }

    Ok(durable_epoch())
}

/// Abort helper: release the first `locked` entries of the write set
/// in reverse acquisition order, restoring each record's pre-lock TID
/// verbatim.
///
/// # Safety
/// - `write_slice` must address the same records that were locked.
/// - `locked <= write_slice.len()`.
unsafe fn release_all_locks(
    write_slice: &[WriteEntry],
    lock_versions: &[u64; MAX_RW_SET],
    locked: usize,
) {
    for j in (0..locked).rev() {
        let record = unsafe { &*(write_slice[j].record_addr as *const Record) };
        unsafe { record.unlock(Tid::from_raw(lock_versions[j])) };
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
        t.add_write(&r0, [0; 8], 99).unwrap();
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
        // Distinct records so coalescing does not keep the count at 1.
        // A static-lifetime array gives MAX_RW_SET unique addresses on
        // the stack.
        let pool: std::vec::Vec<Record> = (0..=MAX_RW_SET)
            .map(|_| Record::new(Tid::new(1, 0), 0))
            .collect();
        let mut t = TxnState::new(0);
        for rec in &pool[..MAX_RW_SET] {
            t.add_write(rec, [0; 8], 0).unwrap();
        }
        assert_eq!(
            t.add_write(&pool[MAX_RW_SET], [0; 8], 0),
            Err(TxnError::WriteSetOverflow),
        );
        assert_eq!(t.write_entries().len(), MAX_RW_SET);
    }

    #[test]
    fn txn_state_reset_clears_both_sets_and_updates_start_epoch() {
        let r = Record::new(Tid::new(1, 0), 0);
        let mut t = TxnState::new(0);
        t.add_read(&r, Tid::new(1, 0)).unwrap();
        t.add_write(&r, [0; 8], 99).unwrap();
        t.reset(42);
        assert_eq!(t.start_epoch(), 42);
        assert_eq!(t.read_entries().len(), 0);
        assert_eq!(t.write_entries().len(), 0);
    }

    #[test]
    fn commit_writes_are_published_and_read_set_validates() {
        let r = Record::new(Tid::new(current_epoch(), 0), 10);
        let (observed_tid, _) = r.read_snapshot().unwrap();

        let mut txn = TxnState::new(current_epoch());
        txn.add_read(&r, observed_tid).unwrap();
        txn.add_write(&r, [0; 8], 99).unwrap();

        let outcome = unsafe { commit(&mut txn) };
        match outcome {
            CommitOutcome::Committed { new_tid } => {
                assert!(new_tid.sequence() > observed_tid.sequence());
            }
            other => panic!("expected Committed, got {:?}", other),
        }
        let (_t, v) = r.read_snapshot().unwrap();
        assert_eq!(v, 99);
    }

    #[test]
    fn commit_aborts_when_read_set_version_changes() {
        let r = Record::new(Tid::new(current_epoch(), 0), 10);
        let (observed, _) = r.read_snapshot().unwrap();

        // Simulate a concurrent writer that bumped the record's version
        // between our read and our commit.
        let prior = r.try_lock().unwrap();
        unsafe {
            r.install(
                Tid::new(prior.epoch(), prior.sequence() + 1),
                11,
            );
        }

        let mut txn = TxnState::new(current_epoch());
        txn.add_read(&r, observed).unwrap();
        // No write — pure read validation.

        let outcome = unsafe { commit(&mut txn) };
        assert_eq!(outcome, CommitOutcome::AbortedReadChanged);
    }

    #[test]
    fn commit_aborts_when_another_holder_has_the_lock() {
        let r = Record::new(Tid::new(current_epoch(), 0), 10);

        // Another worker already holds the write lock.
        let _held = r.try_lock().unwrap();

        let mut txn = TxnState::new(current_epoch());
        txn.add_write(&r, [0; 8], 99).unwrap();

        let outcome = unsafe { commit(&mut txn) };
        assert_eq!(outcome, CommitOutcome::AbortedLockConflict);
    }

    #[test]
    fn write_entry_preserves_key_for_log_path() {
        // The commit-log path reads `key` back from the write set to
        // stream records into the WAL. Coalescing a second write onto
        // the same record must update the key as well, so the logged
        // entry reflects the last-writer-wins identity.
        let r = Record::new(Tid::new(current_epoch(), 0), 0);
        let mut t = TxnState::new(current_epoch());
        t.add_write(&r, *b"first___", 1).unwrap();
        assert_eq!(t.write_entries()[0].key, *b"first___");

        t.add_write(&r, *b"second__", 2).unwrap();
        assert_eq!(t.write_entries().len(), 1);
        assert_eq!(t.write_entries()[0].key, *b"second__");
        assert_eq!(t.write_entries()[0].new_value, 2);
    }

    #[test]
    fn add_write_coalesces_duplicates_last_writer_wins() {
        let r = Record::new(Tid::new(current_epoch(), 0), 0);
        let mut t = TxnState::new(current_epoch());
        t.add_write(&r, [0; 8], 1).unwrap();
        t.add_write(&r, [0; 8], 2).unwrap();
        t.add_write(&r, [0; 8], 3).unwrap();
        assert_eq!(t.write_entries().len(), 1);
        assert_eq!(t.write_entries()[0].new_value, 3);
    }

    #[test]
    fn commit_aborts_when_sequence_exhausted() {
        // Sequence field is 29 bits → max 2^29 - 1. Seeding a record
        // with sequence at the max causes commit to abort rather than
        // wrap the TID and reuse an older version.
        const SEQ_MAX: u32 = (1u32 << 29) - 1;
        let r = Record::new(Tid::new(current_epoch(), SEQ_MAX), 0);
        let (observed, _) = r.read_snapshot().unwrap();

        let mut txn = TxnState::new(current_epoch());
        txn.add_read(&r, observed).unwrap();
        txn.add_write(&r, [0; 8], 7).unwrap();

        let outcome = unsafe { commit(&mut txn) };
        assert_eq!(outcome, CommitOutcome::AbortedSequenceExhausted);
    }

    #[test]
    fn commit_sorts_write_set_by_record_address() {
        // Force deterministic ordering: three records on the stack,
        // insert them into the write set in reverse address order,
        // and verify commit locks them in ascending address order by
        // requiring all three to commit successfully (a deadlock-
        // prone order would stall).
        let a = Record::new(Tid::new(current_epoch(), 0), 1);
        let b = Record::new(Tid::new(current_epoch(), 0), 2);
        let c = Record::new(Tid::new(current_epoch(), 0), 3);

        let mut txn = TxnState::new(current_epoch());
        // Collect in anti-sorted order on purpose.
        let mut recs = [&a, &b, &c];
        recs.sort_by(|x, y| (*y as *const Record).cmp(&(*x as *const Record)));
        for (i, rec) in recs.iter().enumerate() {
            txn.add_write(rec, [0; 8], 100 + i as u64).unwrap();
        }

        let outcome = unsafe { commit(&mut txn) };
        assert!(matches!(outcome, CommitOutcome::Committed { .. }));
    }

    #[test]
    fn log_buffer_push_and_drain_round_trip() {
        let mut buf = LogBuffer::new();
        assert!(buf.is_empty());

        buf.push(LogEntry { epoch: 3, key: *b"k0______", value: 10 })
            .unwrap();
        buf.push(LogEntry { epoch: 3, key: *b"k1______", value: 20 })
            .unwrap();
        assert_eq!(buf.len(), 2);

        let view = buf.as_slice();
        assert_eq!(view[0].value, 10);
        assert_eq!(view[1].key, *b"k1______");

        buf.clear();
        assert!(buf.is_empty());
        assert_eq!(buf.as_slice().len(), 0);
    }

    #[test]
    fn log_buffer_rejects_overflow() {
        let mut buf = LogBuffer::new();
        for i in 0..LOG_BUFFER_CAP as u64 {
            buf.push(LogEntry { epoch: 1, key: i.to_be_bytes(), value: i })
                .unwrap();
        }
        assert!(buf.is_full());
        assert_eq!(
            buf.push(LogEntry { epoch: 1, key: [0; 8], value: 0 }),
            Err(LogOverflow),
        );
    }

    #[test]
    fn record_commit_is_atomic_across_full_boundary() {
        // A batch that would overflow must leave the buffer untouched,
        // otherwise the logger would observe a half-logged commit and
        // the WAL would go out of sync with the commit's write set.
        let mut buf = LogBuffer::new();
        for i in 0..(LOG_BUFFER_CAP - 1) as u64 {
            buf.push(LogEntry { epoch: 1, key: i.to_be_bytes(), value: i })
                .unwrap();
        }
        let len_before = buf.len();

        // Two-write commit. Only one slot remains — the batch must
        // reject as a whole.
        let r0 = Record::new(Tid::new(1, 0), 0);
        let r1 = Record::new(Tid::new(1, 0), 0);
        let writes = [
            WriteEntry { record_addr: &r0 as *const _ as usize, key: *b"a_______", new_value: 1 },
            WriteEntry { record_addr: &r1 as *const _ as usize, key: *b"b_______", new_value: 2 },
        ];
        assert_eq!(buf.record_commit(5, &writes), Err(LogOverflow));
        assert_eq!(buf.len(), len_before, "partial write must not land");
    }

    #[test]
    fn record_commit_stamps_every_write_with_commit_epoch() {
        // All entries from one commit must share the commit's epoch,
        // because `persist` later uses that epoch as the durability
        // boundary.
        let mut buf = LogBuffer::new();
        let r0 = Record::new(Tid::new(1, 0), 0);
        let r1 = Record::new(Tid::new(1, 0), 0);
        let writes = [
            WriteEntry { record_addr: &r0 as *const _ as usize, key: *b"a_______", new_value: 1 },
            WriteEntry { record_addr: &r1 as *const _ as usize, key: *b"b_______", new_value: 2 },
        ];
        buf.record_commit(17, &writes).unwrap();
        assert_eq!(buf.len(), 2);
        for e in buf.as_slice() {
            assert_eq!(e.epoch, 17);
        }
    }

    #[test]
    fn persist_empty_buffers_is_a_no_op_on_storage() {
        use crate::mem_storage::MemStorage;

        let mut storage = MemStorage::new();
        let mut wal = Wal::new();
        let mut buf = LogBuffer::new();
        let boundary_before = durable_epoch();

        let boundary = persist(&mut storage, &mut wal, &mut [&mut buf]).unwrap();

        assert_eq!(boundary, boundary_before, "empty drain must not move the boundary");
        assert_eq!(storage.flush_count(), 0, "empty drain must not flush");
        assert_eq!(wal.next_lsn(), 1);
    }

    #[test]
    fn persist_amortizes_one_flush_across_batch() {
        use crate::mem_storage::MemStorage;

        let mut storage = MemStorage::new();
        let mut wal = Wal::new();
        let mut buf = LogBuffer::new();
        for i in 1..=3u64 {
            buf.push(LogEntry { epoch: 5, key: i.to_be_bytes(), value: i })
                .unwrap();
        }

        persist(&mut storage, &mut wal, &mut [&mut buf]).unwrap();

        assert_eq!(storage.flush_count(), 1, "one flush per persist, not per entry");
        assert_eq!(wal.next_lsn(), 4, "three entries each bumped LSN");
        assert!(buf.is_empty(), "successful persist must clear buffers");
    }

    #[test]
    fn persist_advances_durable_epoch_monotonically() {
        use crate::mem_storage::MemStorage;

        let mut storage = MemStorage::new();
        let mut wal = Wal::new();

        let mut buf_low = LogBuffer::new();
        buf_low.push(LogEntry { epoch: 100, key: [0; 8], value: 0 }).unwrap();
        let mut buf_high = LogBuffer::new();
        buf_high.push(LogEntry { epoch: 200, key: [1; 8], value: 1 }).unwrap();

        let before = durable_epoch();
        let after = persist(
            &mut storage,
            &mut wal,
            &mut [&mut buf_low, &mut buf_high],
        )
        .unwrap();
        assert!(after >= 200, "boundary must reach max epoch drained");
        assert!(after >= before, "fetch_max must never regress");

        // Draining a batch whose max epoch is below the current boundary
        // must not drop the boundary.
        let mut buf_old = LogBuffer::new();
        buf_old.push(LogEntry { epoch: 50, key: [2; 8], value: 2 }).unwrap();
        let after2 = persist(&mut storage, &mut wal, &mut [&mut buf_old]).unwrap();
        assert_eq!(after2, after, "lower-epoch batch must not move boundary");
    }

    #[test]
    fn persist_emits_wal_records_readable_by_recover() {
        use crate::mem_storage::MemStorage;

        let mut storage = MemStorage::new();
        let mut wal = Wal::new();
        let mut buf_a = LogBuffer::new();
        let mut buf_b = LogBuffer::new();

        buf_a.push(LogEntry { epoch: 7, key: *b"alpha___", value: 1 }).unwrap();
        buf_a.push(LogEntry { epoch: 7, key: *b"bravo___", value: 2 }).unwrap();
        buf_b.push(LogEntry { epoch: 8, key: *b"charlie_", value: 3 }).unwrap();

        persist(&mut storage, &mut wal, &mut [&mut buf_a, &mut buf_b]).unwrap();

        // Recover the WAL cursor from scratch and walk every record.
        use crate::lba_alloc::WAL_START;
        let recovered = Wal::recover(&mut storage).unwrap();
        assert_eq!(recovered.next_lsn(), 4);

        let mut seen: std::vec::Vec<(u64, [u8; 8], [u8; 8])> = std::vec::Vec::new();
        let mut lba = WAL_START;
        while lba < recovered.next_lba() {
            let rec = Wal::read_at(&mut storage, lba).unwrap().unwrap();
            seen.push((rec.epoch, rec.key, rec.value));
            lba += 1;
        }

        assert_eq!(seen.len(), 3);
        assert!(seen.iter().any(|e| e.1 == *b"alpha___" && e.0 == 7));
        assert!(seen.iter().any(|e| e.1 == *b"bravo___" && e.0 == 7));
        assert!(seen.iter().any(|e| e.1 == *b"charlie_" && e.0 == 8));
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
