//! Transactional engine: key-addressable Silo OCC over an ordered index.
//!
//! This is the coordinator the two lower layers were missing. [`silo`]
//! runs real optimistic transactions but locates tuples by raw address;
//! the index ([`bptree`]) maps keys to slots but knows nothing about
//! concurrency. [`Engine`] binds them: a `key` routes through the index
//! to a slot in a fixed [`Record`] pool, reads snapshot the record into
//! the transaction's read set, writes buffer into its write set, and
//! `commit` runs the Silo precommit protocol over those records.
//!
//! Memory model: every [`Record`] lives for the whole life of the
//! `Engine` (a fixed inline array, used behind a `static` on bare metal
//! or a heap box in tests). Because the pool never moves and never frees
//! a slot, the raw `*const Record` addresses Silo's commit protocol
//! stores in the read/write set stay valid for the transaction's
//! lifetime — which is what makes the `unsafe silo::commit` call safe to
//! wrap here.
//!
//! Keyed slots: the first time a transaction references a key (read *or*
//! write) the engine binds it to a permanent [`Record`] slot and inserts
//! `key -> slot` into the index. A slot starts [absent] and becomes
//! present only when a committed write installs a value. Reading an
//! absent key still joins the read set, so a concurrent insert of that
//! key bumps the slot's version and aborts the reader — point-key
//! phantom protection with no concurrent index yet. Because the first
//! reference (not the first commit) allocates the slot, there is no
//! window where an absent key has no record to validate against.
//!
//! Slots are never reclaimed here: a distinct key consumes a slot for
//! the life of the engine, and an aborted transaction that touched a key
//! never referenced again keeps that key's slot. A retry loop re-runs
//! the same closure over the same keys, so it reuses slots rather than
//! leaking per attempt. The pool is therefore sized for the working set
//! of distinct keys; slot reclamation (EBR) and pool growth are later
//! increments.
//!
//! Durability: [`Engine::transaction_durable`] logs a committed write set
//! plus a commit-boundary record to the WAL and flushes once, so the
//! whole transaction is durable atomically — a crash leaves either the
//! full group or nothing that [`Engine::recover`] will apply. Durable
//! commits are serialized under the WAL lock (held across precommit,
//! append, and flush), so log order equals commit order and no
//! transaction becomes durable ahead of one it read from; recovery then
//! replays fully-committed groups in that order and discards a torn tail.
//! One flush per commit is a correctness-first baseline — batched group
//! commit is a later optimization. The in-memory [`Engine::transaction`]
//! path stays for callers that do not need durability.
//!
//! Scope so far: point `get`/`put`/`delete` (tombstones) with
//! read-your-writes, absent-read validation, a bounded retry loop, and
//! durable multi-write commit with crash recovery. Not yet here: range
//! scans with phantom protection, WAL space reclamation, and the
//! concurrent index.
//!
//! [absent]: Tid::is_absent

use core::sync::atomic::{AtomicU32, Ordering};

use crate::bptree::{BpTree, Key};
use crate::cbptree::Tree as CbTree;
use crate::lba_alloc::WAL_START;
use crate::masstree::Masstree;
use crate::silo::{
    self, CommitOutcome, MAX_RW_SET, Record, Tid, TxnState, current_epoch, ensure_epoch_at_least,
    mark_durable,
};
use crate::storage::BlockStorage;
use crate::sync::SpinLock;
use crate::wal::{Op, Wal};

/// Tuples the engine can hold. Bounds the static footprint (each
/// [`Record`] is a 64-byte cache line, so this array is
/// `ENGINE_RECORDS * 64` bytes). The index may run out of nodes before
/// this cap is hit; both limits surface as [`EngineError::OutOfSpace`].
pub const ENGINE_RECORDS: usize = 512;

/// Sentinel `cpu` for callers with no dedicated core (single-threaded use,
/// recovery, the default `begin`/`transaction`). A concurrent index treats
/// any value >= its CPU count as "the shared allocation path", so this both
/// selects that path and never indexes a real per-cpu slot.
pub const NO_CPU: usize = usize::MAX;

/// Ordered key → record-slot map. Both methods take `&self`: a
/// concurrent index (see [`CbTreeIndex`]) resolves lookups lock-free and
/// serializes writers internally, so the engine keeps no lock on the read
/// path. The sequential [`BpTreeIndex`] wraps its tree in a lock to honor
/// the same shape.
pub trait Index {
    /// Slot currently mapped to `key`, or `None` if unmapped.
    fn get(&self, key: Key) -> Option<u32>;
    /// Bind `key` to `slot` if it is unmapped, else return the slot it is
    /// already bound to -- an ATOMIC get-or-insert, so concurrent first-
    /// touchers of the same key agree on one slot without a shared lock.
    /// `cpu` is the caller's dedicated core index for per-cpu allocation
    /// (or a value >= the index's CPU count for the shared path).
    /// [`IndexFull`] if the index cannot grow to hold another key.
    fn get_or_insert(&self, cpu: usize, key: Key, slot: u32) -> Result<u32, IndexFull>;
}

/// The index cannot accept another key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexFull;

/// [`Index`] backed by the sequential B+tree, serialized behind a lock so
/// it can present a `&self` writer. The tree's 8-byte value slot carries
/// the record index as a big-endian `u64`. Correct but fully serial —
/// [`CbTreeIndex`] is the concurrent path.
pub struct BpTreeIndex {
    tree: SpinLock<BpTree>,
}

impl BpTreeIndex {
    pub const fn new() -> Self {
        Self {
            tree: SpinLock::new(BpTree::new()),
        }
    }
}

impl Default for BpTreeIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl Index for BpTreeIndex {
    fn get(&self, key: Key) -> Option<u32> {
        self.tree.lock().lookup(key).map(|v| u64::from_be_bytes(v) as u32)
    }

    fn get_or_insert(&self, _cpu: usize, key: Key, slot: u32) -> Result<u32, IndexFull> {
        // The lock spans get + insert, so the get-or-insert is atomic.
        let mut tree = self.tree.lock();
        if let Some(v) = tree.lookup(key) {
            return Ok(u64::from_be_bytes(v) as u32);
        }
        tree.upsert(key, (slot as u64).to_be_bytes())
            .map(|_| slot)
            .map_err(|_| IndexFull)
    }
}

/// [`Index`] backed by the lock-free concurrent B+tree. Lookups run
/// without any engine lock; writers serialize on the tree's internal
/// writer lock. The 8-byte value slot carries the record index directly
/// as a `u64`.
pub struct CbTreeIndex {
    tree: CbTree,
}

impl CbTreeIndex {
    pub const fn new() -> Self {
        Self { tree: CbTree::new() }
    }
}

impl Default for CbTreeIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl Index for CbTreeIndex {
    fn get(&self, key: Key) -> Option<u32> {
        self.tree.lookup(&key).map(|v| v as u32)
    }

    fn get_or_insert(&self, _cpu: usize, key: Key, slot: u32) -> Result<u32, IndexFull> {
        self.tree
            .get_or_insert(key, slot as u64)
            .map(|v| v as u32)
            .map_err(|_| IndexFull)
    }
}

/// [`Index`] backed by the faithful concurrent Masstree. Like
/// [`CbTreeIndex`] its lookups take no engine lock, but writers lock only
/// the nodes they touch instead of a single tree-wide writer lock, so
/// inserts of disjoint keys proceed in parallel. The 8-byte value slot
/// carries the record index directly as a `u64`.
pub struct MasstreeIndex {
    tree: Masstree,
}

impl MasstreeIndex {
    pub const fn new() -> Self {
        Self { tree: Masstree::new() }
    }
}

impl Default for MasstreeIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl Index for MasstreeIndex {
    fn get(&self, key: Key) -> Option<u32> {
        self.tree.get(key).map(|v| v as u32)
    }

    fn get_or_insert(&self, cpu: usize, key: Key, slot: u32) -> Result<u32, IndexFull> {
        self.tree
            .get_or_put_on(cpu, key, slot as u64)
            .map(|v| v as u32)
            .map_err(|_| IndexFull)
    }
}

/// Reason a transaction operation could not proceed. Distinct from a
/// commit *abort* (see [`CommitOutcome`]): these are terminal for the
/// attempt and are not retried by [`Engine::transaction`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineError {
    /// The transaction's read or write set is full ([`silo::MAX_RW_SET`]).
    TxnTooLarge,
    /// The record pool or the index is full; the key has no slot.
    OutOfSpace,
}

/// A key-addressable transactional store over Silo OCC.
pub struct Engine<I: Index = BpTreeIndex> {
    records: [Record; ENGINE_RECORDS],
    /// Key → slot map. Read off the hot path with no engine lock; the
    /// concurrent index makes lookups lock-free.
    index: I,
    /// Next unused record slot. A brand-new key claims one with a saturating
    /// atomic bump, then binds it through the index's atomic get-or-insert --
    /// no lock, so first-touches of disjoint keys never serialize. Existing
    /// keys never touch this at all (they resolve through `index` alone).
    next_slot: AtomicU32,
    /// Write-ahead log cursor for durable commits. Guarded so concurrent
    /// durable commits serialize their append + flush.
    wal: SpinLock<Wal>,
}

/// A fresh slot is `absent`: no key maps to it yet, and a reader that
/// reaches one (e.g. after an aborted insert left an index entry) sees
/// `None` rather than a zero value.
const EMPTY_RECORDS: [Record; ENGINE_RECORDS] =
    [const { Record::new(Tid::from_raw(Tid::ABSENT), 0) }; ENGINE_RECORDS];

impl Engine<BpTreeIndex> {
    /// Construct an empty engine on the sequential index. `const` so it
    /// can back a `static` on bare metal without a runtime initializer.
    pub const fn new() -> Self {
        Self {
            records: EMPTY_RECORDS,
            index: BpTreeIndex::new(),
            next_slot: AtomicU32::new(0),
            wal: SpinLock::new(Wal::new()),
        }
    }
}

impl Engine<CbTreeIndex> {
    /// Construct an empty engine on the lock-free concurrent index.
    pub const fn concurrent() -> Self {
        Self {
            records: EMPTY_RECORDS,
            index: CbTreeIndex::new(),
            next_slot: AtomicU32::new(0),
            wal: SpinLock::new(Wal::new()),
        }
    }
}

impl Engine<MasstreeIndex> {
    /// Construct an empty engine on the fine-grained-locking Masstree index,
    /// where inserts of disjoint keys do not serialize on one writer lock.
    pub const fn masstree() -> Self {
        Self {
            records: EMPTY_RECORDS,
            index: MasstreeIndex::new(),
            next_slot: AtomicU32::new(0),
            wal: SpinLock::new(Wal::new()),
        }
    }
}

impl Default for Engine<BpTreeIndex> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: Index> Engine<I> {
    /// Begin a transaction stamped with the current epoch, with no dedicated
    /// core (the index allocates new-key slots from its shared path).
    pub fn begin(&self) -> Txn<'_, I> {
        self.begin_on(NO_CPU)
    }

    /// Begin a transaction whose new-key slot allocations route to `cpu`'s
    /// private index pool. `cpu` must be unique to the calling thread (or
    /// [`NO_CPU`]) for the duration -- the same contract the concurrent index
    /// requires of a per-cpu index.
    pub fn begin_on(&self, cpu: usize) -> Txn<'_, I> {
        Txn {
            engine: self,
            state: TxnState::new(current_epoch()),
            cpu,
        }
    }

    /// Run `body` as a transaction, retrying on OCC abort up to
    /// `max_attempts` times. Returns the body's value on commit, or
    /// `None` if every attempt aborted or the body returned an error
    /// (a full set / out-of-space is terminal, not retried).
    pub fn transaction<F, R>(&self, max_attempts: u32, body: F) -> Option<R>
    where
        F: FnMut(&mut Txn<'_, I>) -> Result<R, EngineError>,
    {
        self.transaction_on(NO_CPU, max_attempts, body)
    }

    /// [`transaction`](Self::transaction) with new-key slot allocation routed
    /// to `cpu`'s private index pool (see [`begin_on`](Self::begin_on)).
    pub fn transaction_on<F, R>(&self, cpu: usize, max_attempts: u32, mut body: F) -> Option<R>
    where
        F: FnMut(&mut Txn<'_, I>) -> Result<R, EngineError>,
    {
        for _ in 0..max_attempts {
            let mut txn = self.begin_on(cpu);
            let value = match body(&mut txn) {
                Ok(v) => v,
                Err(_) => return None,
            };
            match txn.commit() {
                CommitOutcome::Committed { .. } => return Some(value),
                // Any abort: drop the attempt and retry with a fresh
                // read/write set and epoch snapshot.
                _ => continue,
            }
        }
        None
    }

    /// Run `body` as a durable transaction, retrying on OCC abort up to
    /// `max_attempts` times. On a committed attempt the write set and a
    /// commit-boundary record are flushed to `storage` before returning,
    /// so the result is durable. Returns `Ok(Some(value))` on a durable
    /// commit, `Ok(None)` if every attempt aborted or the body errored,
    /// or `Err` on a storage failure.
    pub fn transaction_durable<S, F, R>(
        &self,
        storage: &mut S,
        max_attempts: u32,
        mut body: F,
    ) -> Result<Option<R>, S::Error>
    where
        S: BlockStorage,
        F: FnMut(&mut Txn<'_, I>) -> Result<R, EngineError>,
    {
        for _ in 0..max_attempts {
            let mut txn = self.begin();
            let value = match body(&mut txn) {
                Ok(v) => v,
                Err(_) => return Ok(None),
            };
            match txn.commit_durable(storage)? {
                Some(_tid) => return Ok(Some(value)),
                None => continue,
            }
        }
        Ok(None)
    }

    /// Rebuild the record pool, index, and WAL cursor from the durable
    /// log. Every fully-committed transaction group is replayed in commit
    /// order (each group's records are contiguous, so log order is commit
    /// order); a torn tail — writes whose commit record never landed — is
    /// discarded, making recovery atomic per transaction. The global and
    /// durable epochs are advanced past the highest recovered epoch.
    ///
    /// Run on a fresh engine, on a single CPU, before any transaction.
    pub fn recover<S: BlockStorage>(&self, storage: &mut S) -> Result<(), S::Error> {
        let scanned = Wal::recover(storage)?;
        let end = scanned.next_lba();

        // Writes of the commit group currently being accumulated. A group
        // is at most one full write set.
        let mut pending: [(Key, u64, bool); MAX_RW_SET] = [([0; 8], 0, false); MAX_RW_SET];
        let mut pending_len = 0usize;
        let mut max_epoch = 0u32;
        // Cursor just past the last *accepted* commit boundary. Orphan
        // records after it (a torn group's writes whose commit never
        // landed) are not durable, so the next append overwrites them.
        let mut durable_lba = WAL_START;
        let mut durable_lsn = 1u64;

        let mut lba = WAL_START;
        while lba < end {
            let rec = Wal::read_at(storage, lba)?
                .expect("record within the recovered prefix must decode");
            match rec.op() {
                Some(Op::Put) => {
                    if pending_len < MAX_RW_SET {
                        pending[pending_len] = (rec.key, u64::from_be_bytes(rec.value), false);
                        pending_len += 1;
                    }
                }
                Some(Op::Delete) => {
                    if pending_len < MAX_RW_SET {
                        pending[pending_len] = (rec.key, 0, true);
                        pending_len += 1;
                    }
                }
                Some(Op::Commit) => {
                    let count = u64::from_be_bytes(rec.value) as usize;
                    let tid = Tid::from_raw(rec.epoch);
                    // A count mismatch means writes were lost between this
                    // record and its group — skip the whole group.
                    if count == pending_len {
                        for &(key, value, absent) in &pending[..pending_len] {
                            self.recover_apply(key, value, absent, tid);
                        }
                        max_epoch = max_epoch.max(tid.epoch());
                        durable_lba = lba + 1;
                        durable_lsn = rec.lsn + 1;
                    }
                    pending_len = 0;
                }
                None => break,
            }
            lba += 1;
        }
        // Any leftover pending writes have no trailing commit record —
        // a torn tail — and are dropped. Resume appends right after the
        // last durable commit so those orphan records are overwritten and
        // cannot corrupt a later recovery's group boundaries.
        let mut wal = Wal::new();
        wal.restore((durable_lba, durable_lsn));
        *self.wal.lock() = wal;

        ensure_epoch_at_least(max_epoch.saturating_add(1));
        mark_durable(max_epoch);
        Ok(())
    }

    /// Install a recovered committed write into its record + index,
    /// bypassing the OCC protocol. Later groups overwrite earlier ones
    /// for the same key (last committed wins).
    fn recover_apply(&self, key: Key, value: u64, absent: bool, tid: Tid) {
        let Some(slot) = self.slot_for(NO_CPU, key) else {
            return; // recovered working set exceeds the pool; drop the tail
        };
        let record = self.record(slot);
        let install_tid = if absent {
            Tid::from_raw(tid.raw() | Tid::ABSENT)
        } else {
            tid
        };
        // Safety: recovery runs single-threaded before the store opens,
        // so there is no concurrent reader or writer of this record.
        unsafe { record.restore(install_tid, value) };
    }

    /// Slot for `key`, allocating a fresh record if the key is new.
    ///
    /// The common case — a key already in the index — resolves lock-free
    /// through `index.get` with no engine lock held. A brand-new key claims a
    /// tentative slot with a saturating atomic bump (never past
    /// `ENGINE_RECORDS`, so it can't wrap and reissue a live slot), then binds
    /// it through the index's ATOMIC get-or-insert: if another thread bound
    /// the key first, get-or-insert returns that slot and our tentative one is
    /// simply left unused (a fresh, absent record). So a key is never assigned
    /// two slots, and disjoint-key first-touches never serialize on a lock.
    /// `cpu` routes the index's node allocation to that core's private pool.
    fn slot_for(&self, cpu: usize, key: Key) -> Option<u32> {
        if let Some(slot) = self.index.get(key) {
            return Some(slot);
        }
        let tentative = self
            .next_slot
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                if n as usize >= ENGINE_RECORDS {
                    None
                } else {
                    Some(n + 1)
                }
            })
            .ok()?;
        self.index.get_or_insert(cpu, key, tentative).ok()
    }

    fn record(&self, slot: u32) -> &Record {
        &self.records[slot as usize]
    }
}

/// A live transaction. Owns its [`TxnState`] and borrows the engine so
/// the record pool cannot move while raw record addresses sit in the
/// read/write set.
pub struct Txn<'e, I: Index> {
    engine: &'e Engine<I>,
    state: TxnState,
    /// Dedicated core for new-key slot allocation, or [`NO_CPU`].
    cpu: usize,
}

impl<'e, I: Index> Txn<'e, I> {
    /// Read the value bound to `key`.
    ///
    /// Read-your-writes: a value this transaction already `put` is
    /// returned from the write set without touching the record (an
    /// uncommitted write is never validated against itself).
    ///
    /// Otherwise the key is bound to its (possibly freshly allocated)
    /// slot and the record is snapshotted into the read set — including
    /// an absent record, so a concurrent insert of `key` invalidates
    /// this read at commit time. An absent slot reads as `None`.
    pub fn get(&mut self, key: Key) -> Result<Option<u64>, EngineError> {
        for w in self.state.write_entries() {
            if w.key == key {
                // A tombstone this transaction buffered reads back as gone.
                return Ok(if w.absent { None } else { Some(w.new_value) });
            }
        }
        // Allocate a stable slot even for an absent key so this read can
        // be validated: without a record, a later insert of `key` would
        // be an undetectable phantom.
        let slot = self.engine.slot_for(self.cpu, key).ok_or(EngineError::OutOfSpace)?;
        let record = self.engine.record(slot);
        // A clean snapshot is preferred; under a racing writer fall back
        // to a forced load, which seeds the read set with the (possibly
        // locked) TID so validation aborts this doomed transaction.
        let (tid, value) = record.read_snapshot().unwrap_or_else(|| record.load_forced());
        self.state
            .add_read(record, tid)
            .map_err(|_| EngineError::TxnTooLarge)?;
        if tid.is_absent() {
            return Ok(None);
        }
        Ok(Some(value))
    }

    /// Buffer a write of `value` to `key`, allocating a record slot for
    /// a new key. The write becomes visible only after [`commit`].
    ///
    /// [`commit`]: Txn::commit
    pub fn put(&mut self, key: Key, value: u64) -> Result<(), EngineError> {
        let slot = self.engine.slot_for(self.cpu, key).ok_or(EngineError::OutOfSpace)?;
        let record = self.engine.record(slot);
        self.state
            .add_write(record, key, value)
            .map_err(|_| EngineError::TxnTooLarge)
    }

    /// Buffer a delete of `key`, recorded as a tombstone write. On commit
    /// the key's record is published absent, so a later `get` returns
    /// `None` and any concurrent reader of the pre-delete value aborts.
    /// Deleting an absent key is a no-op that still advances the record's
    /// version.
    pub fn delete(&mut self, key: Key) -> Result<(), EngineError> {
        let slot = self.engine.slot_for(self.cpu, key).ok_or(EngineError::OutOfSpace)?;
        let record = self.engine.record(slot);
        self.state
            .add_delete(record, key)
            .map_err(|_| EngineError::TxnTooLarge)
    }

    /// Run the Silo precommit protocol over the buffered read/write set
    /// and consume the transaction. A read-only transaction commits iff
    /// its read set still validates. The result is in-memory only; use
    /// [`Txn::commit_durable`] to make it survive a crash.
    pub fn commit(mut self) -> CommitOutcome {
        // Safety: every record address in the sets points into
        // `engine.records`, which outlives `self` (the `&'e` borrow) and
        // never moves or frees a slot.
        unsafe { silo::commit(&mut self.state) }
    }

    /// Commit and make the write set durable before returning.
    ///
    /// On OCC success the write set plus a commit-boundary record are
    /// appended to the WAL and flushed once, so the group is durable
    /// atomically. Returns `Ok(Some(tid))` on a durable commit,
    /// `Ok(None)` on an OCC abort (retry), or `Err` on a storage failure.
    ///
    /// The WAL lock is held across the whole commit — Silo precommit,
    /// append, and flush — so durable commits are serialized. That makes
    /// log (LBA) order equal commit order, which recovery relies on, and
    /// it means a transaction cannot become durable while a commit it may
    /// have read from is not: the earlier commit holds this lock until
    /// its own flush completes. (In the current API the exclusive
    /// `&mut storage` borrow already serializes durable commits; holding
    /// the lock across the install makes the ordering invariant explicit
    /// and robust to a future shared-storage path.)
    ///
    /// Ordering note: Silo installs the new versions in memory before
    /// this logs them, so a crash before the flush loses the in-memory
    /// effect too and memory stays consistent with disk. A storage
    /// *error* (not a crash) after the install leaves memory ahead of the
    /// log; the caller should treat that as a fatal durability fault.
    ///
    /// Throughput note: one flush per commit and full serialization are a
    /// correctness-first baseline; Silo's batched group commit (amortize
    /// one flush across many transactions via per-worker log buffers and
    /// an epoch fence) is a later optimization.
    pub fn commit_durable<S: BlockStorage>(
        mut self,
        storage: &mut S,
    ) -> Result<Option<Tid>, S::Error> {
        let mut wal = self.engine.wal.lock();

        let new_tid = match unsafe { silo::commit(&mut self.state) } {
            CommitOutcome::Committed { new_tid } => new_tid,
            _ => return Ok(None),
        };

        let writes = self.state.write_entries();
        for w in writes {
            let op = if w.absent { Op::Delete } else { Op::Put };
            wal.append_no_flush(storage, op, new_tid.raw(), w.key, w.new_value.to_be_bytes())?;
        }
        // Commit boundary: `value` carries the write count so recovery
        // can confirm the whole group landed.
        wal.append_no_flush(
            storage,
            Op::Commit,
            new_tid.raw(),
            [0; 8],
            (writes.len() as u64).to_be_bytes(),
        )?;
        wal.flush(storage)?;

        // This commit and every commit ordered before it are now durable,
        // so publishing the epoch boundary here is sound.
        mark_durable(new_tid.epoch());
        Ok(Some(new_tid))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::boxed::Box;

    fn k(i: u64) -> Key {
        i.to_be_bytes()
    }

    #[test]
    fn three_key_txn_commits_and_reads_back() {
        let engine = Box::new(Engine::new());
        let ok = engine.transaction(4, |txn| {
            txn.put(k(1), 100)?;
            txn.put(k(2), 200)?;
            txn.put(k(3), 300)?;
            Ok(())
        });
        assert_eq!(ok, Some(()), "3-key write txn must commit");

        let sum = engine.transaction(4, |txn| {
            let a = txn.get(k(1))?.unwrap();
            let b = txn.get(k(2))?.unwrap();
            let c = txn.get(k(3))?.unwrap();
            Ok(a + b + c)
        });
        assert_eq!(sum, Some(600));
    }

    #[test]
    fn read_your_writes_within_a_txn() {
        let engine = Box::new(Engine::new());
        let mut txn = engine.begin();
        txn.put(k(7), 42).unwrap();
        // Visible to this txn before commit, straight from the write set.
        assert_eq!(txn.get(k(7)).unwrap(), Some(42));
        // Overwrite in the same txn — last write wins.
        txn.put(k(7), 43).unwrap();
        assert_eq!(txn.get(k(7)).unwrap(), Some(43));
        assert!(matches!(txn.commit(), CommitOutcome::Committed { .. }));
    }

    #[test]
    fn interleaved_write_aborts_a_readers_commit() {
        let engine = Box::new(Engine::new());
        // Seed key 5.
        assert!(engine.transaction(4, |t| t.put(k(5), 1)).is_some());

        // Reader observes key 5 into its read set.
        let mut reader = engine.begin();
        assert_eq!(reader.get(k(5)).unwrap(), Some(1));

        // A concurrent transaction commits a new value for key 5,
        // bumping its version.
        assert!(engine.transaction(4, |t| t.put(k(5), 2)).is_some());

        // The reader's commit must now fail validation.
        assert_eq!(reader.commit(), CommitOutcome::AbortedReadChanged);
    }

    #[test]
    fn absent_read_then_concurrent_insert_aborts_the_reader() {
        // Serializability of a read-absent transaction: a reader observes
        // key 8 as absent, a concurrent transaction inserts it, and the
        // reader's commit must fail — there is no serial order in which
        // the reader both saw None and the insert happened before it.
        let engine = Box::new(Engine::new());

        let mut reader = engine.begin();
        assert_eq!(reader.get(k(8)).unwrap(), None, "key 8 starts absent");

        // A concurrent transaction inserts key 8.
        assert!(engine.transaction(4, |t| t.put(k(8), 5)).is_some());

        // The reader tried to build on "key 8 absent", which no longer
        // holds: its commit must abort.
        assert_eq!(reader.commit(), CommitOutcome::AbortedReadChanged);
    }

    #[test]
    fn absent_key_reads_none_and_overwrite_returns_latest() {
        let engine = Box::new(Engine::new());
        let got = engine.transaction(4, |t| t.get(k(99)));
        assert_eq!(got, Some(None), "never-written key reads None");

        engine.transaction(4, |t| t.put(k(1), 10)).unwrap();
        engine.transaction(4, |t| t.put(k(1), 20)).unwrap();
        let v = engine.transaction(4, |t| t.get(k(1))).unwrap();
        assert_eq!(v, Some(20), "overwrite is visible after commit");
    }

    #[test]
    fn delete_removes_a_committed_key() {
        let engine = Box::new(Engine::new());
        engine.transaction(4, |t| t.put(k(3), 30)).unwrap();
        assert_eq!(engine.transaction(4, |t| t.get(k(3))).unwrap(), Some(30));

        assert!(engine.transaction(4, |t| t.delete(k(3))).is_some());
        assert_eq!(engine.transaction(4, |t| t.get(k(3))).unwrap(), None);
    }

    #[test]
    fn read_your_deletes_and_put_delete_coalesce() {
        let engine = Box::new(Engine::new());
        engine.transaction(4, |t| t.put(k(1), 1)).unwrap();

        // Within one transaction: delete is visible to a later get, and a
        // put-then-delete on the same key lands as a single tombstone.
        let ok = engine.transaction(4, |t| {
            t.delete(k(1))?;
            assert_eq!(t.get(k(1))?, None, "read-your-deletes");
            t.put(k(2), 2)?;
            t.delete(k(2))?;
            assert_eq!(t.get(k(2))?, None, "put then delete reads as gone");
            Ok(())
        });
        assert!(ok.is_some());
        assert_eq!(engine.transaction(4, |t| t.get(k(1))).unwrap(), None);
        assert_eq!(engine.transaction(4, |t| t.get(k(2))).unwrap(), None);
    }

    #[test]
    fn concurrent_delete_aborts_a_readers_commit() {
        let engine = Box::new(Engine::new());
        engine.transaction(4, |t| t.put(k(5), 9)).unwrap();

        let mut reader = engine.begin();
        assert_eq!(reader.get(k(5)).unwrap(), Some(9));

        // A concurrent transaction deletes key 5.
        assert!(engine.transaction(4, |t| t.delete(k(5))).is_some());

        assert_eq!(reader.commit(), CommitOutcome::AbortedReadChanged);
    }

    #[test]
    fn read_only_txn_commits_when_nothing_changed() {
        let engine = Box::new(Engine::new());
        engine.transaction(4, |t| t.put(k(1), 5)).unwrap();
        let mut txn = engine.begin();
        assert_eq!(txn.get(k(1)).unwrap(), Some(5));
        assert!(matches!(txn.commit(), CommitOutcome::Committed { .. }));
    }

    #[test]
    fn multi_key_commit_is_atomic_across_reads() {
        // Both keys move to the new epoch together: a reader sees either
        // both old or both new, never a mix (checked by equal values).
        let engine = Box::new(Engine::new());
        engine
            .transaction(4, |t| {
                t.put(k(1), 0)?;
                t.put(k(2), 0)?;
                Ok(())
            })
            .unwrap();
        engine
            .transaction(4, |t| {
                t.put(k(1), 7)?;
                t.put(k(2), 7)?;
                Ok(())
            })
            .unwrap();
        let pair = engine
            .transaction(4, |t| Ok((t.get(k(1))?.unwrap(), t.get(k(2))?.unwrap())))
            .unwrap();
        assert_eq!(pair, (7, 7));
    }

    // ---- durability + crash recovery (MemStorage) --------------------

    use crate::lba_alloc::BLOCK_SIZE;
    use crate::mem_storage::MemStorage;

    fn garbage_block() -> [u8; BLOCK_SIZE] {
        let mut b = [0u8; BLOCK_SIZE];
        b[0] = 0x55; // non-magic first byte → invalid record
        b
    }

    #[test]
    fn durable_commit_persists_and_recovers() {
        let mut storage = MemStorage::new();
        let engine = Box::new(Engine::new());
        let r = engine
            .transaction_durable(&mut storage, 4, |t| {
                t.put(k(1), 100)?;
                t.put(k(2), 200)?;
                t.put(k(3), 300)?;
                Ok(())
            })
            .unwrap();
        assert_eq!(r, Some(()));

        // Crash: a fresh engine rebuilds its state from the log.
        let recovered = Box::new(Engine::new());
        recovered.recover(&mut storage).unwrap();
        assert_eq!(recovered.transaction(4, |t| t.get(k(1))).unwrap(), Some(100));
        assert_eq!(recovered.transaction(4, |t| t.get(k(2))).unwrap(), Some(200));
        assert_eq!(recovered.transaction(4, |t| t.get(k(3))).unwrap(), Some(300));
    }

    #[test]
    fn torn_commit_boundary_discards_the_whole_group() {
        let mut storage = MemStorage::new();
        let engine = Box::new(Engine::new());
        engine
            .transaction_durable(&mut storage, 4, |t| {
                t.put(k(1), 100)?;
                t.put(k(2), 200)?;
                t.put(k(3), 300)?;
                Ok(())
            })
            .unwrap();

        // The group is [put, put, put, commit] at WAL_START..WAL_START+4.
        // Losing the commit boundary must drop all three writes.
        storage.force_write(WAL_START + 3, garbage_block());

        let recovered = Box::new(Engine::new());
        recovered.recover(&mut storage).unwrap();
        assert_eq!(recovered.transaction(4, |t| t.get(k(1))).unwrap(), None);
        assert_eq!(recovered.transaction(4, |t| t.get(k(2))).unwrap(), None);
        assert_eq!(recovered.transaction(4, |t| t.get(k(3))).unwrap(), None);
    }

    #[test]
    fn torn_write_in_group_discards_the_whole_group() {
        let mut storage = MemStorage::new();
        let engine = Box::new(Engine::new());
        engine
            .transaction_durable(&mut storage, 4, |t| {
                t.put(k(1), 100)?;
                t.put(k(2), 200)?;
                Ok(())
            })
            .unwrap();

        // Tear the second write: recovery stops at the gap, so it never
        // reaches the commit record and the first write is dropped too.
        storage.force_write(WAL_START + 1, garbage_block());

        let recovered = Box::new(Engine::new());
        recovered.recover(&mut storage).unwrap();
        assert_eq!(recovered.transaction(4, |t| t.get(k(1))).unwrap(), None);
        assert_eq!(recovered.transaction(4, |t| t.get(k(2))).unwrap(), None);
    }

    #[test]
    fn earlier_committed_group_survives_a_torn_later_group() {
        let mut storage = MemStorage::new();
        let engine = Box::new(Engine::new());
        // First durable commit: 1 write + boundary at WAL_START..+2.
        engine
            .transaction_durable(&mut storage, 4, |t| t.put(k(1), 11))
            .unwrap();
        // Second durable commit: 1 write + boundary at WAL_START+2..+4.
        engine
            .transaction_durable(&mut storage, 4, |t| t.put(k(2), 22))
            .unwrap();

        // Tear the second group's commit boundary (at WAL_START+3).
        storage.force_write(WAL_START + 3, garbage_block());

        let recovered = Box::new(Engine::new());
        recovered.recover(&mut storage).unwrap();
        assert_eq!(recovered.transaction(4, |t| t.get(k(1))).unwrap(), Some(11));
        assert_eq!(recovered.transaction(4, |t| t.get(k(2))).unwrap(), None);
    }

    #[test]
    fn durable_commit_after_recovering_a_torn_tail_survives_next_recovery() {
        // Regression: recovery must resume the WAL cursor after the last
        // durable commit, not after orphaned torn-tail writes — otherwise
        // a commit made after recovery is mis-grouped and lost on the next
        // recovery.
        let mut storage = MemStorage::new();
        let engine = Box::new(Engine::new());
        // A: 2 writes + commit at WAL_START..+3 (durable).
        engine
            .transaction_durable(&mut storage, 4, |t| {
                t.put(k(1), 1)?;
                t.put(k(2), 2)?;
                Ok(())
            })
            .unwrap();
        // B: 2 writes + commit at WAL_START+3..+6; tear B's commit (at +5).
        engine
            .transaction_durable(&mut storage, 4, |t| {
                t.put(k(3), 3)?;
                t.put(k(4), 4)?;
                Ok(())
            })
            .unwrap();
        storage.force_write(WAL_START + 5, garbage_block());

        // Recovery keeps A, discards torn B, and rewinds the cursor to +3.
        let r1 = Box::new(Engine::new());
        r1.recover(&mut storage).unwrap();
        assert_eq!(r1.transaction(4, |t| t.get(k(1))).unwrap(), Some(1));
        assert_eq!(r1.transaction(4, |t| t.get(k(3))).unwrap(), None);

        // C commits durably; it must overwrite B's orphan records.
        r1.transaction_durable(&mut storage, 4, |t| t.put(k(5), 5)).unwrap();

        // Second recovery: A and C survive, B stays gone.
        let r2 = Box::new(Engine::new());
        r2.recover(&mut storage).unwrap();
        assert_eq!(r2.transaction(4, |t| t.get(k(1))).unwrap(), Some(1));
        assert_eq!(r2.transaction(4, |t| t.get(k(5))).unwrap(), Some(5));
        assert_eq!(r2.transaction(4, |t| t.get(k(3))).unwrap(), None);
    }

    #[test]
    fn durable_delete_recovers_as_absent() {
        let mut storage = MemStorage::new();
        let engine = Box::new(Engine::new());
        engine
            .transaction_durable(&mut storage, 4, |t| t.put(k(1), 5))
            .unwrap();
        engine
            .transaction_durable(&mut storage, 4, |t| t.delete(k(1)))
            .unwrap();

        let recovered = Box::new(Engine::new());
        recovered.recover(&mut storage).unwrap();
        assert_eq!(recovered.transaction(4, |t| t.get(k(1))).unwrap(), None);
    }

    #[test]
    fn commit_after_recovery_outranks_recovered_versions() {
        let mut storage = MemStorage::new();
        let engine = Box::new(Engine::new());
        engine
            .transaction_durable(&mut storage, 4, |t| t.put(k(1), 1))
            .unwrap();

        let recovered = Box::new(Engine::new());
        recovered.recover(&mut storage).unwrap();

        // Overwriting a recovered key requires the epoch to have advanced
        // past the recovered version, or the new TID would not outrank it.
        let r = recovered
            .transaction_durable(&mut storage, 4, |t| {
                assert_eq!(t.get(k(1))?, Some(1));
                t.put(k(1), 2)?;
                Ok(())
            })
            .unwrap();
        assert_eq!(r, Some(()));
        assert_eq!(recovered.transaction(4, |t| t.get(k(1))).unwrap(), Some(2));

        // And it survives a second crash.
        let again = Box::new(Engine::new());
        again.recover(&mut storage).unwrap();
        assert_eq!(again.transaction(4, |t| t.get(k(1))).unwrap(), Some(2));
    }

    #[test]
    fn concurrent_index_lands_every_disjoint_key() {
        use std::sync::Arc;
        use std::thread;

        // Drive the lock-free index through the transaction path from several
        // threads at once. Each thread owns a disjoint key range, so no write
        // conflicts: every put must commit, and every value must read back —
        // proving concurrent lookups, first-touch slot allocation, and tree
        // inserts stay consistent under contention.
        const THREADS: u64 = 4;
        const PER: u64 = 64; // THREADS * PER = 256 keys <= ENGINE_RECORDS

        let engine = Arc::new(Engine::concurrent());
        let mut handles = std::vec::Vec::new();
        for t in 0..THREADS {
            let e = Arc::clone(&engine);
            handles.push(thread::spawn(move || {
                for i in 0..PER {
                    let key = t * PER + i;
                    let committed = e.transaction(8, |txn| {
                        txn.put(k(key), key + 1)?;
                        Ok(())
                    });
                    assert_eq!(committed, Some(()), "disjoint put must commit");
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        // Every key from every thread is present with its value.
        for key in 0..(THREADS * PER) {
            let got = engine.transaction(8, |txn| txn.get(k(key))).unwrap();
            assert_eq!(got, Some(key + 1), "key {key} lost or wrong after concurrent load");
        }
    }

    #[test]
    fn masstree_index_lands_every_disjoint_key() {
        use std::sync::Arc;
        use std::thread;

        // Same disjoint-key concurrent load, but on the fine-grained Masstree
        // index: threads inserting brand-new keys into different subtrees lock
        // only the nodes they touch, so this also exercises concurrent
        // first-touch allocation + tree inserts that do NOT serialize on one
        // writer lock. Every put must still commit and read back exactly.
        const THREADS: u64 = 4;
        const PER: u64 = 64;

        let engine = Arc::new(Engine::masstree());
        let mut handles = std::vec::Vec::new();
        for t in 0..THREADS {
            let e = Arc::clone(&engine);
            handles.push(thread::spawn(move || {
                for i in 0..PER {
                    let key = t * PER + i;
                    // transaction_on(t): this thread's new keys allocate from
                    // cpu t's private index pool via the lock-free slot bump --
                    // the #6 path, exercised concurrently.
                    let committed = e.transaction_on(t as usize, 8, |txn| {
                        txn.put(k(key), key + 1)?;
                        Ok(())
                    });
                    assert_eq!(committed, Some(()), "disjoint put must commit");
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        for key in 0..(THREADS * PER) {
            let got = engine.transaction(8, |txn| txn.get(k(key))).unwrap();
            assert_eq!(got, Some(key + 1), "key {key} lost or wrong on masstree index");
        }
    }

    #[test]
    fn concurrent_readers_never_see_a_half_installed_key() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU64, Ordering as O};
        use std::thread;

        // One writer streams new keys into the concurrent index while readers
        // hammer the whole committed prefix. A committed key must never read
        // back absent or with a wrong value mid-restructuring (leaf/internal
        // splits happen as the tree grows).
        let engine = Arc::new(Engine::concurrent());
        let committed = Arc::new(AtomicU64::new(0));
        let n = 256u64;

        let mut readers = std::vec::Vec::new();
        for _ in 0..3 {
            let e = Arc::clone(&engine);
            let c = Arc::clone(&committed);
            readers.push(thread::spawn(move || {
                while c.load(O::Acquire) < n {
                    let hi = c.load(O::Acquire);
                    for key in 0..hi {
                        let got = e.transaction(8, |txn| txn.get(k(key))).unwrap();
                        assert_eq!(got, Some(key + 1), "committed key {key} vanished");
                    }
                }
            }));
        }

        for key in 0..n {
            let ok = engine.transaction(8, |txn| {
                txn.put(k(key), key + 1)?;
                Ok(())
            });
            assert_eq!(ok, Some(()));
            committed.store(key + 1, O::Release);
        }
        for r in readers {
            r.join().unwrap();
        }
    }
}
