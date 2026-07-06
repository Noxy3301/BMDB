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
//! Scope of this first cut: point `get`/`put` with read-your-writes,
//! absent-read validation, and a bounded retry loop. Not yet here:
//! durable commit (writes are in-memory only), delete/tombstones, range
//! scans with phantom protection, and the concurrent index.
//!
//! [absent]: Tid::is_absent

use crate::bptree::{BpTree, Key};
use crate::silo::{self, CommitOutcome, Record, Tid, TxnState, current_epoch};
use crate::sync::SpinLock;

/// Tuples the engine can hold. Bounds the static footprint (each
/// [`Record`] is a 64-byte cache line, so this array is
/// `ENGINE_RECORDS * 64` bytes). The index may run out of nodes before
/// this cap is hit; both limits surface as [`EngineError::OutOfSpace`].
pub const ENGINE_RECORDS: usize = 512;

/// Ordered key → record-slot map. A thin trait so the engine can move
/// from the sequential [`BpTree`] to the concurrent index later without
/// touching the transaction path.
pub trait Index {
    /// Slot currently mapped to `key`, or `None` if unmapped.
    fn get(&self, key: Key) -> Option<u32>;
    /// Map `key` to `slot`. Returns [`IndexFull`] if the index cannot
    /// grow to hold another key.
    fn insert(&mut self, key: Key, slot: u32) -> Result<(), IndexFull>;
}

/// The index cannot accept another key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexFull;

/// [`Index`] backed by the sequential B+tree. The tree's 8-byte value
/// slot carries the record index as a big-endian `u64`.
pub struct BpTreeIndex {
    tree: BpTree,
}

impl BpTreeIndex {
    pub const fn new() -> Self {
        Self { tree: BpTree::new() }
    }
}

impl Default for BpTreeIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl Index for BpTreeIndex {
    fn get(&self, key: Key) -> Option<u32> {
        self.tree.lookup(key).map(|v| u64::from_be_bytes(v) as u32)
    }

    fn insert(&mut self, key: Key, slot: u32) -> Result<(), IndexFull> {
        self.tree
            .upsert(key, (slot as u64).to_be_bytes())
            .map(|_| ())
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

/// Index + slot allocator, guarded together so find-or-allocate is
/// atomic across concurrent transactions touching new keys.
struct Directory<I: Index> {
    index: I,
    next_slot: u32,
}

/// A key-addressable transactional store over Silo OCC.
pub struct Engine<I: Index = BpTreeIndex> {
    records: [Record; ENGINE_RECORDS],
    dir: SpinLock<Directory<I>>,
}

impl Engine<BpTreeIndex> {
    /// Construct an empty engine. `const` so it can back a `static` on
    /// bare metal without a runtime initializer.
    pub const fn new() -> Self {
        Self {
            // A fresh slot is `absent`: no key maps to it yet, and a
            // reader that reaches one (e.g. after an aborted insert
            // left an index entry) sees `None` rather than a zero value.
            records: [const { Record::new(Tid::from_raw(Tid::ABSENT), 0) }; ENGINE_RECORDS],
            dir: SpinLock::new(Directory {
                index: BpTreeIndex::new(),
                next_slot: 0,
            }),
        }
    }
}

impl Default for Engine<BpTreeIndex> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: Index> Engine<I> {
    /// Begin a transaction stamped with the current epoch.
    pub fn begin(&self) -> Txn<'_, I> {
        Txn {
            engine: self,
            state: TxnState::new(current_epoch()),
        }
    }

    /// Run `body` as a transaction, retrying on OCC abort up to
    /// `max_attempts` times. Returns the body's value on commit, or
    /// `None` if every attempt aborted or the body returned an error
    /// (a full set / out-of-space is terminal, not retried).
    pub fn transaction<F, R>(&self, max_attempts: u32, mut body: F) -> Option<R>
    where
        F: FnMut(&mut Txn<'_, I>) -> Result<R, EngineError>,
    {
        for _ in 0..max_attempts {
            let mut txn = self.begin();
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

    /// Slot for `key`, allocating a fresh record if the key is new.
    fn slot_for(&self, key: Key) -> Option<u32> {
        let mut dir = self.dir.lock();
        if let Some(slot) = dir.index.get(key) {
            return Some(slot);
        }
        let slot = dir.next_slot;
        if slot as usize >= ENGINE_RECORDS {
            return None;
        }
        dir.index.insert(key, slot).ok()?;
        dir.next_slot = slot + 1;
        Some(slot)
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
                return Ok(Some(w.new_value));
            }
        }
        // Allocate a stable slot even for an absent key so this read can
        // be validated: without a record, a later insert of `key` would
        // be an undetectable phantom.
        let slot = self.engine.slot_for(key).ok_or(EngineError::OutOfSpace)?;
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
        let slot = self.engine.slot_for(key).ok_or(EngineError::OutOfSpace)?;
        let record = self.engine.record(slot);
        self.state
            .add_write(record, key, value)
            .map_err(|_| EngineError::TxnTooLarge)
    }

    /// Run the Silo precommit protocol over the buffered read/write set
    /// and consume the transaction. A read-only transaction commits iff
    /// its read set still validates.
    pub fn commit(mut self) -> CommitOutcome {
        // Safety: every record address in the sets points into
        // `engine.records`, which outlives `self` (the `&'e` borrow) and
        // never moves or frees a slot.
        unsafe { silo::commit(&mut self.state) }
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
}
