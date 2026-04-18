//! Durable key-value store built from the WAL and the in-memory B+tree.
//!
//! `put` appends a WAL record (durable once the call returns) and inserts
//! into the tree. `get` reads directly from the tree. `recover` rebuilds
//! the tree by replaying every WAL record in order.
//!
//! Ordering on `put`: WAL first, then tree. If a crash happens between the
//! WAL write and the tree insert, `recover` on the next boot replays the
//! WAL record and the in-memory state is restored — the WAL is the source
//! of truth, the tree is a cache.
//!
//! Phase 3 limitations:
//! - No update semantics: a second `put` for an existing key returns
//!   `DuplicateKey` (the B+tree rejects duplicates). Upsert comes with a
//!   future `BpTree::update`.
//! - `Op::Delete` records are skipped during replay: the B+tree has no
//!   remove yet. A Put-then-Delete-then-Put sequence will recover
//!   incorrectly (first Put wins instead of last Put). Safe because Phase 3
//!   doesn't expose a `delete` method.

use crate::bptree::{BpTree, InsertError, Key, POOL_SIZE, Value};
use crate::lba_alloc::WAL_START;
use crate::storage::BlockStorage;
use crate::wal::{Op, Wal};

pub struct Kv {
    wal: Wal,
    tree: BpTree,
}

#[derive(Debug)]
pub enum PutError<E> {
    Io(E),
    DuplicateKey,
    TreeFull,
}

#[derive(Debug)]
pub enum RecoverError<E> {
    Io(E),
    TreeFull,
    /// Two `Op::Put` records for the same key during replay. Indicates the
    /// WAL was written by code that supported upsert, but this version does
    /// not.
    DuplicateReplay,
    /// A WAL record with an unrecognized operation code. Either the WAL was
    /// written by a newer version or the record is corrupt.
    MalformedRecord,
}

impl Kv {
    pub const fn new() -> Self {
        Self {
            wal: Wal::new(),
            tree: BpTree::new(),
        }
    }

    /// Replay the WAL into a fresh B+tree. Returns a `Kv` whose cursor
    /// continues from the end of the recovered log.
    pub fn recover<S: BlockStorage>(storage: &mut S) -> Result<Self, RecoverError<S::Error>> {
        let wal = Wal::recover(storage).map_err(RecoverError::Io)?;
        let mut tree = BpTree::new();

        let mut lba = WAL_START;
        while lba < wal.next_lba() {
            let rec = Wal::read_at(storage, lba)
                .map_err(RecoverError::Io)?
                .expect("recovered WAL contains invalid record at known-valid LBA");
            match rec.op() {
                Some(Op::Put) => {
                    tree.insert(rec.key, rec.value).map_err(|e| match e {
                        InsertError::DuplicateKey => RecoverError::DuplicateReplay,
                        InsertError::NodePoolExhausted => RecoverError::TreeFull,
                    })?;
                }
                Some(Op::Delete) => {
                    // Deferred: B+tree has no remove yet.
                }
                None => {
                    return Err(RecoverError::MalformedRecord);
                }
            }
            lba += 1;
        }

        Ok(Self { wal, tree })
    }

    /// Insert a new key/value pair. The record is durable before this call
    /// returns. Rejects `DuplicateKey` and `TreeFull` *before* touching the
    /// WAL so that a durable record is never produced that a same-size tree
    /// cannot later replay.
    pub fn put<S: BlockStorage>(
        &mut self,
        storage: &mut S,
        key: Key,
        value: Value,
    ) -> Result<(), PutError<S::Error>> {
        if self.tree.lookup(key).is_some() {
            return Err(PutError::DuplicateKey);
        }
        // The tree's own pre-check is identical to this one; mirror it here
        // so the durable WAL append never runs if the tree is already full.
        // Without this mirror, a `TreeFull` after `wal.append` would leave a
        // record the next `recover` cannot absorb, permanently poisoning
        // recovery at this pool size.
        if !self.tree_has_capacity_for_insert() {
            return Err(PutError::TreeFull);
        }
        self.wal
            .append(storage, Op::Put, 0, key, value)
            .map_err(PutError::Io)?;
        // Both preconditions are now established: neither arm can fire.
        self.tree.insert(key, value).map_err(|e| match e {
            InsertError::DuplicateKey => PutError::DuplicateKey,
            InsertError::NodePoolExhausted => PutError::TreeFull,
        })?;
        Ok(())
    }

    fn tree_has_capacity_for_insert(&self) -> bool {
        // One split per level on the descent plus a new root if the top
        // splits; empty tree needs one leaf.
        let (nodes, height) = self.tree_stats();
        let required = if height == 0 { 1 } else { height + 1 };
        (POOL_SIZE as u32).saturating_sub(nodes) >= required
    }

    pub fn get(&self, key: Key) -> Option<Value> {
        self.tree.lookup(key)
    }

    pub fn next_lsn(&self) -> u64 {
        self.wal.next_lsn()
    }

    /// Debug: (node count, tree height) for the B+tree backing the store.
    pub fn tree_stats(&self) -> (u32, u32) {
        (self.tree.num_nodes(), self.tree.height())
    }
}

impl Default for Kv {
    fn default() -> Self {
        Self::new()
    }
}
