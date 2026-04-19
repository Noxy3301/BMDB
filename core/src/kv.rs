//! Durable key-value store built from the WAL and the in-memory B+tree.
//!
//! `put` appends a WAL record (durable once the call returns) and upserts
//! into the tree, returning the previous value if the key already existed.
//! `delete` appends a Delete record and removes the key from the tree.
//! `get` reads directly from the tree. `recover` rebuilds the tree by
//! replaying every WAL record in order.
//!
//! Ordering on `put` / `delete`: WAL first, then tree. If a crash happens
//! between the WAL write and the tree mutation, `recover` on the next boot
//! replays the WAL record and the in-memory state is restored — the WAL is
//! the source of truth, the tree is a cache.
//!
//! Capacity invariant: `put` refuses a new-key insert if the B+tree pool
//! could not absorb the resulting splits, *before* the WAL append. Without
//! that pre-check, a durable record could remain that a same-size tree
//! cannot later replay, permanently poisoning recovery at this pool size.
//! Overwriting an existing key never allocates, so that path is exempt
//! from the pre-check.

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
    TreeFull,
}

#[derive(Debug)]
pub enum RecoverError<E> {
    Io(E),
    TreeFull,
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
                    // Upsert (not insert): replaying a sequence like
                    // Put(k, v1) → Put(k, v2) leaves the latest value in
                    // the tree, matching the semantics of runtime `put`.
                    tree.upsert(rec.key, rec.value).map_err(|e| match e {
                        InsertError::NodePoolExhausted => RecoverError::TreeFull,
                        InsertError::DuplicateKey => {
                            unreachable!("upsert cannot return DuplicateKey")
                        }
                    })?;
                }
                Some(Op::Delete) => {
                    // Deleting an absent key is a no-op; the return value is
                    // discarded. This handles both an honest delete and a
                    // Delete-for-absent-key produced by unconditional
                    // `Kv::delete` below.
                    let _ = tree.delete(rec.key);
                }
                None => {
                    return Err(RecoverError::MalformedRecord);
                }
            }
            lba += 1;
        }

        Ok(Self { wal, tree })
    }

    /// Insert a new key/value pair, or overwrite the value if the key is
    /// already present. The record is durable before this call returns.
    /// Returns the previous value when overwriting, `None` for a fresh
    /// insert.
    ///
    /// Rejects `TreeFull` *before* touching the WAL on the new-key path so
    /// that a durable record is never produced that a same-size tree cannot
    /// later replay. An overwrite never allocates and is therefore allowed
    /// even when the tree pool is otherwise full.
    pub fn put<S: BlockStorage>(
        &mut self,
        storage: &mut S,
        key: Key,
        value: Value,
    ) -> Result<Option<Value>, PutError<S::Error>> {
        let will_allocate = self.tree.lookup(key).is_none();
        if will_allocate && !self.tree_has_capacity_for_insert() {
            return Err(PutError::TreeFull);
        }
        self.wal
            .append(storage, Op::Put, 0, key, value)
            .map_err(PutError::Io)?;
        // Pre-check guarantees this cannot fail: an existing key takes the
        // alloc-free fast path, and a new key has reserved capacity.
        let old = self
            .tree
            .upsert(key, value)
            .expect("tree.upsert after capacity pre-check must succeed");
        Ok(old)
    }

    /// Remove `key` from the store. The Delete record is durable before
    /// this call returns, regardless of whether the key was actually
    /// present — so recovery sees exactly the same Delete sequence the
    /// caller issued. Returns the removed value, or `None` if the key was
    /// absent.
    pub fn delete<S: BlockStorage>(
        &mut self,
        storage: &mut S,
        key: Key,
    ) -> Result<Option<Value>, S::Error> {
        self.wal.append(storage, Op::Delete, 0, key, [0; 8])?;
        Ok(self.tree.delete(key))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mem_storage::MemStorage;

    fn k(i: u64) -> Key {
        i.to_be_bytes()
    }
    fn v(i: u64) -> Value {
        (i * 100).to_be_bytes()
    }

    #[test]
    fn put_then_get_returns_value() {
        let mut storage = MemStorage::new();
        let mut kv = Kv::new();
        let old = kv.put(&mut storage, k(1), v(1)).unwrap();
        assert_eq!(old, None);
        assert_eq!(kv.get(k(1)), Some(v(1)));
        assert_eq!(kv.get(k(2)), None);
    }

    #[test]
    fn put_overwrites_existing_key_and_returns_old_value() {
        let mut storage = MemStorage::new();
        let mut kv = Kv::new();
        kv.put(&mut storage, k(1), v(1)).unwrap();
        let flushes_before = storage.flush_count();

        let old = kv.put(&mut storage, k(1), v(99)).unwrap();
        assert_eq!(old, Some(v(1)));
        assert_eq!(kv.get(k(1)), Some(v(99)));

        // Overwrite must have written a fresh WAL record: both LSN and
        // flush count advance.
        assert!(storage.flush_count() > flushes_before);
        assert_eq!(kv.next_lsn(), 3);
    }

    #[test]
    fn delete_removes_key_and_returns_old_value() {
        let mut storage = MemStorage::new();
        let mut kv = Kv::new();
        kv.put(&mut storage, k(5), v(5)).unwrap();

        let removed = kv.delete(&mut storage, k(5)).unwrap();
        assert_eq!(removed, Some(v(5)));
        assert_eq!(kv.get(k(5)), None);
    }

    #[test]
    fn delete_of_absent_key_returns_none_but_still_appends_wal() {
        let mut storage = MemStorage::new();
        let mut kv = Kv::new();
        let flushes_before = storage.flush_count();

        let removed = kv.delete(&mut storage, k(42)).unwrap();
        assert_eq!(removed, None);

        // The WAL record is written unconditionally so recovery sees the
        // same Delete sequence the caller issued.
        assert!(storage.flush_count() > flushes_before);
        assert_eq!(kv.next_lsn(), 2);
    }

    #[test]
    fn recover_on_empty_storage_is_empty_kv() {
        let mut storage = MemStorage::new();
        let kv = Kv::recover(&mut storage).unwrap();
        assert_eq!(kv.next_lsn(), 1);
        assert_eq!(kv.get(k(1)), None);
    }

    #[test]
    fn recover_replays_every_put_in_order() {
        let mut storage = MemStorage::new();
        {
            let mut kv = Kv::new();
            for i in 1..=10u64 {
                kv.put(&mut storage, k(i), v(i)).unwrap();
            }
        }

        // Simulate a crash: throw away the in-memory Kv, rebuild from WAL.
        let kv = Kv::recover(&mut storage).unwrap();
        assert_eq!(kv.next_lsn(), 11);
        for i in 1..=10u64 {
            assert_eq!(kv.get(k(i)), Some(v(i)), "missing key {} after recover", i);
        }
    }

    #[test]
    fn recover_applies_delete_records() {
        // Put a key, delete it, recover. The delete must take effect during
        // replay, otherwise the key would remain.
        let mut storage = MemStorage::new();
        {
            let mut kv = Kv::new();
            kv.put(&mut storage, k(7), v(7)).unwrap();
            kv.delete(&mut storage, k(7)).unwrap();
        }
        let kv = Kv::recover(&mut storage).unwrap();
        assert_eq!(kv.get(k(7)), None, "Delete must be applied during replay");
        assert_eq!(kv.next_lsn(), 3);
    }

    #[test]
    fn recover_preserves_last_value_on_put_delete_put_sequence() {
        let mut storage = MemStorage::new();
        {
            let mut kv = Kv::new();
            kv.put(&mut storage, k(3), v(1)).unwrap(); // v1
            kv.delete(&mut storage, k(3)).unwrap();
            kv.put(&mut storage, k(3), v(2)).unwrap(); // v2 is the survivor
        }
        let kv = Kv::recover(&mut storage).unwrap();
        assert_eq!(kv.get(k(3)), Some(v(2)));
    }

    #[test]
    fn recover_handles_overwrite_in_log() {
        // Two Puts on the same key: replay must land the second value,
        // mirroring the runtime upsert semantics of `put`.
        let mut storage = MemStorage::new();
        {
            let mut kv = Kv::new();
            kv.put(&mut storage, k(9), v(1)).unwrap();
            kv.put(&mut storage, k(9), v(2)).unwrap();
        }
        let kv = Kv::recover(&mut storage).unwrap();
        assert_eq!(kv.get(k(9)), Some(v(2)));
    }

    #[test]
    fn put_durability_across_simulated_crash() {
        // Gate: put returns Ok → crash → recover → get returns the value.
        let mut storage = MemStorage::new();
        {
            let mut kv = Kv::new();
            kv.put(&mut storage, k(42), v(42)).unwrap();
            // Drop kv here; represents a process crash.
        }
        let kv = Kv::recover(&mut storage).unwrap();
        assert_eq!(kv.get(k(42)), Some(v(42)));
    }

    #[test]
    fn many_put_then_recover_preserves_all() {
        // Stress-test: insert enough keys to trigger B+tree splits, then
        // recover and confirm every key is still accessible.
        let mut storage = MemStorage::new();
        {
            let mut kv = Kv::new();
            for i in 1..=40u64 {
                kv.put(&mut storage, k(i), v(i)).unwrap();
            }
        }
        let kv = Kv::recover(&mut storage).unwrap();
        for i in 1..=40u64 {
            assert_eq!(kv.get(k(i)), Some(v(i)), "lost key {} after recover", i);
        }
        let (_nodes, height) = kv.tree_stats();
        assert!(height >= 2, "40 inserts should grow the tree beyond a single leaf");
    }

    #[test]
    fn put_and_delete_survive_multiple_crash_cycles() {
        // Three boots, two simulated crashes: put → crash → recover+delete
        // → crash → recover. Verifies that Delete records written in one
        // boot are correctly applied when replayed in the next.
        let mut storage = MemStorage::new();
        {
            let mut kv = Kv::new();
            kv.put(&mut storage, k(1), v(1)).unwrap();
            kv.put(&mut storage, k(2), v(2)).unwrap();
        }
        {
            let mut kv = Kv::recover(&mut storage).unwrap();
            assert_eq!(kv.get(k(1)), Some(v(1)));
            assert_eq!(kv.get(k(2)), Some(v(2)));
            kv.delete(&mut storage, k(1)).unwrap();
            kv.put(&mut storage, k(3), v(3)).unwrap();
        }
        let kv = Kv::recover(&mut storage).unwrap();
        assert_eq!(kv.get(k(1)), None);
        assert_eq!(kv.get(k(2)), Some(v(2)));
        assert_eq!(kv.get(k(3)), Some(v(3)));
    }

    #[test]
    fn put_overwrite_succeeds_when_tree_pool_full() {
        // Fill the tree pool until a new-key put fails, then overwrite an
        // existing key. The in-place upsert must succeed even though no
        // new-key insert would fit.
        let mut storage = MemStorage::new();
        let mut kv = Kv::new();

        let mut last_inserted = 0u64;
        for i in 1..=10_000u64 {
            match kv.put(&mut storage, k(i), v(i)) {
                Ok(_) => last_inserted = i,
                Err(PutError::TreeFull) => break,
                Err(e) => panic!("unexpected error during fill: {:?}", e),
            }
        }
        // A fresh key must still be rejected.
        let rejected = kv.put(&mut storage, k(last_inserted + 1), v(0));
        assert!(matches!(rejected, Err(PutError::TreeFull)));

        // But an existing-key overwrite must succeed.
        let old = kv
            .put(&mut storage, k(1), v(42))
            .expect("overwrite must succeed on full pool");
        assert_eq!(old, Some(v(1)));
        assert_eq!(kv.get(k(1)), Some(v(42)));
    }

    #[test]
    fn delete_of_absent_key_during_recover_is_noop() {
        // Synthetic WAL: Delete for a key that was never Put. Replay must
        // absorb it without error.
        let mut storage = MemStorage::new();
        let mut wal = Wal::new();
        wal.append(&mut storage, Op::Delete, 0, k(99), [0; 8]).unwrap();

        let kv = Kv::recover(&mut storage).unwrap();
        assert_eq!(kv.get(k(99)), None);
        assert_eq!(kv.next_lsn(), 2);
    }
}
