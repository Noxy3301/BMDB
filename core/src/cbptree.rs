//! Concurrent B+tree with Masstree-style optimistic version counters.
//!
//! Readers do not latch; they re-read the node's version word before
//! and after the in-node search and retry on any mismatch. Writers
//! take a per-node lock bit via CAS, mark the node as `inserting` or
//! `splitting` while mid-mutation, then clear the state bits and bump
//! `v_insert` / `v_split` with a single release-store — that's the
//! reader-visible commit point.
//!
//! References:
//!   Mao, Kohler, Morris — "Cache Craftiness for Fast Multicore
//!   Key-Value Storage", EuroSys 2012, §3.2 (version layout, reader
//!   retry protocol).
//!
//! Scope for feasibility: single-layer B+tree (BMDB keys are fixed
//! 8 bytes so Masstree's multi-layer trie is unnecessary), fixed-size
//! node pool (`POOL_SIZE`), retire-through-EBR for deletes. Range
//! scan follows the leaf `next_leaf` chain.
//!
//! This module currently ships the layout + lookup path; insert,
//! split, and delete land in follow-up commits.

use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

pub type Key = [u8; 8];
/// Opaque 8-byte payload. For the KV crate this is a value; for the
/// Silo integration it is a `Record` index (or pointer bits the
/// caller transmutes).
pub type Value = u64;

pub const ORDER: usize = 16;
pub const MAX_KEYS: usize = ORDER - 1;
const CHILD_SLOTS: usize = ORDER;

/// Nodes the pool can hand out. Overflow -> `PoolExhausted`.
pub const POOL_SIZE: usize = 256;

pub type NodeId = u32;
pub const NULL_NODE: NodeId = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    Leaf,
    Internal,
}

// ---------------------------------------------------------------------
// Version word.
//
// Masstree EuroSys 2012 §3.2, slightly modified for u32 width on a
// fixed-pool tree:
//   bit  0:  locked     (writer holds the latch)
//   bit  1:  inserting  (a writer is mid-insert; readers retry)
//   bit  2:  splitting  (a writer is mid-split; readers retry and
//                        re-descend if v_split changes)
//   bit  3:  deleted    (node is retired; EBR will reclaim)
//   bits 4..19 (16):   v_insert — bumped on each completed insert
//   bits 20..31 (12):  v_split  — bumped on each completed split
// ---------------------------------------------------------------------

const LOCKED: u32 = 1 << 0;
const INSERTING: u32 = 1 << 1;
const SPLITTING: u32 = 1 << 2;
const DELETED: u32 = 1 << 3;
const V_INSERT_SHIFT: u32 = 4;
const V_INSERT_BITS: u32 = 16;
const V_INSERT_MASK: u32 = ((1u32 << V_INSERT_BITS) - 1) << V_INSERT_SHIFT;
const V_SPLIT_SHIFT: u32 = V_INSERT_SHIFT + V_INSERT_BITS;
const V_SPLIT_BITS: u32 = 12;
const V_SPLIT_MASK: u32 = ((1u32 << V_SPLIT_BITS) - 1) << V_SPLIT_SHIFT;

/// Thin decode wrapper so callers can reason about the word without
/// memorising bit offsets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Version(pub u32);

impl Version {
    pub const fn zero() -> Self {
        Version(0)
    }

    #[inline]
    pub const fn is_locked(self) -> bool {
        (self.0 & LOCKED) != 0
    }

    #[inline]
    pub const fn is_inserting(self) -> bool {
        (self.0 & INSERTING) != 0
    }

    #[inline]
    pub const fn is_splitting(self) -> bool {
        (self.0 & SPLITTING) != 0
    }

    #[inline]
    pub const fn is_deleted(self) -> bool {
        (self.0 & DELETED) != 0
    }

    #[inline]
    pub const fn v_insert(self) -> u32 {
        (self.0 & V_INSERT_MASK) >> V_INSERT_SHIFT
    }

    #[inline]
    pub const fn v_split(self) -> u32 {
        (self.0 & V_SPLIT_MASK) >> V_SPLIT_SHIFT
    }

    /// The portion of the version word readers compare between the
    /// pre- and post-search loads. `locked` is OK if equal on both
    /// sides (a writer that grabbed and released the latch during
    /// the read bumped `v_insert` / `v_split`, which we'll catch).
    /// `inserting` / `splitting` must be clear on the pre-load, so
    /// comparing the whole word is fine.
    #[inline]
    pub const fn stable_snapshot(self) -> Self {
        Version(self.0)
    }
}

/// Node in the concurrent B+tree. Cache-line aligned so contended
/// version CAS doesn't false-share with adjacent nodes' payloads.
#[repr(C, align(64))]
pub struct Node {
    version: AtomicU32,
    kind: NodeKind,
    n_keys: u8,
    _pad: u16,
    keys: [Key; MAX_KEYS],
    // Leaves populate `values` with the payload paired with each key.
    values: [AtomicU64; MAX_KEYS],
    // Internal nodes populate `children`; `children[i]` is the
    // subtree rooted to the left of `keys[i]` (for `i == n_keys`,
    // rightmost subtree).
    children: [AtomicU32; CHILD_SLOTS],
    // Leaf-only: next leaf in key order. `NULL_NODE` at the right edge.
    next_leaf: AtomicU32,
}

impl Node {
    const EMPTY: Self = Self {
        version: AtomicU32::new(0),
        kind: NodeKind::Leaf,
        n_keys: 0,
        _pad: 0,
        keys: [[0u8; 8]; MAX_KEYS],
        values: {
            const Z: AtomicU64 = AtomicU64::new(0);
            [Z; MAX_KEYS]
        },
        children: {
            const Z: AtomicU32 = AtomicU32::new(NULL_NODE);
            [Z; CHILD_SLOTS]
        },
        next_leaf: AtomicU32::new(NULL_NODE),
    };

    #[inline]
    fn load_version(&self, order: Ordering) -> Version {
        Version(self.version.load(order))
    }

    /// Scan for `key` in a leaf. Caller is responsible for the
    /// surrounding version-check retry loop; this function does the
    /// non-atomic work of comparing keys against an assumed-stable
    /// snapshot. Returns the value if present.
    ///
    /// # Safety
    /// The caller must only trust the returned value if the version
    /// word read *before* and *after* this call match and neither
    /// has the inserting / splitting bit set.
    unsafe fn leaf_scan(&self, key: &Key) -> Option<Value> {
        // `n_keys` is a plain byte; the version word's stability is
        // what protects the read. Binary search over 15 slots is
        // cheap enough that linear probe with branch-predictable
        // compares ties or wins; keep linear here.
        let n = self.n_keys as usize;
        for i in 0..n {
            if self.keys[i] == *key {
                let v = self.values[i].load(Ordering::Acquire);
                return Some(v);
            }
            if self.keys[i] > *key {
                return None;
            }
        }
        None
    }

    /// Choose the child subtree for `key` in an internal node. Same
    /// version-stability contract as `leaf_scan`.
    #[inline]
    unsafe fn internal_descend(&self, key: &Key) -> NodeId {
        let n = self.n_keys as usize;
        for i in 0..n {
            if self.keys[i] > *key {
                return self.children[i].load(Ordering::Acquire);
            }
        }
        self.children[n].load(Ordering::Acquire)
    }
}

/// Concurrent B+tree. Readers take `&Tree`; writers (to land in a
/// later commit) also take `&Tree` and rely on per-node locks.
pub struct Tree {
    pool: [Node; POOL_SIZE],
    root: AtomicU32,
}

impl Tree {
    pub const fn new() -> Self {
        const EMPTY_NODE: Node = Node::EMPTY;
        Self {
            pool: [EMPTY_NODE; POOL_SIZE],
            root: AtomicU32::new(NULL_NODE),
        }
    }

    #[inline]
    fn node(&self, id: NodeId) -> &Node {
        &self.pool[id as usize]
    }

    /// Lock-free lookup. Follows the root → leaf descent, checking
    /// the version word before and after each in-node read. A
    /// mismatch or in-flight writer state rewinds the descent from
    /// the root — Masstree's protocol.
    pub fn lookup(&self, key: &Key) -> Option<Value> {
        loop {
            let root = self.root.load(Ordering::Acquire);
            if root == NULL_NODE {
                return None;
            }
            if let Some(result) = self.descend(root, key) {
                return result;
            }
            // None-of-Some here means the descent saw an in-flight
            // split and needs to restart from the root. We model
            // "retry" via an inner Option: `Some(None)` is a clean
            // "key not found", `None` is "retry".
            core::hint::spin_loop();
        }
    }

    /// Recursive-ish descent, returns `None` when the caller must
    /// restart from the root and `Some(x)` when the descent produced
    /// a linearizable answer (`x` is the lookup result).
    fn descend(&self, start: NodeId, key: &Key) -> Option<Option<Value>> {
        let mut node_id = start;
        loop {
            let node = self.node(node_id);
            let v1 = node.load_version(Ordering::Acquire);
            if v1.is_locked() || v1.is_inserting() || v1.is_splitting() {
                return None; // restart
            }
            let result: Option<NextStep> = unsafe { self.step(node, key) };
            // Re-read and compare. Any concurrent writer that
            // completed between our reads will have bumped
            // v_insert or v_split, so the raw u32 compare catches it.
            let v2 = node.load_version(Ordering::Acquire);
            if v1 != v2 {
                return None;
            }
            if v2.is_inserting() || v2.is_splitting() {
                return None;
            }
            match result {
                Some(NextStep::Descend(next)) => {
                    node_id = next;
                    continue;
                }
                Some(NextStep::Found(value)) => return Some(Some(value)),
                Some(NextStep::Absent) => return Some(None),
                None => return None,
            }
        }
    }

    /// One step of the descent: either "this leaf resolved the key"
    /// or "follow this child". Factored out so the version-check
    /// loop in `descend` is linear.
    ///
    /// # Safety
    /// Caller must bracket this call with matching version loads.
    unsafe fn step(&self, node: &Node, key: &Key) -> Option<NextStep> {
        match node.kind {
            NodeKind::Leaf => match unsafe { node.leaf_scan(key) } {
                Some(v) => Some(NextStep::Found(v)),
                None => Some(NextStep::Absent),
            },
            NodeKind::Internal => {
                let next = unsafe { node.internal_descend(key) };
                if next == NULL_NODE {
                    // Malformed — internal nodes always have at
                    // least n_keys + 1 children. Treat as "restart".
                    None
                } else {
                    Some(NextStep::Descend(next))
                }
            }
        }
    }
}

impl Default for Tree {
    fn default() -> Self {
        Self::new()
    }
}

enum NextStep {
    Descend(NodeId),
    Found(Value),
    Absent,
}

// ---------------------------------------------------------------------
// Test-only helpers: directly write node contents to set up a tree
// without going through the (yet-unimplemented) insert path. Once
// insert lands these go away.
// ---------------------------------------------------------------------

#[cfg(test)]
impl Tree {
    /// Initialise a single-leaf tree with the given sorted entries.
    ///
    /// # Safety
    /// Only valid before any concurrent access; `entries` must be
    /// sorted by key and contain at most MAX_KEYS items.
    pub unsafe fn seed_leaf(&self, entries: &[(Key, Value)]) {
        assert!(entries.len() <= MAX_KEYS);
        // Verify sorted.
        for w in entries.windows(2) {
            assert!(w[0].0 < w[1].0);
        }
        let node = &self.pool[0];
        let node_mut = &self.pool[0] as *const Node as *mut Node;
        unsafe {
            (*node_mut).kind = NodeKind::Leaf;
            (*node_mut).n_keys = entries.len() as u8;
            for (i, (k, _)) in entries.iter().enumerate() {
                (*node_mut).keys[i] = *k;
            }
        }
        for (i, (_, v)) in entries.iter().enumerate() {
            node.values[i].store(*v, Ordering::Release);
        }
        node.version.store(0, Ordering::Release);
        self.root.store(0, Ordering::Release);
    }

    /// Raise the lock bit on the root node (test-only). Used to
    /// verify that readers retry while a writer is mid-mutation.
    pub fn test_lock_root(&self) {
        let root = self.root.load(Ordering::Acquire);
        self.pool[root as usize]
            .version
            .fetch_or(LOCKED, Ordering::Release);
    }

    pub fn test_unlock_root_and_bump(&self) {
        let root = self.root.load(Ordering::Acquire);
        // Clear LOCKED, bump v_insert.
        let node = &self.pool[root as usize];
        let mut v = node.version.load(Ordering::Relaxed);
        loop {
            let new_v = (v & !LOCKED).wrapping_add(1u32 << V_INSERT_SHIFT);
            match node.version.compare_exchange_weak(
                v,
                new_v,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(observed) => v = observed,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k(x: u64) -> Key {
        x.to_be_bytes()
    }

    #[test]
    fn empty_tree_returns_none_for_any_key() {
        let t = Tree::new();
        assert_eq!(t.lookup(&k(0)), None);
        assert_eq!(t.lookup(&k(42)), None);
    }

    #[test]
    fn single_leaf_lookup_hit_and_miss() {
        let t = Tree::new();
        unsafe {
            t.seed_leaf(&[(k(1), 100), (k(5), 500), (k(9), 900)]);
        }
        assert_eq!(t.lookup(&k(1)), Some(100));
        assert_eq!(t.lookup(&k(5)), Some(500));
        assert_eq!(t.lookup(&k(9)), Some(900));

        assert_eq!(t.lookup(&k(0)), None, "below min");
        assert_eq!(t.lookup(&k(3)), None, "between");
        assert_eq!(t.lookup(&k(99)), None, "above max");
    }

    #[test]
    fn version_word_bit_layout_is_consistent() {
        // Spot-check the masks so a future edit that renumbers the
        // bits doesn't silently break readers.
        let v = Version(LOCKED | INSERTING | SPLITTING | DELETED);
        assert!(v.is_locked());
        assert!(v.is_inserting());
        assert!(v.is_splitting());
        assert!(v.is_deleted());
        assert_eq!(v.v_insert(), 0);
        assert_eq!(v.v_split(), 0);

        let v = Version((7 << V_INSERT_SHIFT) | (11 << V_SPLIT_SHIFT));
        assert_eq!(v.v_insert(), 7);
        assert_eq!(v.v_split(), 11);
        assert!(!v.is_locked());
    }

    #[test]
    fn concurrent_reader_sees_retry_then_completed_write() {
        // Thread layout: main thread seeds the tree, raises LOCKED on
        // the root (simulates a writer mid-mutation), a reader thread
        // spins on lookup, then the main thread clears LOCKED and
        // bumps v_insert. The reader must eventually return the
        // current value — it must not hang or return a stale None.
        use std::sync::Arc;
        use std::thread;
        use std::time::Duration;

        let t = Arc::new(Tree::new());
        unsafe {
            t.seed_leaf(&[(k(7), 777)]);
        }
        t.test_lock_root();

        let reader_t = Arc::clone(&t);
        let reader = thread::spawn(move || reader_t.lookup(&k(7)));

        // Let the reader spin on retry for a moment.
        thread::sleep(Duration::from_millis(10));
        t.test_unlock_root_and_bump();

        let got = reader.join().unwrap();
        assert_eq!(got, Some(777));
    }
}
