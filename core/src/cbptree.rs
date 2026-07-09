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

use core::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};

/// Interpret a key as a `u64` so it can live in an atomic cell. Keys are
/// compared lexicographically; big-endian bytes make the `u64` order match.
#[inline]
const fn key_word(key: &Key) -> u64 {
    u64::from_be_bytes(*key)
}

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

/// Maximum tree height the insert path buffer supports. Far above any
/// height a `POOL_SIZE` pool can reach (branching factor `ORDER`).
const MAX_DEPTH: usize = 16;

pub type NodeId = u32;
pub const NULL_NODE: NodeId = u32::MAX;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    Leaf = 0,
    Internal = 1,
}

impl NodeKind {
    #[inline]
    const fn from_u8(v: u8) -> Self {
        if v == NodeKind::Internal as u8 {
            NodeKind::Internal
        } else {
            NodeKind::Leaf
        }
    }
}

// ---------------------------------------------------------------------
// Version word (64-bit).
//
// Masstree EuroSys 2012 §3.2. A 64-bit word gives the sequence counters
// enough width that they cannot wrap within any realistic reader window
// — a 32-bit `v_insert` would need 2^32 completed inserts on this exact
// node between a reader's two version loads to alias (ABA), which cannot
// happen (the node splits long before). Layout:
//   bit  0:      locked     (writer holds the latch)
//   bit  1:      inserting  (a writer is mid-insert; readers retry)
//   bit  2:      splitting  (a writer is mid-split; readers retry and
//                            re-descend if v_split changes)
//   bit  3:      deleted    (node is retired; EBR will reclaim)
//   bits 4..35 (32):   v_insert — bumped on each completed insert
//   bits 36..63 (28):  v_split  — bumped on each completed split
// ---------------------------------------------------------------------

const LOCKED: u64 = 1 << 0;
const INSERTING: u64 = 1 << 1;
const SPLITTING: u64 = 1 << 2;
const DELETED: u64 = 1 << 3;
const V_INSERT_SHIFT: u32 = 4;
const V_INSERT_BITS: u32 = 32;
const V_INSERT_MASK: u64 = ((1u64 << V_INSERT_BITS) - 1) << V_INSERT_SHIFT;
const V_SPLIT_SHIFT: u32 = V_INSERT_SHIFT + V_INSERT_BITS;
const V_SPLIT_BITS: u32 = 28;
const V_SPLIT_MASK: u64 = ((1u64 << V_SPLIT_BITS) - 1) << V_SPLIT_SHIFT;

/// Thin decode wrapper so callers can reason about the word without
/// memorising bit offsets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Version(pub u64);

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
    pub const fn v_insert(self) -> u64 {
        (self.0 & V_INSERT_MASK) >> V_INSERT_SHIFT
    }

    #[inline]
    pub const fn v_split(self) -> u64 {
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
    version: AtomicU64,
    // Set once in `alloc` before the node is published and never changed
    // after; atomic so the write is not aliasing UB through `&Node`.
    kind: AtomicU8,
    n_keys: AtomicU8,
    // Keys as big-endian `u64` words so readers can load them atomically.
    keys: [AtomicU64; MAX_KEYS],
    // Leaves populate `values` with the payload paired with each key.
    values: [AtomicU64; MAX_KEYS],
    // Internal nodes populate `children`; `children[i]` is the
    // subtree rooted to the left of `keys[i]` (for `i == n_keys`,
    // rightmost subtree).
    children: [AtomicU32; CHILD_SLOTS],
    // Leaf-only: next leaf in key order. `NULL_NODE` at the right edge.
    next_leaf: AtomicU32,
    // B-link upper fence (as a big-endian key word): every key in this leaf
    // is `< high_key`. A search key `>= high_key` belongs to a right sibling
    // — which happens when a split moved it there after a reader had already
    // routed to this (now-truncated) leaf; the reader follows `next_leaf`.
    // `u64::MAX` on a leaf that owns the right edge of its range.
    high_key: AtomicU64,
}

impl Node {
    const EMPTY: Self = Self {
        version: AtomicU64::new(0),
        kind: AtomicU8::new(NodeKind::Leaf as u8),
        n_keys: AtomicU8::new(0),
        keys: {
            const Z: AtomicU64 = AtomicU64::new(0);
            [Z; MAX_KEYS]
        },
        values: {
            const Z: AtomicU64 = AtomicU64::new(0);
            [Z; MAX_KEYS]
        },
        children: {
            const Z: AtomicU32 = AtomicU32::new(NULL_NODE);
            [Z; CHILD_SLOTS]
        },
        next_leaf: AtomicU32::new(NULL_NODE),
        high_key: AtomicU64::new(u64::MAX),
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
        // Atomic loads keep this race-free; the version word's stability
        // is what makes the *set* of loads consistent. Linear probe over
        // the (small, branch-predictable) key array.
        let kw = key_word(key);
        let n = (self.n_keys.load(Ordering::Acquire) as usize).min(MAX_KEYS);
        for i in 0..n {
            let stored = self.keys[i].load(Ordering::Acquire);
            if stored == kw {
                return Some(self.values[i].load(Ordering::Acquire));
            }
            if stored > kw {
                return None;
            }
        }
        None
    }

    /// Choose the child subtree for `key` in an internal node. Same
    /// version-stability contract as `leaf_scan`.
    #[inline]
    unsafe fn internal_descend(&self, key: &Key) -> NodeId {
        let kw = key_word(key);
        let n = (self.n_keys.load(Ordering::Acquire) as usize).min(MAX_KEYS);
        for i in 0..n {
            if self.keys[i].load(Ordering::Acquire) > kw {
                return self.children[i].load(Ordering::Acquire);
            }
        }
        self.children[n].load(Ordering::Acquire)
    }

    /// Spin-acquire the writer lock (the version word's LOCKED bit). Only
    /// one writer holds it at a time; readers observe LOCKED and retry.
    fn lock(&self) {
        loop {
            let v = self.version.load(Ordering::Relaxed);
            if v & LOCKED == 0
                && self
                    .version
                    .compare_exchange_weak(v, v | LOCKED, Ordering::Acquire, Ordering::Relaxed)
                    .is_ok()
            {
                return;
            }
            core::hint::spin_loop();
        }
    }

    /// Release the writer lock and publish the mutation: clear LOCKED and
    /// INSERTING and bump `v_insert` (wrapping inside its 32-bit field,
    /// never into `v_split`) in a single release store. This is the
    /// reader-visible commit point; the release publishes the relaxed key/
    /// value/n_keys stores made under the lock. The caller holds the lock,
    /// so the version word is ours.
    fn unlock_bump_insert(&self) {
        let v = self.version.load(Ordering::Relaxed);
        let bumped = (v & V_INSERT_MASK).wrapping_add(1 << V_INSERT_SHIFT) & V_INSERT_MASK;
        let new = (v & !(LOCKED | INSERTING | V_INSERT_MASK)) | bumped;
        self.version.store(new, Ordering::Release);
    }

    /// Insert or replace `key`/`value` in this leaf, keeping keys sorted.
    /// Returns [`LeafPut::Full`] if the key is new and the leaf has no room.
    ///
    /// # Safety
    /// The node's writer lock must be held and the node must be a leaf.
    /// `unlock_bump_insert` must publish the result afterwards.
    unsafe fn leaf_insert(&self, key: &Key, value: Value) -> LeafPut {
        let kw = key_word(key);
        let n = self.n_keys.load(Ordering::Relaxed) as usize;
        let mut pos = n;
        for i in 0..n {
            let stored = self.keys[i].load(Ordering::Relaxed);
            if stored == kw {
                self.values[i].store(value, Ordering::Release);
                return LeafPut::Done(Inserted::Updated);
            }
            if stored > kw {
                pos = i;
                break;
            }
        }
        if n >= MAX_KEYS {
            return LeafPut::Full;
        }
        // Tell racing readers the key array is being shifted; the unlock
        // clears this and bumps the version so they re-read. Relaxed stores
        // are fine — the unlock's release publishes them together.
        self.version.fetch_or(INSERTING, Ordering::Release);
        let mut i = n;
        while i > pos {
            self.keys[i].store(self.keys[i - 1].load(Ordering::Relaxed), Ordering::Relaxed);
            self.values[i].store(self.values[i - 1].load(Ordering::Relaxed), Ordering::Relaxed);
            i -= 1;
        }
        self.keys[pos].store(kw, Ordering::Relaxed);
        self.values[pos].store(value, Ordering::Relaxed);
        self.n_keys.store((n + 1) as u8, Ordering::Relaxed);
        LeafPut::Done(Inserted::New)
    }

    /// Release the lock publishing a structural change: clear LOCKED,
    /// SPLITTING and INSERTING (a split may also insert the new key into
    /// this half, setting INSERTING) and bump `v_split`. Readers that
    /// overlapped re-descend from the root, since a split can move a key to
    /// a different node.
    fn unlock_bump_split(&self) {
        let v = self.version.load(Ordering::Relaxed);
        let bumped = (v & V_SPLIT_MASK).wrapping_add(1 << V_SPLIT_SHIFT) & V_SPLIT_MASK;
        let new = (v & !(LOCKED | SPLITTING | INSERTING | V_SPLIT_MASK)) | bumped;
        self.version.store(new, Ordering::Release);
    }

    /// Move the upper half of this full leaf into `right` (a fresh, not-yet-
    /// published leaf) and return the separator (the right leaf's first
    /// key): keys `< sep` stay here, keys `>= sep` live in `right`. Caller
    /// holds the writer lock and has set SPLITTING on `self`.
    ///
    /// # Safety
    /// `self` must be a full leaf and `right` an unpublished fresh node.
    unsafe fn leaf_split_into(&self, right: &Node) -> Key {
        let n = self.n_keys.load(Ordering::Relaxed) as usize;
        let mid = n / 2;
        let mut j = 0;
        for i in mid..n {
            right.keys[j].store(self.keys[i].load(Ordering::Relaxed), Ordering::Relaxed);
            right.values[j].store(self.values[i].load(Ordering::Relaxed), Ordering::Relaxed);
            j += 1;
        }
        right.kind.store(NodeKind::Leaf as u8, Ordering::Relaxed);
        right.n_keys.store(j as u8, Ordering::Relaxed);
        let sep_word = right.keys[0].load(Ordering::Relaxed);
        // B-link fences: `right` inherits self's old upper bound; self's new
        // upper bound is the separator. A reader stranded on the truncated
        // self with a key `>= sep` follows next_leaf to `right`.
        right.high_key.store(self.high_key.load(Ordering::Relaxed), Ordering::Relaxed);
        self.high_key.store(sep_word, Ordering::Relaxed);
        // Truncate self to the lower half.
        self.n_keys.store(mid as u8, Ordering::Relaxed);
        sep_word.to_be_bytes()
    }

    /// Insert separator `sep` with right child `right` into this internal
    /// node, keeping it sorted. Returns false (no change) if the node is
    /// full. Caller holds the writer lock; the matching unlock publishes.
    ///
    /// # Safety
    /// `self` must be an internal node under the writer lock.
    unsafe fn internal_insert(&self, sep: &Key, right: NodeId) -> bool {
        let sw = key_word(sep);
        let n = self.n_keys.load(Ordering::Relaxed) as usize;
        if n >= MAX_KEYS {
            return false;
        }
        let mut pos = n;
        for i in 0..n {
            if self.keys[i].load(Ordering::Relaxed) > sw {
                pos = i;
                break;
            }
        }
        self.version.fetch_or(INSERTING, Ordering::Release);
        // Shift keys [pos..n) and children [pos+1..n+1) up by one.
        let mut i = n;
        while i > pos {
            self.keys[i].store(self.keys[i - 1].load(Ordering::Relaxed), Ordering::Relaxed);
            i -= 1;
        }
        let mut c = n + 1;
        while c > pos + 1 {
            self.children[c].store(self.children[c - 1].load(Ordering::Relaxed), Ordering::Relaxed);
            c -= 1;
        }
        self.keys[pos].store(sw, Ordering::Relaxed);
        self.children[pos + 1].store(right, Ordering::Relaxed);
        self.n_keys.store((n + 1) as u8, Ordering::Relaxed);
        true
    }

    /// Split this full internal node, moving the upper half into `right`
    /// (a fresh, unpublished node) and returning the promoted median key
    /// (which goes up to the parent, not into either half). `self` keeps
    /// `keys[0..mid)` / `children[0..mid]`; `right` gets `keys[mid+1..n)` /
    /// `children[mid+1..n+1)`. Sets the B-link fences; the caller wires
    /// `next_leaf` with the sibling ids.
    ///
    /// # Safety
    /// `self` must be a full internal node under the writer lock; `right`
    /// an unpublished fresh node.
    unsafe fn internal_split_into(&self, right: &Node) -> Key {
        let n = self.n_keys.load(Ordering::Relaxed) as usize;
        let mid = n / 2;
        let median = self.keys[mid].load(Ordering::Relaxed);
        let mut j = 0;
        for i in (mid + 1)..n {
            right.keys[j].store(self.keys[i].load(Ordering::Relaxed), Ordering::Relaxed);
            j += 1;
        }
        for (cj, c) in ((mid + 1)..(n + 1)).enumerate() {
            right.children[cj].store(self.children[c].load(Ordering::Relaxed), Ordering::Relaxed);
        }
        right.kind.store(NodeKind::Internal as u8, Ordering::Relaxed);
        right.n_keys.store(j as u8, Ordering::Relaxed);
        // B-link fences: `right` inherits self's old upper bound; self's new
        // upper bound is the promoted median.
        right.high_key.store(self.high_key.load(Ordering::Relaxed), Ordering::Relaxed);
        self.high_key.store(median, Ordering::Relaxed);
        self.n_keys.store(mid as u8, Ordering::Relaxed);
        median.to_be_bytes()
    }
}

/// Outcome of a successful [`Tree::insert`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Inserted {
    /// The key was new; a slot was added.
    New,
    /// The key already existed; its value was replaced.
    Updated,
}

/// Internal result of a single leaf put.
enum LeafPut {
    Done(Inserted),
    Full,
}

/// Why an [`Tree::insert`] could not complete.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InsertError {
    /// The fixed node pool is exhausted.
    PoolExhausted,
    /// A leaf split propagated a separator into an internal node that is
    /// itself full. Recursive internal-node splitting lands in a later
    /// commit; until then the tree is limited to one internal level.
    TreeTooDeep,
}

/// Concurrent B+tree. Readers take `&Tree` and never latch. Writers
/// serialize on a single tree-level lock (fine-grained writer concurrency
/// is a later refinement) while readers stay lock-free via the version
/// protocol.
pub struct Tree {
    pool: [Node; POOL_SIZE],
    root: AtomicU32,
    /// Bump allocator cursor into `pool`. Deletes retire through EBR and
    /// (later) recycle; for now allocation only moves forward.
    alloc_next: AtomicU32,
    /// Tree-wide writer lock: 0 free, 1 held. Structural changes (splits)
    /// touch multiple nodes, so all writers serialize here.
    writer: AtomicU8,
}

impl Tree {
    pub const fn new() -> Self {
        const EMPTY_NODE: Node = Node::EMPTY;
        Self {
            pool: [EMPTY_NODE; POOL_SIZE],
            root: AtomicU32::new(NULL_NODE),
            alloc_next: AtomicU32::new(0),
            writer: AtomicU8::new(0),
        }
    }

    fn writer_lock(&self) {
        while self
            .writer
            .compare_exchange_weak(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            core::hint::spin_loop();
        }
    }

    fn writer_unlock(&self) {
        self.writer.store(0, Ordering::Release);
    }

    #[inline]
    fn node(&self, id: NodeId) -> &Node {
        &self.pool[id as usize]
    }

    /// Hand out a fresh node of `kind` from the pool. The returned node is
    /// zeroed (empty leaf/internal) and not yet linked into the tree, so
    /// its non-atomic header can be initialized without synchronization.
    fn alloc(&self, kind: NodeKind) -> Option<NodeId> {
        let id = self.alloc_next.fetch_add(1, Ordering::AcqRel);
        if id as usize >= POOL_SIZE {
            return None;
        }
        // This id was just claimed and is not reachable from the root yet,
        // so no other thread can observe the node until we publish it (CAS
        // into root or a child slot). `EMPTY` already zeroed the pool.
        let node = self.node(id);
        node.kind.store(kind as u8, Ordering::Release);
        node.n_keys.store(0, Ordering::Release);
        node.version.store(0, Ordering::Release);
        Some(id)
    }

    /// Insert `value` under `key`, replacing any existing value. Writers
    /// serialize on the tree lock; lock-free readers stay correct via the
    /// version protocol. Splits a full leaf and grows the root as needed.
    pub fn insert(&self, key: Key, value: Value) -> Result<Inserted, InsertError> {
        self.writer_lock();
        let r = self.insert_locked(&key, value);
        self.writer_unlock();
        r
    }

    fn insert_locked(&self, key: &Key, value: Value) -> Result<Inserted, InsertError> {
        // Ensure a root leaf exists (we hold the writer lock, so no race).
        let mut root = self.root.load(Ordering::Acquire);
        if root == NULL_NODE {
            root = self.alloc(NodeKind::Leaf).ok_or(InsertError::PoolExhausted)?;
            self.root.store(root, Ordering::Release);
        }

        // Descend to the target leaf, recording the full path of internal
        // ancestors so a split can cascade the separator upward.
        let mut path = [NULL_NODE; MAX_DEPTH];
        let mut depth = 0usize;
        let mut node_id = root;
        while NodeKind::from_u8(self.node(node_id).kind.load(Ordering::Acquire))
            == NodeKind::Internal
        {
            if depth >= MAX_DEPTH {
                return Err(InsertError::TreeTooDeep);
            }
            path[depth] = node_id;
            depth += 1;
            // Safety: single writer; internal layout is stable here.
            node_id = unsafe { self.node(node_id).internal_descend(key) };
        }

        let leaf = self.node(node_id);
        leaf.lock();
        // Safety: writer lock held; readers retry on the version.
        match unsafe { leaf.leaf_insert(key, value) } {
            LeafPut::Done(ins) => {
                leaf.unlock_bump_insert();
                Ok(ins)
            }
            // `split_and_propagate` owns the leaf's unlock on every path.
            LeafPut::Full => self.split_and_propagate(node_id, &path, depth, root, key, value),
        }
    }

    /// The leaf `leaf_id` is full. Split it, then cascade the separator up
    /// `path[0..depth]`, splitting any full internal ancestor, and grow a
    /// new root if the split reaches the top. Caller holds the writer lock
    /// and `leaf_id` is locked; this releases that lock on every path.
    ///
    /// Correctness against readers uses the Lehman-Yao B-link right links:
    /// each split sets the left node's `next_leaf` to its new right sibling
    /// before unlocking, so the sibling is reachable via the link the moment
    /// the (truncated) left node becomes visible — the parent pointer is
    /// only an optimization that shortens future descents.
    fn split_and_propagate(
        &self,
        leaf_id: NodeId,
        path: &[NodeId; MAX_DEPTH],
        depth: usize,
        root: NodeId,
        key: &Key,
        value: Value,
    ) -> Result<Inserted, InsertError> {
        let leaf = self.node(leaf_id);

        // Pre-allocate every node the cascade might need, so once we start
        // mutating there is no fallible step that could wedge the tree: one
        // right leaf, plus up to `depth` right internals and one new root.
        let right_leaf = match self.alloc(NodeKind::Leaf) {
            Some(id) => id,
            None => {
                leaf.unlock_bump_insert();
                return Err(InsertError::PoolExhausted);
            }
        };
        let mut spares = [NULL_NODE; MAX_DEPTH + 1];
        for spare in spares.iter_mut().take(depth + 1) {
            match self.alloc(NodeKind::Internal) {
                // Unused spares just leak in the bump pool; harmless.
                Some(id) => *spare = id,
                None => {
                    leaf.unlock_bump_insert();
                    return Err(InsertError::PoolExhausted);
                }
            }
        }

        // Split the leaf.
        leaf.version.fetch_or(SPLITTING, Ordering::Release);
        let right = self.node(right_leaf);
        // Safety: writer lock held; `right` is fresh and unpublished.
        let sep = unsafe { leaf.leaf_split_into(right) };
        right.next_leaf.store(leaf.next_leaf.load(Ordering::Relaxed), Ordering::Relaxed);
        leaf.next_leaf.store(right_leaf, Ordering::Relaxed);
        let target = if key_word(key) < key_word(&sep) { leaf } else { right };
        // Safety: exclusive under the lock / unpublished; room guaranteed.
        let ins = match unsafe { target.leaf_insert(key, value) } {
            LeafPut::Done(i) => i,
            LeafPut::Full => unreachable!("half of a fresh split always has room"),
        };
        right.version.store(0, Ordering::Release);
        // `right` is now reachable via `leaf.next_leaf`; release the leaf.
        leaf.unlock_bump_split();

        // Cascade (cur_sep, cur_right) up the ancestors.
        let mut cur_sep = sep;
        let mut cur_right = right_leaf;
        let mut spare_idx = 0usize;
        // Stays true only if the cascade runs off the top and needs a new root.
        let mut grow = true;
        for level in (0..depth).rev() {
            let parent = self.node(path[level]);
            parent.lock();
            // Safety: writer lock + node lock held.
            if unsafe { parent.internal_insert(&cur_sep, cur_right) } {
                parent.unlock_bump_insert();
                grow = false;
                break;
            }
            // Parent is full: split it, promoting a median further up.
            let new_int = spares[spare_idx];
            spare_idx += 1;
            parent.version.fetch_or(SPLITTING, Ordering::Release);
            let ni = self.node(new_int);
            // Safety: parent locked; `ni` fresh and unpublished.
            let median = unsafe { parent.internal_split_into(ni) };
            ni.next_leaf.store(parent.next_leaf.load(Ordering::Relaxed), Ordering::Relaxed);
            parent.next_leaf.store(new_int, Ordering::Relaxed);
            // Insert (cur_sep, cur_right) into whichever half owns it.
            let ok = if key_word(&cur_sep) < key_word(&median) {
                unsafe { parent.internal_insert(&cur_sep, cur_right) }
            } else {
                unsafe { ni.internal_insert(&cur_sep, cur_right) }
            };
            debug_assert!(ok, "a half of a fresh internal split always has room");
            ni.version.store(0, Ordering::Release);
            // `new_int` is reachable via `parent.next_leaf`; release parent.
            parent.unlock_bump_split();
            cur_sep = median;
            cur_right = new_int;
        }

        if grow {
            // The split ran off the top; grow a new root over it. The old top
            // is the root we started from (the leaf itself when the tree was a
            // single leaf, else the topmost internal on the path).
            let old_top = if depth == 0 { leaf_id } else { root };
            self.install_root(spares[spare_idx], old_top, &cur_sep, cur_right);
            spare_idx += 1;
        }

        // Return the internal spares we over-provisioned but never used. We
        // allocate `right_leaf` then `depth + 1` spares contiguously, so the
        // unused ones (`spares[spare_idx..]`) are the top of the bump region;
        // none were published and we alone hold the writer lock, so no other
        // allocation can sit above them. Rewinding `alloc_next` reclaims them.
        let unused = (depth + 1) - spare_idx;
        if unused > 0 {
            self.alloc_next.fetch_sub(unused as u32, Ordering::AcqRel);
        }
        Ok(ins)
    }

    /// Populate the pre-allocated internal node `root_id` as a new root over
    /// `left`/`right` split by `sep`, and publish it.
    fn install_root(&self, root_id: NodeId, left: NodeId, sep: &Key, right: NodeId) {
        let root = self.node(root_id);
        root.keys[0].store(key_word(sep), Ordering::Relaxed);
        root.children[0].store(left, Ordering::Relaxed);
        root.children[1].store(right, Ordering::Relaxed);
        root.n_keys.store(1, Ordering::Relaxed);
        root.version.store(0, Ordering::Release);
        // Publish: readers load `root` with Acquire and re-descend.
        self.root.store(root_id, Ordering::Release);
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

    /// Lock-free descent to the leaf whose range covers `key`, returning its
    /// id (or None for an empty tree). Retries from the root on any
    /// in-flight writer, like `lookup`.
    fn leaf_for_read(&self, key: &Key) -> Option<NodeId> {
        'restart: loop {
            let mut id = self.root.load(Ordering::Acquire);
            if id == NULL_NODE {
                return None;
            }
            loop {
                let node = self.node(id);
                let v1 = node.load_version(Ordering::Acquire);
                if v1.is_locked() || v1.is_inserting() || v1.is_splitting() {
                    core::hint::spin_loop();
                    continue 'restart;
                }
                let kind = NodeKind::from_u8(node.kind.load(Ordering::Acquire));
                // Safety: bracketed by the version check below.
                let child = if kind == NodeKind::Leaf {
                    None
                } else if key_word(key) >= node.high_key.load(Ordering::Acquire)
                    && node.next_leaf.load(Ordering::Acquire) != NULL_NODE
                {
                    // Internal B-link: follow the right sibling.
                    Some(node.next_leaf.load(Ordering::Acquire))
                } else {
                    Some(unsafe { node.internal_descend(key) })
                };
                let v2 = node.load_version(Ordering::Acquire);
                if v1 != v2 || v2.is_inserting() || v2.is_splitting() {
                    core::hint::spin_loop();
                    continue 'restart;
                }
                match child {
                    None => return Some(id),
                    Some(next) => id = next,
                }
            }
        }
    }

    /// Visit every `(key, value)` with `start <= key < end`, in ascending
    /// key order. Lock-free: each leaf is emitted from a version-stable
    /// snapshot and the B-link `next_leaf` chain is followed across leaf
    /// boundaries. Per-leaf consistent, but not a single linearizable
    /// snapshot of the whole range — a key inserted concurrently mid-scan
    /// may or may not appear.
    pub fn range<F: FnMut(Key, Value)>(&self, start: &Key, end: &Key, mut visit: F) {
        let start_w = key_word(start);
        let end_w = key_word(end);
        if start_w >= end_w {
            return;
        }
        let mut leaf_id = match self.leaf_for_read(start) {
            Some(id) => id,
            None => return,
        };
        loop {
            let leaf = self.node(leaf_id);
            let mut buf: [(u64, u64); MAX_KEYS] = [(0, 0); MAX_KEYS];
            let count;
            let next;
            let reached_end;
            // Snapshot this leaf's in-range entries under version validation.
            loop {
                let v1 = leaf.load_version(Ordering::Acquire);
                if v1.is_locked() || v1.is_inserting() || v1.is_splitting() {
                    core::hint::spin_loop();
                    continue;
                }
                let mut c = 0;
                let mut done = false;
                let n = (leaf.n_keys.load(Ordering::Acquire) as usize).min(MAX_KEYS);
                for i in 0..n {
                    let kw = leaf.keys[i].load(Ordering::Acquire);
                    if kw >= end_w {
                        done = true;
                        break;
                    }
                    if kw >= start_w {
                        buf[c] = (kw, leaf.values[i].load(Ordering::Acquire));
                        c += 1;
                    }
                }
                let nl = leaf.next_leaf.load(Ordering::Acquire);
                let v2 = leaf.load_version(Ordering::Acquire);
                if v1 == v2 && !v2.is_inserting() && !v2.is_splitting() {
                    count = c;
                    next = nl;
                    reached_end = done;
                    break;
                }
                core::hint::spin_loop();
            }
            for &(kw, v) in buf.iter().take(count) {
                visit(kw.to_be_bytes(), v);
            }
            if reached_end || next == NULL_NODE {
                return;
            }
            leaf_id = next;
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
        match NodeKind::from_u8(node.kind.load(Ordering::Acquire)) {
            NodeKind::Leaf => match unsafe { node.leaf_scan(key) } {
                Some(v) => Some(NextStep::Found(v)),
                None => {
                    // B-link: if the key is at or past this leaf's upper
                    // fence, a split moved it to the right sibling after we
                    // routed here — follow the link instead of reporting it
                    // absent. Bracketed by the same version check as the
                    // scan, so a mid-split read is rejected and retried.
                    if key_word(key) >= node.high_key.load(Ordering::Acquire) {
                        let next = node.next_leaf.load(Ordering::Acquire);
                        if next != NULL_NODE {
                            return Some(NextStep::Descend(next));
                        }
                    }
                    Some(NextStep::Absent)
                }
            },
            NodeKind::Internal => {
                // B-link, internal level: if the key is at or past this
                // node's fence, a split moved the covering subtree to the
                // right sibling after we routed here — follow the right link
                // instead of descending a now-wrong child.
                if key_word(key) >= node.high_key.load(Ordering::Acquire) {
                    let right = node.next_leaf.load(Ordering::Acquire);
                    if right != NULL_NODE {
                        return Some(NextStep::Descend(right));
                    }
                }
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
        node.kind.store(NodeKind::Leaf as u8, Ordering::Release);
        node.n_keys.store(entries.len() as u8, Ordering::Release);
        for (i, (k, v)) in entries.iter().enumerate() {
            node.keys[i].store(key_word(k), Ordering::Release);
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
            let new_v = (v & !LOCKED).wrapping_add(1u64 << V_INSERT_SHIFT);
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

        let v = Version((7u64 << V_INSERT_SHIFT) | (11u64 << V_SPLIT_SHIFT));
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

    #[test]
    fn insert_keeps_keys_sorted_and_readable() {
        let t = Tree::new();
        // Insert out of order; lookups must find every key.
        for x in [5u64, 1, 9, 3, 7, 2, 8, 4, 6] {
            assert_eq!(t.insert(k(x), x * 100), Ok(Inserted::New));
        }
        for x in 1..=9u64 {
            assert_eq!(t.lookup(&k(x)), Some(x * 100));
        }
        assert_eq!(t.lookup(&k(0)), None);
        assert_eq!(t.lookup(&k(10)), None);
    }

    #[test]
    fn insert_replaces_existing_value() {
        let t = Tree::new();
        assert_eq!(t.insert(k(4), 400), Ok(Inserted::New));
        assert_eq!(t.insert(k(4), 4000), Ok(Inserted::Updated));
        assert_eq!(t.lookup(&k(4)), Some(4000));
    }

    #[test]
    fn leaf_split_grows_the_tree() {
        let t = Tree::new();
        // Fill the root leaf, then one more forces a split into an internal
        // root over two leaves.
        for x in 0..=MAX_KEYS as u64 {
            assert_eq!(t.insert(k(x), x + 1), Ok(Inserted::New));
        }
        for x in 0..=MAX_KEYS as u64 {
            assert_eq!(t.lookup(&k(x)), Some(x + 1), "key {x} lost across split");
        }
        // Updates still resolve to the right leaf after the split.
        assert_eq!(t.insert(k(0), 999), Ok(Inserted::Updated));
        assert_eq!(t.lookup(&k(0)), Some(999));
        assert_eq!(t.lookup(&k(9999)), None);
    }

    #[test]
    fn insert_routed_to_left_half_of_split_stays_readable() {
        // Fill the leaf with 1..=MAX_KEYS, then insert 0, which routes to the
        // LEFT half of the split. The split path inserts into that half
        // (setting INSERTING); the commit must clear it or a lookup on the
        // left leaf spins forever.
        let t = Tree::new();
        for x in 1..=MAX_KEYS as u64 {
            assert_eq!(t.insert(k(x), x + 1), Ok(Inserted::New));
        }
        assert_eq!(t.insert(k(0), 1), Ok(Inserted::New));
        assert_eq!(t.lookup(&k(0)), Some(1));
        for x in 1..=MAX_KEYS as u64 {
            assert_eq!(t.lookup(&k(x)), Some(x + 1), "key {x} lost");
        }
    }

    #[test]
    fn many_inserts_across_multiple_splits() {
        // Enough keys to cascade splits through several internal levels.
        // Insert in a scrambled order so both halves of every split get
        // exercised; with recursive internal splits every key must survive.
        let t = Tree::new();
        let n = 200u64;
        for i in 0..n {
            let x = (i.wrapping_mul(37) + 11) % n; // pseudo-shuffle, distinct
            t.insert(k(x), x + 1).expect("insert must not fail below pool size");
        }
        for x in 0..n {
            assert_eq!(t.lookup(&k(x)), Some(x + 1), "key {x} lost or wrong value");
        }
        // A key never inserted must read absent (right down the split fences).
        assert_eq!(t.lookup(&k(n)), None);
    }

    #[test]
    fn reader_is_consistent_across_a_split() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering as O};
        use std::thread;

        // A reader hammers lookups while a writer inserts enough keys to
        // split. Any value observed must match its key.
        let t = Arc::new(Tree::new());
        let done = Arc::new(AtomicBool::new(false));
        let tr = Arc::clone(&t);
        let dr = Arc::clone(&done);
        let reader = thread::spawn(move || {
            while !dr.load(O::Acquire) {
                for x in 0..(2 * MAX_KEYS as u64) {
                    if let Some(v) = tr.lookup(&k(x)) {
                        assert_eq!(v, x + 1, "key {x} read a mismatched value {v}");
                    }
                }
            }
        });
        for x in 0..(2 * MAX_KEYS as u64) {
            let _ = t.insert(k(x), x + 1);
        }
        done.store(true, O::Release);
        reader.join().unwrap();
    }

    #[test]
    fn concurrent_writers_disjoint_keys_all_land() {
        use std::sync::Arc;
        use std::thread;
        use std::vec::Vec;

        // 5 writers x 3 disjoint keys = 15 = MAX_KEYS, all into the one leaf.
        let t = Arc::new(Tree::new());
        let mut handles = Vec::new();
        for w in 0..5u64 {
            let tw = Arc::clone(&t);
            handles.push(thread::spawn(move || {
                for j in 0..3u64 {
                    let key = w * 3 + j;
                    assert_eq!(tw.insert(k(key), key + 1), Ok(Inserted::New));
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        for key in 0..15u64 {
            assert_eq!(t.lookup(&k(key)), Some(key + 1), "key {key} missing");
        }
    }

    #[test]
    fn concurrent_reader_never_sees_torn_state() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering as O};
        use std::thread;

        // A writer fills a leaf while a reader hammers lookups on a moving
        // set of keys. Every observed value must be the one paired with the
        // key (or absent) — never a mismatched or garbage value.
        let t = Arc::new(Tree::new());
        let done = Arc::new(AtomicBool::new(false));

        let tr = Arc::clone(&t);
        let dr = Arc::clone(&done);
        let reader = thread::spawn(move || {
            while !dr.load(O::Acquire) {
                for x in 0..MAX_KEYS as u64 {
                    if let Some(v) = tr.lookup(&k(x)) {
                        assert_eq!(v, x + 1, "key {x} read a mismatched value {v}");
                    }
                }
            }
        });

        for x in 0..MAX_KEYS as u64 {
            assert_eq!(t.insert(k(x), x + 1), Ok(Inserted::New));
        }
        done.store(true, O::Release);
        reader.join().unwrap();

        for x in 0..MAX_KEYS as u64 {
            assert_eq!(t.lookup(&k(x)), Some(x + 1));
        }
    }

    #[test]
    fn reader_never_loses_a_committed_key_during_splits() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as O};
        use std::thread;

        // The stale-route hazard: a reader routes to a leaf, the writer
        // splits it moving the key to a new sibling, and the reader reads the
        // truncated leaf as stable. A committed key must NEVER read absent —
        // the B-link `next_leaf` follow must carry the reader to the sibling.
        let t = Arc::new(Tree::new());
        let committed = Arc::new(AtomicU64::new(0)); // keys 0..committed are live
        let done = Arc::new(AtomicBool::new(false));

        let tr = Arc::clone(&t);
        let cr = Arc::clone(&committed);
        let dr = Arc::clone(&done);
        let reader = thread::spawn(move || {
            while !dr.load(O::Acquire) {
                let hi = cr.load(O::Acquire);
                for x in 0..hi {
                    assert_eq!(tr.lookup(&k(x)), Some(x + 1), "committed key {x} vanished");
                }
            }
        });

        let mut x = 0u64;
        while t.insert(k(x), x + 1).is_ok() {
            x += 1;
            committed.store(x, O::Release);
            if x > 200 {
                break;
            }
        }
        done.store(true, O::Release);
        reader.join().unwrap();
        assert!(x > MAX_KEYS as u64, "tree never split");
        for y in 0..x {
            assert_eq!(t.lookup(&k(y)), Some(y + 1));
        }
    }

    #[test]
    fn range_scans_in_key_order_across_splits() {
        use std::vec::Vec;

        let t = Tree::new();
        // Insert (in order) enough keys to build a multi-leaf tree.
        let mut inserted: Vec<u64> = Vec::new();
        let mut x = 0u64;
        while t.insert(k(x), x + 1).is_ok() {
            inserted.push(x);
            x += 1;
            if x > 60 {
                break;
            }
        }
        assert!(inserted.len() > MAX_KEYS, "tree never split");

        // Full range yields every key, ascending, with the right value.
        let mut seen: Vec<u64> = Vec::new();
        t.range(&k(0), &k(u64::MAX), |key, v| {
            let kx = u64::from_be_bytes(key);
            assert_eq!(v, kx + 1);
            seen.push(kx);
        });
        assert_eq!(seen, inserted, "full range: all keys, in order");

        // Bounded [10, 20): start inclusive, end exclusive.
        let mut mid: Vec<u64> = Vec::new();
        t.range(&k(10), &k(20), |key, _| mid.push(u64::from_be_bytes(key)));
        let expected: Vec<u64> = inserted
            .iter()
            .copied()
            .filter(|&y| (10..20).contains(&y))
            .collect();
        assert_eq!(mid, expected);

        // Empty ranges yield nothing.
        let mut n = 0;
        t.range(&k(20), &k(20), |_, _| n += 1);
        t.range(&k(30), &k(10), |_, _| n += 1);
        assert_eq!(n, 0);
    }

    #[test]
    fn range_on_single_leaf_and_empty_tree() {
        use std::vec::Vec;
        let t = Tree::new();
        // Empty tree: nothing.
        let mut n = 0;
        t.range(&k(0), &k(100), |_, _| n += 1);
        assert_eq!(n, 0);

        for x in [3u64, 1, 4, 1, 5, 9, 2, 6] {
            let _ = t.insert(k(x), x * 10);
        }
        let mut seen: Vec<u64> = Vec::new();
        t.range(&k(2), &k(6), |key, v| {
            let kx = u64::from_be_bytes(key);
            assert_eq!(v, kx * 10);
            seen.push(kx);
        });
        assert_eq!(seen, std::vec![2u64, 3, 4, 5]);
    }

    #[test]
    fn range_scan_is_safe_under_concurrent_splits() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering as O};
        use std::thread;

        // A scanner repeatedly walks the whole range while a writer inserts
        // and splits. Every emitted pair must be well-formed (value == key+1)
        // and strictly ascending, and the scan must never hang or fault.
        let t = Arc::new(Tree::new());
        let done = Arc::new(AtomicBool::new(false));
        let tr = Arc::clone(&t);
        let dr = Arc::clone(&done);
        let scanner = thread::spawn(move || {
            while !dr.load(O::Acquire) {
                let mut last: Option<u64> = None;
                tr.range(&k(0), &k(u64::MAX), |key, v| {
                    let kx = u64::from_be_bytes(key);
                    assert_eq!(v, kx + 1, "range saw a mismatched value");
                    if let Some(l) = last {
                        assert!(kx > l, "range out of order: {l} then {kx}");
                    }
                    last = Some(kx);
                });
            }
        });
        let mut x = 0u64;
        while t.insert(k(x), x + 1).is_ok() {
            x += 1;
            if x > 60 {
                break;
            }
        }
        done.store(true, O::Release);
        scanner.join().unwrap();
    }

    #[test]
    fn deep_tree_keeps_every_key_and_scans_in_order() {
        use std::vec::Vec;

        // A single internal root fans out to CHILD_SLOTS (16) leaves; past
        // ~16 * (MAX_KEYS/2) keys the root itself must split, forcing a second
        // internal level. Insert well beyond that so the cascade runs, and
        // check no key is lost and the leaf chain stays globally ordered.
        let t = Tree::new();
        let n = 220u64;
        for i in 0..n {
            // A full-period LCG over [0, n): distinct, scrambled order so both
            // sides of every leaf and internal split are exercised.
            let x = (i.wrapping_mul(97).wrapping_add(41)) % n;
            t.insert(k(x), x + 1).expect("insert must not fail below pool size");
        }
        for x in 0..n {
            assert_eq!(t.lookup(&k(x)), Some(x + 1), "key {x} lost in a deep tree");
        }
        // Full range walks the whole leaf chain: every key once, ascending.
        let mut seen: Vec<u64> = Vec::new();
        t.range(&k(0), &k(u64::MAX), |key, v| {
            let kx = u64::from_be_bytes(key);
            assert_eq!(v, kx + 1);
            seen.push(kx);
        });
        let expected: Vec<u64> = (0..n).collect();
        assert_eq!(seen, expected, "deep-tree range: all keys, in order");
    }

    #[test]
    fn reader_survives_cascading_internal_splits() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as O};
        use std::thread;

        // Push the writer past the single-internal-level boundary so the root
        // itself splits under a concurrent reader. Every already-committed key
        // must stay continuously visible through the whole restructuring — the
        // B-link at both the leaf and internal levels is what guarantees it.
        let t = Arc::new(Tree::new());
        let committed = Arc::new(AtomicU64::new(0));
        let done = Arc::new(AtomicBool::new(false));

        let tr = Arc::clone(&t);
        let cr = Arc::clone(&committed);
        let dr = Arc::clone(&done);
        let reader = thread::spawn(move || {
            while !dr.load(O::Acquire) {
                let hi = cr.load(O::Acquire);
                for x in 0..hi {
                    assert_eq!(tr.lookup(&k(x)), Some(x + 1), "committed key {x} vanished");
                }
            }
        });

        let n = 220u64; // forces several internal-node splits
        for x in 0..n {
            t.insert(k(x), x + 1).expect("insert must not fail below pool size");
            committed.store(x + 1, O::Release);
        }
        done.store(true, O::Release);
        reader.join().unwrap();
        for x in 0..n {
            assert_eq!(t.lookup(&k(x)), Some(x + 1));
        }
    }
}
