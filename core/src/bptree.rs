//! In-memory B+tree index.
//!
//! Classical Comer 1979 B+-tree. Values live only in leaves; internal nodes
//! hold separator keys and child pointers. Leaves are linked left-to-right
//! to support future range scans.
//!
//! Algorithm: bottom-up split propagation, following CMU 15-445 Lecture 08.
//! The leaf / internal split asymmetry matters: a leaf split *copies* the
//! split key up to the parent (the key also stays in the right leaf), while
//! an internal split *moves* the split key up (no duplicate in the routing
//! layer). See `.note/bptree-design.md`.
//!
//! Separator convention (half-open `[k_{i-1}, k_i)`): for an internal node
//! with keys `k_0..k_{n-1}` and children `c_0..c_n`, every key in `c_i` is
//! strictly less than `k_i`, and every key in `c_{i+1}` is at least `k_i`.
//!
//! Supported operations: `insert`, `lookup`, `delete`. No range scan, no
//! concurrency. The node pool is a static array; allocation uses a bump
//! pointer with a free list of nodes freed by merges during delete, so
//! delete-then-insert traffic does not leak capacity.

use core::fmt;

pub type Key = [u8; 8];
pub type Value = [u8; 8];

/// Order of the tree. Internal nodes hold up to `ORDER` children, leaves hold
/// up to `ORDER - 1` entries. Chosen to exercise splits in tests without
/// making the tree so shallow that bugs hide.
pub const ORDER: usize = 16;

const MAX_KEYS: usize = ORDER - 1;
// Storage is sized for one extra slot: during insert a node is allowed to
// overflow by one before it is split.
const KEY_SLOTS: usize = MAX_KEYS + 1;
const CHILD_SLOTS: usize = ORDER + 1;

// Minimum fill for a non-root node. Leaves and internal nodes both use
// MAX_KEYS / 2 so that any two sibling nodes at the minimum can be merged
// into a single node without overflow: 2 * MIN <= MAX_KEYS, and for internal
// merges 2 * MIN + 1 (the pulled-down separator) <= MAX_KEYS still holds.
const MIN_LEAF_KEYS: usize = MAX_KEYS / 2;
const MIN_INTERNAL_KEYS: usize = MAX_KEYS / 2;

/// How many nodes the static pool holds. 64 * size_of::<Node>() fits on the
/// kernel stack with room to spare.
pub const POOL_SIZE: usize = 64;

pub type NodeId = u32;
pub const NULL_NODE: NodeId = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    Leaf,
    Internal,
}

#[derive(Clone, Copy)]
struct Node {
    kind: NodeKind,
    n_keys: u16,
    keys: [Key; KEY_SLOTS],
    // Leaf: `values[i]` is the value paired with `keys[i]`.
    values: [Value; KEY_SLOTS],
    // Internal: `children[i]` is the subtree rooted between keys[i-1] and keys[i].
    children: [NodeId; CHILD_SLOTS],
    // Leaf only: next leaf in key order, or NULL_NODE at the right edge.
    next_leaf: NodeId,
}

const EMPTY_NODE: Node = Node {
    kind: NodeKind::Leaf,
    n_keys: 0,
    keys: [[0; 8]; KEY_SLOTS],
    values: [[0; 8]; KEY_SLOTS],
    children: [NULL_NODE; CHILD_SLOTS],
    next_leaf: NULL_NODE,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertError {
    DuplicateKey,
    NodePoolExhausted,
}

pub struct BpTree {
    pool: [Node; POOL_SIZE],
    // Bump allocator: next slot that has never been handed out. Monotonic.
    next_free_idx: u32,
    // Head of the free list, threaded through each freed node's `next_leaf`
    // field (unused for non-leaves, harmless to overwrite for leaves since
    // a freed leaf is no longer part of the leaf chain). NULL_NODE when empty.
    free_head: NodeId,
    // Number of nodes currently on the free list. `next_free_idx - free_count`
    // is the live node count.
    free_count: u32,
    root: NodeId,
}

impl BpTree {
    pub const fn new() -> Self {
        Self {
            pool: [EMPTY_NODE; POOL_SIZE],
            next_free_idx: 0,
            free_head: NULL_NODE,
            free_count: 0,
            root: NULL_NODE,
        }
    }

    /// Live node count — slots consumed by the bump allocator minus slots
    /// currently on the free list. Decreases as merges free nodes.
    pub fn num_nodes(&self) -> u32 {
        self.next_free_idx - self.free_count
    }

    /// Height from root to any leaf. Returns 0 for an empty tree, 1 when the
    /// root is a leaf.
    pub fn height(&self) -> u32 {
        if self.root == NULL_NODE {
            return 0;
        }
        let mut node_id = self.root;
        let mut depth: u32 = 1;
        loop {
            let node = &self.pool[node_id as usize];
            match node.kind {
                NodeKind::Leaf => return depth,
                NodeKind::Internal => {
                    node_id = node.children[0];
                    depth += 1;
                }
            }
        }
    }

    pub fn insert(&mut self, key: Key, value: Value) -> Result<(), InsertError> {
        // One split may happen at each level along the descent, plus a new
        // root if the top splits. Reserve that many slots upfront so every
        // `alloc_node` call below is guaranteed to succeed — otherwise a
        // failed alloc mid-insert would leave a node in an overfull state.
        let required: u32 = if self.root == NULL_NODE {
            1
        } else {
            self.height() + 1
        };
        let available = (POOL_SIZE as u32).saturating_sub(self.num_nodes());
        if available < required {
            return Err(InsertError::NodePoolExhausted);
        }

        if self.root == NULL_NODE {
            let leaf_id = self.alloc_node(NodeKind::Leaf)?;
            let leaf = &mut self.pool[leaf_id as usize];
            leaf.keys[0] = key;
            leaf.values[0] = value;
            leaf.n_keys = 1;
            self.root = leaf_id;
            return Ok(());
        }

        let result = self.insert_rec(self.root, key, value)?;
        if let Some((sep, new_right)) = result {
            let new_root_id = self.alloc_node(NodeKind::Internal)?;
            let old_root = self.root;
            let root_node = &mut self.pool[new_root_id as usize];
            root_node.keys[0] = sep;
            root_node.children[0] = old_root;
            root_node.children[1] = new_right;
            root_node.n_keys = 1;
            self.root = new_root_id;
        }
        Ok(())
    }

    pub fn lookup(&self, key: Key) -> Option<Value> {
        if self.root == NULL_NODE {
            return None;
        }
        self.lookup_rec(self.root, key)
    }

    fn lookup_rec(&self, node_id: NodeId, key: Key) -> Option<Value> {
        let node = &self.pool[node_id as usize];
        let n = node.n_keys as usize;
        match node.kind {
            NodeKind::Leaf => {
                let mut i = 0;
                while i < n && node.keys[i] < key {
                    i += 1;
                }
                if i < n && node.keys[i] == key {
                    Some(node.values[i])
                } else {
                    None
                }
            }
            NodeKind::Internal => {
                let mut i = 0;
                while i < n && node.keys[i] <= key {
                    i += 1;
                }
                self.lookup_rec(node.children[i], key)
            }
        }
    }

    /// Insert into the subtree rooted at `node_id`. Returns `Some((sep, new_right))`
    /// when the subtree root split and the parent must absorb a new separator
    /// and right sibling; `None` when the insert was absorbed without split.
    fn insert_rec(
        &mut self,
        node_id: NodeId,
        key: Key,
        value: Value,
    ) -> Result<Option<(Key, NodeId)>, InsertError> {
        let kind = self.pool[node_id as usize].kind;
        match kind {
            NodeKind::Leaf => self.insert_into_leaf(node_id, key, value),
            NodeKind::Internal => self.insert_into_internal(node_id, key, value),
        }
    }

    fn insert_into_leaf(
        &mut self,
        leaf_id: NodeId,
        key: Key,
        value: Value,
    ) -> Result<Option<(Key, NodeId)>, InsertError> {
        let overflows = {
            let leaf = &mut self.pool[leaf_id as usize];
            let n = leaf.n_keys as usize;

            let mut i = 0;
            while i < n && leaf.keys[i] < key {
                i += 1;
            }
            if i < n && leaf.keys[i] == key {
                return Err(InsertError::DuplicateKey);
            }

            // Shift entries [i..n) one slot right to make room at i.
            let mut j = n;
            while j > i {
                leaf.keys[j] = leaf.keys[j - 1];
                leaf.values[j] = leaf.values[j - 1];
                j -= 1;
            }
            leaf.keys[i] = key;
            leaf.values[i] = value;
            leaf.n_keys += 1;

            leaf.n_keys as usize > MAX_KEYS
        };

        if overflows {
            Ok(Some(self.split_leaf(leaf_id)?))
        } else {
            Ok(None)
        }
    }

    fn insert_into_internal(
        &mut self,
        node_id: NodeId,
        key: Key,
        value: Value,
    ) -> Result<Option<(Key, NodeId)>, InsertError> {
        // Pick the child whose key range covers `key`: first i with key < keys[i],
        // or n if key is at/beyond the last separator.
        let child_id = {
            let node = &self.pool[node_id as usize];
            let n = node.n_keys as usize;
            let mut i = 0;
            while i < n && node.keys[i] <= key {
                i += 1;
            }
            node.children[i]
        };

        let Some((sep, new_right)) = self.insert_rec(child_id, key, value)? else {
            return Ok(None);
        };

        let overflows = {
            let node = &mut self.pool[node_id as usize];
            let n = node.n_keys as usize;

            let mut i = 0;
            while i < n && node.keys[i] < sep {
                i += 1;
            }

            // Shift keys [i..n) right by one.
            let mut j = n;
            while j > i {
                node.keys[j] = node.keys[j - 1];
                j -= 1;
            }
            // Shift children [i+1..=n) right by one.
            let mut j = n + 1;
            while j > i + 1 {
                node.children[j] = node.children[j - 1];
                j -= 1;
            }
            node.keys[i] = sep;
            node.children[i + 1] = new_right;
            node.n_keys += 1;

            node.n_keys as usize > MAX_KEYS
        };

        if overflows {
            Ok(Some(self.split_internal(node_id)?))
        } else {
            Ok(None)
        }
    }

    /// Split an over-full leaf. The left half keeps the first `mid` entries,
    /// the right half takes the rest. The new right leaf's first key is
    /// *copied* up as the separator — the key itself also stays in the leaf.
    fn split_leaf(&mut self, left_id: NodeId) -> Result<(Key, NodeId), InsertError> {
        let right_id = self.alloc_node(NodeKind::Leaf)?;

        let total = self.pool[left_id as usize].n_keys as usize;
        // Ceiling split so the right side is never smaller than the left.
        let mid = total.div_ceil(2);

        for j in mid..total {
            let idx = j - mid;
            self.pool[right_id as usize].keys[idx] = self.pool[left_id as usize].keys[j];
            self.pool[right_id as usize].values[idx] = self.pool[left_id as usize].values[j];
        }
        self.pool[right_id as usize].n_keys = (total - mid) as u16;
        self.pool[left_id as usize].n_keys = mid as u16;

        // Splice right into the leaf list between left and left's former successor.
        self.pool[right_id as usize].next_leaf = self.pool[left_id as usize].next_leaf;
        self.pool[left_id as usize].next_leaf = right_id;

        let sep = self.pool[right_id as usize].keys[0];
        Ok((sep, right_id))
    }

    /// Split an over-full internal node. The middle key *moves* up to the
    /// caller; it does not remain in either child.
    fn split_internal(&mut self, left_id: NodeId) -> Result<(Key, NodeId), InsertError> {
        let right_id = self.alloc_node(NodeKind::Internal)?;

        let total_keys = self.pool[left_id as usize].n_keys as usize;
        let mid = total_keys / 2;

        let sep = self.pool[left_id as usize].keys[mid];

        for j in (mid + 1)..total_keys {
            let idx = j - mid - 1;
            self.pool[right_id as usize].keys[idx] = self.pool[left_id as usize].keys[j];
        }
        for j in (mid + 1)..=total_keys {
            let idx = j - mid - 1;
            self.pool[right_id as usize].children[idx] = self.pool[left_id as usize].children[j];
        }

        self.pool[right_id as usize].n_keys = (total_keys - mid - 1) as u16;
        self.pool[left_id as usize].n_keys = mid as u16;

        Ok((sep, right_id))
    }

    /// Remove `key` from the tree. Returns the value that was stored, or
    /// `None` if the key was not present. Cannot fail: delete only frees
    /// nodes, it never allocates.
    ///
    /// Rebalancing follows the canonical B+tree rules (Comer 1979, Jannink
    /// 1995): on leaf underflow try to borrow from a sibling, otherwise
    /// merge with a sibling and propagate upward. Separator keys in
    /// ancestors are only rewritten by borrow/merge — a plain leaf delete
    /// leaves the routing layer untouched, even if the deleted key happens
    /// to equal an ancestor separator (Graefe, "Modern B-Tree Techniques").
    pub fn delete(&mut self, key: Key) -> Option<Value> {
        if self.root == NULL_NODE {
            return None;
        }
        let (value, _) = self.delete_rec(self.root, key);

        // Root collapse: an internal root left with a single child after a
        // child-level merge has no keys of its own. Promote the child.
        let root_kind = self.pool[self.root as usize].kind;
        let root_n = self.pool[self.root as usize].n_keys;
        if root_kind == NodeKind::Internal && root_n == 0 {
            let old_root = self.root;
            self.root = self.pool[old_root as usize].children[0];
            self.free_node(old_root);
        } else if root_kind == NodeKind::Leaf && root_n == 0 {
            let old_root = self.root;
            self.free_node(old_root);
            self.root = NULL_NODE;
        }
        value
    }

    /// Returns `(value_if_found, whether_this_node_now_underflows)`. The
    /// underflow flag is what the caller checks to decide whether to
    /// rebalance this node as one of its own children.
    fn delete_rec(&mut self, node_id: NodeId, key: Key) -> (Option<Value>, bool) {
        let kind = self.pool[node_id as usize].kind;
        match kind {
            NodeKind::Leaf => self.delete_from_leaf(node_id, key),
            NodeKind::Internal => self.delete_from_internal(node_id, key),
        }
    }

    fn delete_from_leaf(&mut self, leaf_id: NodeId, key: Key) -> (Option<Value>, bool) {
        let leaf = &mut self.pool[leaf_id as usize];
        let n = leaf.n_keys as usize;

        let mut i = 0;
        while i < n && leaf.keys[i] < key {
            i += 1;
        }
        if i >= n || leaf.keys[i] != key {
            return (None, false);
        }

        let value = leaf.values[i];
        let mut j = i;
        while j + 1 < n {
            leaf.keys[j] = leaf.keys[j + 1];
            leaf.values[j] = leaf.values[j + 1];
            j += 1;
        }
        leaf.n_keys -= 1;

        let underflow = (leaf.n_keys as usize) < MIN_LEAF_KEYS;
        (Some(value), underflow)
    }

    fn delete_from_internal(&mut self, node_id: NodeId, key: Key) -> (Option<Value>, bool) {
        // Descent rule matches lookup: on `key == separator`, go right. This
        // keeps delete consistent with the half-open `[k_{i-1}, k_i)`
        // convention used everywhere else.
        let child_idx = {
            let node = &self.pool[node_id as usize];
            let n = node.n_keys as usize;
            let mut i = 0;
            while i < n && node.keys[i] <= key {
                i += 1;
            }
            i
        };
        let child_id = self.pool[node_id as usize].children[child_idx];
        let (value, child_underflow) = self.delete_rec(child_id, key);

        if child_underflow {
            self.rebalance_child(node_id, child_idx);
        }

        let underflow = (self.pool[node_id as usize].n_keys as usize) < MIN_INTERNAL_KEYS;
        (value, underflow)
    }

    /// Restore the fill invariant for `children[child_idx]` of `parent_id`.
    /// Prefers borrowing from a sibling (cheaper — one node touched beyond
    /// the underflowing one); falls back to a merge (frees one node,
    /// shrinks the parent by one entry).
    fn rebalance_child(&mut self, parent_id: NodeId, child_idx: usize) {
        let parent_n = self.pool[parent_id as usize].n_keys as usize;
        let child_id = self.pool[parent_id as usize].children[child_idx];
        let child_is_leaf = self.pool[child_id as usize].kind == NodeKind::Leaf;
        let min = if child_is_leaf { MIN_LEAF_KEYS } else { MIN_INTERNAL_KEYS };

        if child_idx > 0 {
            let left_id = self.pool[parent_id as usize].children[child_idx - 1];
            if (self.pool[left_id as usize].n_keys as usize) > min {
                self.borrow_from_left(parent_id, child_idx);
                return;
            }
        }
        if child_idx < parent_n {
            let right_id = self.pool[parent_id as usize].children[child_idx + 1];
            if (self.pool[right_id as usize].n_keys as usize) > min {
                self.borrow_from_right(parent_id, child_idx);
                return;
            }
        }

        // No sibling has slack: merge. Always merge the right sibling into
        // the left, so the freed node is the right one. If the underflowing
        // child has no left sibling, it becomes the left of the merge pair.
        if child_idx > 0 {
            self.merge_children(parent_id, child_idx - 1);
        } else {
            self.merge_children(parent_id, child_idx);
        }
    }

    /// Move one entry from the left sibling to the front of `children[child_idx]`.
    /// Updates the parent separator to the new first key of the right child.
    fn borrow_from_left(&mut self, parent_id: NodeId, child_idx: usize) {
        let left_id = self.pool[parent_id as usize].children[child_idx - 1];
        let right_id = self.pool[parent_id as usize].children[child_idx];
        let right_kind = self.pool[right_id as usize].kind;

        match right_kind {
            NodeKind::Leaf => {
                let left_n = self.pool[left_id as usize].n_keys as usize;
                let right_n = self.pool[right_id as usize].n_keys as usize;
                let borrowed_key = self.pool[left_id as usize].keys[left_n - 1];
                let borrowed_val = self.pool[left_id as usize].values[left_n - 1];

                // Shift right-child entries one slot right to open index 0.
                let right = &mut self.pool[right_id as usize];
                let mut j = right_n;
                while j > 0 {
                    right.keys[j] = right.keys[j - 1];
                    right.values[j] = right.values[j - 1];
                    j -= 1;
                }
                right.keys[0] = borrowed_key;
                right.values[0] = borrowed_val;
                right.n_keys += 1;

                self.pool[left_id as usize].n_keys -= 1;

                // New separator is the new first key of the right child.
                self.pool[parent_id as usize].keys[child_idx - 1] = borrowed_key;
            }
            NodeKind::Internal => {
                // Rotate through the parent: parent_sep moves down to the
                // front of the right child, and the left child's last key
                // moves up to become the new separator.
                let left_n = self.pool[left_id as usize].n_keys as usize;
                let right_n = self.pool[right_id as usize].n_keys as usize;
                let parent_sep = self.pool[parent_id as usize].keys[child_idx - 1];
                let new_sep = self.pool[left_id as usize].keys[left_n - 1];
                let moved_child = self.pool[left_id as usize].children[left_n];

                // Shift right's keys/children right by one to make room at 0.
                let right = &mut self.pool[right_id as usize];
                let mut j = right_n;
                while j > 0 {
                    right.keys[j] = right.keys[j - 1];
                    j -= 1;
                }
                let mut j = right_n + 1;
                while j > 0 {
                    right.children[j] = right.children[j - 1];
                    j -= 1;
                }
                right.keys[0] = parent_sep;
                right.children[0] = moved_child;
                right.n_keys += 1;

                self.pool[left_id as usize].n_keys -= 1;
                self.pool[parent_id as usize].keys[child_idx - 1] = new_sep;
            }
        }
    }

    /// Move one entry from the right sibling to the end of `children[child_idx]`.
    /// Updates the parent separator to the new first key of the right sibling.
    fn borrow_from_right(&mut self, parent_id: NodeId, child_idx: usize) {
        let left_id = self.pool[parent_id as usize].children[child_idx];
        let right_id = self.pool[parent_id as usize].children[child_idx + 1];
        let right_kind = self.pool[right_id as usize].kind;

        match right_kind {
            NodeKind::Leaf => {
                let left_n = self.pool[left_id as usize].n_keys as usize;
                let right_n = self.pool[right_id as usize].n_keys as usize;
                let borrowed_key = self.pool[right_id as usize].keys[0];
                let borrowed_val = self.pool[right_id as usize].values[0];

                self.pool[left_id as usize].keys[left_n] = borrowed_key;
                self.pool[left_id as usize].values[left_n] = borrowed_val;
                self.pool[left_id as usize].n_keys += 1;

                // Shift right-child entries one slot left.
                let right = &mut self.pool[right_id as usize];
                let mut j = 0;
                while j + 1 < right_n {
                    right.keys[j] = right.keys[j + 1];
                    right.values[j] = right.values[j + 1];
                    j += 1;
                }
                right.n_keys -= 1;

                // New separator is the new first key of the right child.
                let new_sep = self.pool[right_id as usize].keys[0];
                self.pool[parent_id as usize].keys[child_idx] = new_sep;
            }
            NodeKind::Internal => {
                // Rotate through the parent: parent_sep moves down to the
                // end of the left child, and the right child's first key
                // moves up to become the new separator.
                let left_n = self.pool[left_id as usize].n_keys as usize;
                let right_n = self.pool[right_id as usize].n_keys as usize;
                let parent_sep = self.pool[parent_id as usize].keys[child_idx];
                let new_sep = self.pool[right_id as usize].keys[0];
                let moved_child = self.pool[right_id as usize].children[0];

                let left = &mut self.pool[left_id as usize];
                left.keys[left_n] = parent_sep;
                left.children[left_n + 1] = moved_child;
                left.n_keys += 1;

                // Shift right's keys/children left by one.
                let right = &mut self.pool[right_id as usize];
                let mut j = 0;
                while j + 1 < right_n {
                    right.keys[j] = right.keys[j + 1];
                    j += 1;
                }
                let mut j = 0;
                while j + 1 <= right_n {
                    right.children[j] = right.children[j + 1];
                    j += 1;
                }
                right.n_keys -= 1;

                self.pool[parent_id as usize].keys[child_idx] = new_sep;
            }
        }
    }

    /// Merge `children[left_idx + 1]` into `children[left_idx]`. The freed
    /// right sibling is returned to the free list. The parent loses one
    /// separator and one child pointer.
    fn merge_children(&mut self, parent_id: NodeId, left_idx: usize) {
        let left_id = self.pool[parent_id as usize].children[left_idx];
        let right_id = self.pool[parent_id as usize].children[left_idx + 1];
        let kind = self.pool[left_id as usize].kind;

        match kind {
            NodeKind::Leaf => {
                let left_n = self.pool[left_id as usize].n_keys as usize;
                let right_n = self.pool[right_id as usize].n_keys as usize;

                for j in 0..right_n {
                    let k = self.pool[right_id as usize].keys[j];
                    let v = self.pool[right_id as usize].values[j];
                    self.pool[left_id as usize].keys[left_n + j] = k;
                    self.pool[left_id as usize].values[left_n + j] = v;
                }
                self.pool[left_id as usize].n_keys = (left_n + right_n) as u16;
                // Splice the absorbed leaf out of the forward chain.
                self.pool[left_id as usize].next_leaf =
                    self.pool[right_id as usize].next_leaf;
            }
            NodeKind::Internal => {
                // Pull the separator between the two siblings down into the
                // merged node: internal merge is (left) + (sep) + (right).
                let left_n = self.pool[left_id as usize].n_keys as usize;
                let right_n = self.pool[right_id as usize].n_keys as usize;
                let sep = self.pool[parent_id as usize].keys[left_idx];

                self.pool[left_id as usize].keys[left_n] = sep;
                for j in 0..right_n {
                    let k = self.pool[right_id as usize].keys[j];
                    self.pool[left_id as usize].keys[left_n + 1 + j] = k;
                }
                for j in 0..=right_n {
                    let c = self.pool[right_id as usize].children[j];
                    self.pool[left_id as usize].children[left_n + 1 + j] = c;
                }
                self.pool[left_id as usize].n_keys = (left_n + 1 + right_n) as u16;
            }
        }

        // Remove parent separator at left_idx and child pointer at left_idx + 1.
        let parent = &mut self.pool[parent_id as usize];
        let p_n = parent.n_keys as usize;
        let mut j = left_idx;
        while j + 1 < p_n {
            parent.keys[j] = parent.keys[j + 1];
            j += 1;
        }
        let mut j = left_idx + 1;
        while j < p_n {
            parent.children[j] = parent.children[j + 1];
            j += 1;
        }
        parent.n_keys -= 1;

        self.free_node(right_id);
    }

    fn alloc_node(&mut self, kind: NodeKind) -> Result<NodeId, InsertError> {
        // Prefer the free list: reusing a merge-freed slot keeps the bump
        // pointer from walking off the end of the pool under steady-state
        // delete/insert churn.
        let id = if self.free_head != NULL_NODE {
            let id = self.free_head;
            self.free_head = self.pool[id as usize].next_leaf;
            self.free_count -= 1;
            id
        } else {
            if self.next_free_idx as usize >= POOL_SIZE {
                return Err(InsertError::NodePoolExhausted);
            }
            let id = self.next_free_idx;
            self.next_free_idx += 1;
            id
        };
        // Reset per-node state. keys/values/children retain stale bytes from
        // the previous tenant, but they are never read before being
        // overwritten — n_keys = 0 gates all of them.
        let node = &mut self.pool[id as usize];
        node.kind = kind;
        node.n_keys = 0;
        node.next_leaf = NULL_NODE;
        Ok(id)
    }

    fn free_node(&mut self, id: NodeId) {
        let node = &mut self.pool[id as usize];
        node.n_keys = 0;
        node.next_leaf = self.free_head;
        self.free_head = id;
        self.free_count += 1;
    }
}

impl Default for BpTree {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for BpTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BpTree {{ root={}, nodes={}, height={} }}",
            self.root,
            self.num_nodes(),
            self.height()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k(i: u64) -> Key {
        i.to_be_bytes()
    }
    fn v(i: u64) -> Value {
        (i * 1000).to_be_bytes()
    }

    #[test]
    fn lookup_on_empty_tree_returns_none() {
        let tree = BpTree::new();
        assert_eq!(tree.lookup(k(42)), None);
        assert_eq!(tree.height(), 0);
        assert_eq!(tree.num_nodes(), 0);
    }

    #[test]
    fn insert_single_then_lookup() {
        let mut tree = BpTree::new();
        tree.insert(k(1), v(1)).unwrap();
        assert_eq!(tree.lookup(k(1)), Some(v(1)));
        assert_eq!(tree.lookup(k(2)), None);
        assert_eq!(tree.height(), 1);
        assert_eq!(tree.num_nodes(), 1);
    }

    #[test]
    fn duplicate_key_is_rejected() {
        let mut tree = BpTree::new();
        tree.insert(k(5), v(5)).unwrap();
        assert_eq!(tree.insert(k(5), v(99)), Err(InsertError::DuplicateKey));
        // Original value must still be intact.
        assert_eq!(tree.lookup(k(5)), Some(v(5)));
    }

    #[test]
    fn insert_many_sequential_keys_all_found() {
        let mut tree = BpTree::new();
        for i in 1..=50u64 {
            tree.insert(k(i), v(i)).unwrap();
        }
        for i in 1..=50u64 {
            assert_eq!(tree.lookup(k(i)), Some(v(i)), "missing key {}", i);
        }
        assert!(tree.height() >= 2, "50 keys should trigger at least one split");
    }

    #[test]
    fn insert_many_reverse_keys_all_found() {
        let mut tree = BpTree::new();
        for i in (1..=50u64).rev() {
            tree.insert(k(i), v(i)).unwrap();
        }
        for i in 1..=50u64 {
            assert_eq!(tree.lookup(k(i)), Some(v(i)), "missing key {}", i);
        }
    }

    #[test]
    fn lookup_between_existing_keys_returns_none() {
        let mut tree = BpTree::new();
        for i in [2u64, 4, 6, 8, 10] {
            tree.insert(k(i), v(i)).unwrap();
        }
        for i in [1u64, 3, 5, 7, 9, 11] {
            assert_eq!(tree.lookup(k(i)), None, "phantom hit on key {}", i);
        }
    }

    #[test]
    fn first_split_creates_internal_root() {
        let mut tree = BpTree::new();
        // MAX_KEYS = ORDER - 1 = 15. The 16th insert forces a leaf split and
        // allocates a new internal root.
        for i in 1..=16u64 {
            tree.insert(k(i), v(i)).unwrap();
        }
        assert_eq!(tree.height(), 2, "first split should make root internal");
        assert_eq!(tree.num_nodes(), 3, "root + left leaf + right leaf");
    }

    #[test]
    fn delete_from_empty_tree_returns_none() {
        let mut tree = BpTree::new();
        assert_eq!(tree.delete(k(1)), None);
        assert_eq!(tree.num_nodes(), 0);
        assert_eq!(tree.height(), 0);
    }

    #[test]
    fn delete_nonexistent_key_returns_none_and_keeps_others() {
        let mut tree = BpTree::new();
        tree.insert(k(1), v(1)).unwrap();
        tree.insert(k(3), v(3)).unwrap();
        assert_eq!(tree.delete(k(2)), None);
        assert_eq!(tree.lookup(k(1)), Some(v(1)));
        assert_eq!(tree.lookup(k(3)), Some(v(3)));
    }

    #[test]
    fn delete_only_key_empties_tree() {
        let mut tree = BpTree::new();
        tree.insert(k(42), v(42)).unwrap();
        assert_eq!(tree.delete(k(42)), Some(v(42)));
        assert_eq!(tree.lookup(k(42)), None);
        assert_eq!(tree.num_nodes(), 0);
        assert_eq!(tree.height(), 0);
    }

    #[test]
    fn delete_without_underflow_preserves_siblings() {
        // A single leaf above MIN loses one entry; no rebalance needed.
        let mut tree = BpTree::new();
        for i in 1..=10u64 {
            tree.insert(k(i), v(i)).unwrap();
        }
        assert_eq!(tree.delete(k(5)), Some(v(5)));
        assert_eq!(tree.lookup(k(5)), None);
        for i in [1u64, 2, 3, 4, 6, 7, 8, 9, 10] {
            assert_eq!(tree.lookup(k(i)), Some(v(i)));
        }
    }

    #[test]
    fn delete_triggers_leaf_borrow_from_right() {
        // After 16 sequential inserts the tree is one internal root and two
        // leaves of 8 entries each. Deleting twice from the left leaf drops
        // it to 6 (< MIN=7), which forces a borrow from the right sibling.
        let mut tree = BpTree::new();
        for i in 1..=16u64 {
            tree.insert(k(i), v(i)).unwrap();
        }
        let nodes_before = tree.num_nodes();
        assert_eq!(tree.delete(k(1)), Some(v(1))); // leaf at MIN, no underflow
        assert_eq!(tree.delete(k(2)), Some(v(2))); // underflow → borrow

        // Borrow (not merge): the tree shape is unchanged.
        assert_eq!(tree.num_nodes(), nodes_before);
        assert_eq!(tree.height(), 2);
        for i in 3..=16u64 {
            assert_eq!(tree.lookup(k(i)), Some(v(i)), "lost {} after borrow", i);
        }
    }

    #[test]
    fn delete_triggers_leaf_borrow_from_left() {
        // Symmetric to borrow_from_right: delete twice from the right leaf
        // to force it to borrow from its left sibling.
        let mut tree = BpTree::new();
        for i in 1..=16u64 {
            tree.insert(k(i), v(i)).unwrap();
        }
        let nodes_before = tree.num_nodes();
        assert_eq!(tree.delete(k(16)), Some(v(16)));
        assert_eq!(tree.delete(k(15)), Some(v(15)));

        assert_eq!(tree.num_nodes(), nodes_before);
        assert_eq!(tree.height(), 2);
        for i in 1..=14u64 {
            assert_eq!(tree.lookup(k(i)), Some(v(i)), "lost {} after borrow", i);
        }
    }

    #[test]
    fn delete_triggers_leaf_merge_and_root_collapse() {
        // 16 inserts → 2 leaves of 8. Two deletes borrow once (both leaves
        // down to 7). A third delete from the left underflows it to 6; the
        // right sibling is at MIN so cannot lend — they merge into one leaf
        // and the internal root, now empty, is collapsed away.
        let mut tree = BpTree::new();
        for i in 1..=16u64 {
            tree.insert(k(i), v(i)).unwrap();
        }
        assert_eq!(tree.delete(k(1)), Some(v(1)));
        assert_eq!(tree.delete(k(2)), Some(v(2)));
        assert_eq!(tree.delete(k(3)), Some(v(3)));

        assert_eq!(tree.height(), 1, "root should collapse to single leaf");
        assert_eq!(tree.num_nodes(), 1);
        for i in 4..=16u64 {
            assert_eq!(tree.lookup(k(i)), Some(v(i)), "lost {} after merge", i);
        }
    }

    #[test]
    fn delete_even_keys_preserves_odd_keys_in_deep_tree() {
        let mut tree = BpTree::new();
        for i in 1..=50u64 {
            tree.insert(k(i), v(i)).unwrap();
        }
        let start_height = tree.height();
        assert!(start_height >= 2);

        for i in (2..=50u64).step_by(2) {
            assert_eq!(tree.delete(k(i)), Some(v(i)), "delete({}) failed", i);
        }
        for i in (1..=49u64).step_by(2) {
            assert_eq!(tree.lookup(k(i)), Some(v(i)), "odd {} lost", i);
        }
        for i in (2..=50u64).step_by(2) {
            assert_eq!(tree.lookup(k(i)), None, "phantom even {}", i);
        }
    }

    #[test]
    fn delete_every_key_empties_tree() {
        let mut tree = BpTree::new();
        for i in 1..=50u64 {
            tree.insert(k(i), v(i)).unwrap();
        }
        for i in 1..=50u64 {
            assert_eq!(tree.delete(k(i)), Some(v(i)), "delete({}) missing", i);
        }
        assert_eq!(tree.num_nodes(), 0);
        assert_eq!(tree.height(), 0);
        assert_eq!(tree.lookup(k(1)), None);
    }

    #[test]
    fn delete_then_reinsert_reuses_freed_nodes() {
        // After a merge-driven root collapse, two nodes sit on the free
        // list. The next three inserts must pull from the free list before
        // advancing the bump pointer, so the peak node count never exceeds
        // what the original shape required.
        let mut tree = BpTree::new();
        for i in 1..=16u64 {
            tree.insert(k(i), v(i)).unwrap();
        }
        let peak_nodes = tree.num_nodes();
        assert_eq!(peak_nodes, 3);

        assert_eq!(tree.delete(k(1)), Some(v(1)));
        assert_eq!(tree.delete(k(2)), Some(v(2)));
        assert_eq!(tree.delete(k(3)), Some(v(3)));
        assert_eq!(tree.num_nodes(), 1);

        for i in 1..=3u64 {
            tree.insert(k(i), v(i)).unwrap();
        }
        // Adding 3 keys to a single 13-entry leaf overflows it (16 > 15)
        // and forces one leaf split plus a new internal root — back to the
        // original 3-node shape.
        assert_eq!(tree.num_nodes(), peak_nodes);
        for i in 1..=16u64 {
            assert_eq!(tree.lookup(k(i)), Some(v(i)));
        }
    }

    #[test]
    fn delete_descends_past_stale_separator() {
        // When a deleted key equals an internal separator, descent must go
        // right (half-open routing), and the separator is left stale — the
        // deletion itself does not rewrite it. A subsequent lookup of the
        // deleted key must still return None.
        let mut tree = BpTree::new();
        for i in 1..=16u64 {
            tree.insert(k(i), v(i)).unwrap();
        }
        // Internal root's separator is k(9) (first key of the right leaf).
        assert_eq!(tree.delete(k(9)), Some(v(9)));
        assert_eq!(tree.lookup(k(9)), None);
        for i in (1..=16u64).filter(|&i| i != 9) {
            assert_eq!(tree.lookup(k(i)), Some(v(i)));
        }
    }

    #[test]
    fn delete_exercises_internal_rebalance_in_deep_tree() {
        // Covers merge/borrow at the internal level — not just the leaf
        // level. A tree of height >= 3 has at least one non-root internal
        // node, which is the only kind that can actually underflow (the
        // root is exempt from MIN). This test forces one by inserting
        // until the pool fills, then deletes every key and asserts the
        // pool returns to empty.
        let mut tree = BpTree::new();
        let mut last_inserted = 0u64;
        for i in 1..=2000u64 {
            match tree.insert(k(i), v(i)) {
                Ok(()) => last_inserted = i,
                Err(InsertError::NodePoolExhausted) => break,
                Err(e) => panic!("unexpected insert error: {:?}", e),
            }
        }
        assert!(
            tree.height() >= 3,
            "test needs height >= 3 for internal rebalance, got {}",
            tree.height()
        );

        // Delete forward; spot-check mid-run that the tree is still
        // well-formed by looking up keys that have not yet been deleted.
        for i in 1..=last_inserted {
            assert_eq!(tree.delete(k(i)), Some(v(i)), "delete({}) lost", i);
            if i % 32 == 0 {
                let probe_end = last_inserted.min(i + 16);
                for j in (i + 1)..=probe_end {
                    assert_eq!(tree.lookup(k(j)), Some(v(j)), "lookup({}) after delete({})", j, i);
                }
            }
        }
        assert_eq!(tree.num_nodes(), 0);
        assert_eq!(tree.height(), 0);
    }

    #[test]
    fn pool_exhaustion_is_reported() {
        let mut tree = BpTree::new();
        let mut last_err = None;
        // Insert until the pool refuses. Keys well under u32 range, no
        // duplicates.
        for i in 1..=10_000u64 {
            if let Err(e) = tree.insert(k(i), v(i)) {
                last_err = Some((i, e));
                break;
            }
        }
        let (stop_at, err) = last_err.expect("should eventually exhaust the pool");
        assert_eq!(err, InsertError::NodePoolExhausted);
        // Everything up to (stop_at - 1) was inserted successfully and must
        // still be readable — the capacity pre-check leaves the tree
        // untouched on failure.
        for i in 1..stop_at {
            assert_eq!(tree.lookup(k(i)), Some(v(i)));
        }
    }
}
