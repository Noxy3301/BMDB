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
//! Scope for Phase 3: `insert` + `lookup`. No delete, no range scan, no
//! concurrency. A static node pool + bump allocator stands in for a real
//! allocator; no nodes are reclaimed yet.

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
    next_free_idx: u32,
    root: NodeId,
}

impl BpTree {
    pub const fn new() -> Self {
        Self {
            pool: [EMPTY_NODE; POOL_SIZE],
            next_free_idx: 0,
            root: NULL_NODE,
        }
    }

    /// Number of node slots consumed so far. No reclamation yet, so this is
    /// also the peak.
    pub fn num_nodes(&self) -> u32 {
        self.next_free_idx
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
        let available = (POOL_SIZE as u32).saturating_sub(self.next_free_idx);
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

    fn alloc_node(&mut self, kind: NodeKind) -> Result<NodeId, InsertError> {
        if self.next_free_idx as usize >= POOL_SIZE {
            return Err(InsertError::NodePoolExhausted);
        }
        let id = self.next_free_idx;
        self.next_free_idx += 1;
        // The pool is pre-zeroed in `new()`; only the fields that vary per
        // node need to be reset.
        let node = &mut self.pool[id as usize];
        node.kind = kind;
        node.n_keys = 0;
        node.next_leaf = NULL_NODE;
        Ok(id)
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
