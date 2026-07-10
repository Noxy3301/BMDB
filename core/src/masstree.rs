// Concurrent single-layer Masstree: a B+tree with lock-free readers and
// per-node-locked writers, specialized to fixed 8-byte keys.
//
// Faithful port of the concurrency protocol of masstree-beta (Eddie Kohler,
// Yandong Mao, Robert Morris; (c) 2012-2016 Harvard, MIT): reach_leaf and
// advance_to_key from masstree_struct.hh, find_unlocked from masstree_get.hh,
// find_locked/find_insert from masstree_get.hh/masstree_insert.hh, and
// make_split/split_into from masstree_split.hh, over the already-ported
// version word (nodeversion.rs) and leaf permuter (kpermuter.rs).
//
// Deliberate divergences from the reference, each marked at its site:
//
//  * Storage: the C++ heap-allocates nodes and reclaims them through RCU.
//    Here all nodes live in two fixed pools inside the tree (one per node
//    kind, because a node's isleaf bit is baked into its version word at
//    const-init time) and are named by a u32 NodeId. Allocation is a bump
//    pointer plus a free list of never-published internodes; reclamation of
//    published nodes is deferred to a later round, so the pools only grow.
//
//  * Keys: exactly one layer. A key is one u64 ikey (big-endian bytes, so
//    integer compare == lexicographic compare). All of ksuf/keylenx/layer
//    machinery is deleted; a leaf slot is just (ikey, value). Ikeys within
//    the tree are unique, so split_into's same-ikey regrouping loop is dead
//    and dropped.
//
//  * remove() is omitted this round (TODO), which also kills the reference's
//    modstate_ machinery (it only guards remove/insert transitions) and the
//    marked-pointer dance in btree_leaflink (it only guards concurrent
//    unlink).
//
//  * Memory model: the C++ uses plain loads/stores ordered by compiler-only
//    fences plus x86-TSO. In Rust every field shared with lock-free readers
//    must be an atomic. The mapping, justified like nodeversion.rs: node
//    contents (ikeys, values, permutation, nkeys) are Relaxed, bracketed by
//    the version protocol -- stable()'s trailing acquire fence orders them
//    after the version load, has_changed()'s leading acquire fence orders
//    them before the re-read, and unlock()'s release store publishes them.
//    Link words (parent, child slots, leaf next/prev, the root id) are
//    published with Release stores and chased with Acquire loads: they hand
//    a reader a node the version protocol has not vouched for yet, so the
//    link itself must carry the happens-before that shows the node fully
//    built. This subsumes every fence() the reference issues before a link
//    store. The one extra pair: a leaf's permutation is published Release
//    and read Acquire, because a fresh-slot insert makes a slot live via the
//    permutation word alone, with no version bump (the reference leans on
//    TSO for that edge; see finish_insert).

use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::ebr::MAX_CPUS;
use crate::kpermuter::{Permuter, WIDTH};
use crate::nodeversion::{NodeVersion, Version, VSPLIT_LOWBIT};

/// Node name inside the pools. Values below LEAF_POOL index `leaves`, the
/// rest index `inters`. Pointer-sized links become u32 slots.
type NodeId = u32;
const NULL_ID: NodeId = u32::MAX;

/// Pool sizes. Two pools because the isleaf discriminant lives in the
/// version word, which is fixed at const-init and has no raw setter.
const LEAF_POOL: usize = 768;
const INTER_POOL: usize = 256;

/// Internode fan-out: WIDTH keys, WIDTH+1 children, kept sorted (internodes
/// do not use a permuter). Same 15 as the leaf permuter width.
const IWIDTH: i32 = WIDTH;

/// Upper bound on internodes one split cascade can allocate: one per level
/// plus one new root or height-gap filler. With INTER_POOL internodes and
/// post-split fan-out >= 8, tree height never exceeds 4, so 6 suffices;
/// 8 gives slack. Reserving this many up front (returning the unused ones
/// to the free list) means the cascade can never fail mid-flight, which
/// matters: an internode split truncates the parent before the new sibling
/// is reachable, so there is no consistent way to abandon it halfway.
const CASCADE_MAX: usize = 8;

/// Sentinel cpu for callers with no dedicated core (the engine/index path and
/// single-threaded tests). Allocation then goes straight to the central
/// atomic cursor -- correct under concurrency, just without the per-cpu fast
/// path. A real cpu index (0..MAX_CPUS) must be UNIQUE to the calling thread
/// for the duration, exactly the contract of ebr::enter(cpu).
const SHARED_CPU: usize = MAX_CPUS;

/// Pool slots a per-cpu cache grabs from the central cursor per refill; the
/// one shared atomic touch is amortized over this many allocations.
const REFILL_STRIDE: u32 = 16;

/// Per-cpu allocation cache: private bump ranges carved from the central
/// cursors plus a private LIFO of returned (never-published) internodes.
/// Single-writer (the owning cpu), so every field is a plain Relaxed
/// load/store -- no CAS on the hot insert/split path. align(64) so two cpus'
/// caches never share a cache line.
#[repr(C, align(64))]
struct PerCpuAlloc {
    leaf_cur: AtomicU32,
    leaf_end: AtomicU32,
    inter_cur: AtomicU32,
    inter_end: AtomicU32,
    /// LIFO head of returned internodes, linked through node.parent.
    inter_free: AtomicU32,
}

impl PerCpuAlloc {
    const EMPTY: Self = Self {
        leaf_cur: AtomicU32::new(0),
        leaf_end: AtomicU32::new(0),
        inter_cur: AtomicU32::new(0),
        inter_end: AtomicU32::new(0),
        inter_free: AtomicU32::new(NULL_ID),
    };
}

/// A cache-line-isolated atomic. `align(64)` so a read-hot field (the root
/// id, loaded on every descent) owns its line and is not invalidated by
/// writes to neighbouring header fields (the central refill cursors). Deref
/// keeps call sites reading `self.root.load(..)` unchanged.
#[repr(C, align(64))]
struct Padded(AtomicU32);

impl core::ops::Deref for Padded {
    type Target = AtomicU32;
    fn deref(&self) -> &AtomicU32 {
        &self.0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Inserted {
    New,
    Updated,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InsertError {
    PoolExhausted,
}

/// One pool slot; serves as leaf or internode depending on which pool it
/// sits in (leaf fields and internode fields are disjoint; the few bytes of
/// waste buy a single uniform struct). Every field a concurrent reader can
/// touch is atomic -- a plain shared field would be UB in Rust regardless
/// of the version protocol (same rule nodeversion.rs documents).
///
/// `align(64)` so no two nodes share a cache line: without it, two cores
/// splitting adjacent-in-pool leaves would false-share each other's version
/// word and bounce the line even though the nodes are logically disjoint
/// (cbptree already aligns its node for the same reason).
#[repr(C, align(64))]
struct Node {
    version: NodeVersion,
    /// Leaf only: the kpermuter word ordering the 15 slots.
    permutation: AtomicU64,
    /// Internode only: number of live keys (children = nkeys + 1).
    nkeys: AtomicU32,
    /// Internode only: 1 + height of its children; leaves are height 0.
    height: AtomicU32,
    parent: AtomicU32,
    /// Leaf only: B-link chain.
    next: AtomicU32,
    prev: AtomicU32,
    ikey0: [AtomicU64; WIDTH as usize],
    /// Leaf only: value per slot, in slot (not logical) order.
    lv: [AtomicU64; WIDTH as usize],
    /// Internode only: child_[i] as in the reference.
    child: [AtomicU32; (IWIDTH + 1) as usize],
}

impl Node {
    const fn new_const(isleaf: bool) -> Node {
        const Z64: AtomicU64 = AtomicU64::new(0);
        const ZID: AtomicU32 = AtomicU32::new(NULL_ID);
        Node {
            version: NodeVersion::new(isleaf),
            // Not a valid permuter word; alloc_leaf() stores make_empty()
            // (not a const fn) before the node is ever published.
            permutation: AtomicU64::new(0),
            nkeys: AtomicU32::new(0),
            height: AtomicU32::new(0),
            parent: ZID,
            next: ZID,
            prev: ZID,
            ikey0: [Z64; WIDTH as usize],
            lv: [Z64; WIDTH as usize],
            child: [ZID; (IWIDTH + 1) as usize],
        }
    }

    fn leaf_perm(&self) -> Permuter {
        // ordering: Acquire pairs with the Release in publish_perm(); a
        // reader that sees a permutation exposing a fresh slot must also
        // see that slot's ikey/value stores. The reference gets this edge
        // from fence() + TSO.
        Permuter::from_value(self.permutation.load(Ordering::Acquire))
    }

    fn publish_perm(&self, p: Permuter) {
        // ordering: the reference writes the slot, fence(), then stores the
        // permutation word; Release is that StoreStore edge.
        self.permutation.store(p.value(), Ordering::Release);
    }

    fn leaf_size(&self) -> i32 {
        Permuter::size_of(self.permutation.load(Ordering::Relaxed))
    }

    fn ikey(&self, p: i32) -> u64 {
        self.ikey0[p as usize].load(Ordering::Relaxed)
    }

    /// Slot 0 physically holds the leaf's lower fence even when that key is
    /// no longer logically present (why insert never reuses slot 0 freely).
    fn ikey_bound(&self) -> u64 {
        self.ikey0[0].load(Ordering::Relaxed)
    }

    /// leaf::assign for fixed keys: fill a free slot's ikey and value. The
    /// slot only goes live when a later publish_perm() exposes it, so
    /// Relaxed stores suffice under that Release.
    fn assign(&self, p: i32, ikey: u64, value: u64) {
        self.ikey0[p as usize].store(ikey, Ordering::Relaxed);
        self.lv[p as usize].store(value, Ordering::Relaxed);
    }

    fn isize(&self) -> i32 {
        self.nkeys.load(Ordering::Relaxed) as i32
    }
}

fn cmp3(a: u64, b: u64) -> i32 {
    if a < b {
        -1
    } else if a == b {
        0
    } else {
        1
    }
}

/// XOR-delta split check between two stable snapshots, standing in for the
/// reference's `newv.has_split(oldv)` called on a by-value nodeversion
/// (our NodeVersion::has_split re-reads the live word instead).
fn split_delta(a: Version, b: Version) -> bool {
    (a.value() ^ b.value()) >= VSPLIT_LOWBIT
}

/// key_indexed_position: logical position i, physical slot p (-1 = miss).
#[derive(Clone, Copy)]
struct Kx {
    i: i32,
    p: i32,
}

/// Internode ids the current split cascade may still consume; unused ones
/// go back to the tree's free list when the cascade completes.
struct Reserve {
    ids: [NodeId; CASCADE_MAX],
    n: usize,
}

impl Reserve {
    fn take(&mut self) -> NodeId {
        // Unreachable empty by the CASCADE_MAX bound; checked in debug.
        debug_assert!(self.n > 0, "split cascade outran its reserve");
        if self.n == 0 {
            return NULL_ID;
        }
        self.n -= 1;
        self.ids[self.n]
    }
}

pub struct Masstree {
    leaves: [Node; LEAF_POOL],
    inters: [Node; INTER_POOL],
    /// basic_table root_; NULL until the first put installs the root leaf.
    /// Loaded on every descent, so isolated on its own cache line -- writes to
    /// the central refill cursors below must not invalidate it.
    root: Padded,
    /// Central bump cursors: leaf ids are 0..LEAF_POOL; the inter cursor is
    /// pool-relative (id = LEAF_POOL + cursor). The SHARED path bumps these
    /// directly; per-cpu caches refill a REFILL_STRIDE window from them.
    /// fetch_add only -- no lock. Pools only grow (reclamation is a later,
    /// EBR-backed step).
    leaf_central: AtomicU32,
    inter_central: AtomicU32,
    /// Allocation caches. Slots 0..MAX_CPUS are per-cpu, touched only by the
    /// owning cpu, so disjoint-key writers never contend on an alloc line.
    /// The extra slot at index MAX_CPUS (== SHARED_CPU) serves callers with no
    /// dedicated core; it uses the same cache/free-list logic but under
    /// `shared_lock`, so it too recycles reserves (no leak) -- just not
    /// lock-free. Each slot's align(64) keeps it off the header line above.
    caches: [PerCpuAlloc; MAX_CPUS + 1],
    /// Serializes access to the shared (index MAX_CPUS) cache only; per-cpu
    /// slots never take it. Off the per-cpu hot path.
    shared_lock: AtomicU32,
    /// Startup-only lock serializing the one-time root-leaf install so
    /// concurrent first-put contenders allocate exactly one root between them
    /// (no loser-leaf leak). Distinct from `shared_lock` so ensure_root, which
    /// holds this while calling alloc_leaf (which may take shared_lock), never
    /// self-deadlocks. NOT on the allocation hot path.
    init_lock: AtomicU32,
}

impl Masstree {
    pub const fn new() -> Self {
        const LEAF_INIT: Node = Node::new_const(true);
        const INTER_INIT: Node = Node::new_const(false);
        const CACHE_INIT: PerCpuAlloc = PerCpuAlloc::EMPTY;
        Masstree {
            leaves: [LEAF_INIT; LEAF_POOL],
            inters: [INTER_INIT; INTER_POOL],
            root: Padded(AtomicU32::new(NULL_ID)),
            leaf_central: AtomicU32::new(0),
            inter_central: AtomicU32::new(0),
            caches: [CACHE_INIT; MAX_CPUS + 1],
            shared_lock: AtomicU32::new(0),
            init_lock: AtomicU32::new(0),
        }
    }

    fn node(&self, id: NodeId) -> &Node {
        let i = id as usize;
        if i < LEAF_POOL {
            &self.leaves[i]
        } else {
            &self.inters[i - LEAF_POOL]
        }
    }
}

impl Default for Masstree {
    fn default() -> Self {
        Self::new()
    }
}

impl Masstree {

    // ---- allocation ----------------------------------------------------
    //
    // A central atomic bump cursor is the source of truth; each cpu caches a
    // REFILL_STRIDE window of it and hands out slots with plain relaxed stores
    // (single-writer per cpu). Only a cache refill, a SHARED-path caller, or a
    // per-cpu internode free-list miss touches a shared line, so disjoint-key
    // inserts on different cpus don't contend on allocation. `cpu` must be
    // unique to the calling thread (or SHARED_CPU) -- see SHARED_CPU.

    /// True for the shared cache slot (index MAX_CPUS): its ops run under
    /// `shared_lock`; per-cpu slots are lock-free.
    fn is_shared(cpu: usize) -> bool {
        cpu >= MAX_CPUS
    }

    fn shared_lock_acquire(&self) {
        while self
            .shared_lock
            .compare_exchange_weak(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            core::hint::spin_loop();
        }
    }

    fn shared_lock_release(&self) {
        self.shared_lock.store(0, Ordering::Release);
    }

    /// Claim a refill window `[base, end)` from a central bump `cursor`,
    /// saturating at `pool`. Uses a CAS loop rather than `fetch_add` so the
    /// cursor NEVER advances past `pool`: a plain `fetch_add` would keep
    /// climbing on every post-exhaustion attempt and, after ~2^32 of them,
    /// wrap to 0 and reissue already-published ids. Returns None once the
    /// pool is exhausted.
    fn claim_window(cursor: &AtomicU32, pool: u32) -> Option<(u32, u32)> {
        let mut base = cursor.load(Ordering::Relaxed);
        loop {
            if base >= pool {
                return None;
            }
            let end = core::cmp::min(base + REFILL_STRIDE, pool);
            match cursor.compare_exchange_weak(base, end, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => return Some((base, end)),
                Err(cur) => base = cur,
            }
        }
    }

    fn alloc_leaf(&self, cpu: usize) -> Option<NodeId> {
        let shared = Self::is_shared(cpu);
        if shared {
            self.shared_lock_acquire();
        }
        let cache = &self.caches[if shared { MAX_CPUS } else { cpu }];
        let mut cur = cache.leaf_cur.load(Ordering::Relaxed);
        let id = if cur < cache.leaf_end.load(Ordering::Relaxed) {
            cache.leaf_cur.store(cur + 1, Ordering::Relaxed);
            Some(cur)
        } else {
            // Window exhausted: grab a fresh stride from the central cursor.
            match Self::claim_window(&self.leaf_central, LEAF_POOL as u32) {
                None => None,
                Some((base, end)) => {
                    cur = base;
                    cache.leaf_cur.store(cur + 1, Ordering::Relaxed);
                    cache.leaf_end.store(end, Ordering::Relaxed);
                    Some(cur)
                }
            }
        };
        if shared {
            self.shared_lock_release();
        }
        let id = id?;
        // The permutation word was const-initialized to 0, not a valid
        // permuter; fix it before anyone can see the node.
        self.leaves[id as usize]
            .permutation
            .store(Permuter::make_empty(), Ordering::Relaxed);
        Some(id)
    }

    fn alloc_inter(&self, cpu: usize) -> Option<NodeId> {
        let shared = Self::is_shared(cpu);
        if shared {
            self.shared_lock_acquire();
        }
        let cache = &self.caches[if shared { MAX_CPUS } else { cpu }];
        // Prefer a returned reserve from this slot's free list.
        let head = cache.inter_free.load(Ordering::Relaxed);
        let id = if head != NULL_ID {
            let next = self.node(head).parent.load(Ordering::Relaxed);
            cache.inter_free.store(next, Ordering::Relaxed);
            // Free-listed nodes are pristine except the repurposed link.
            self.node(head).parent.store(NULL_ID, Ordering::Relaxed);
            Some(head)
        } else {
            let mut cur = cache.inter_cur.load(Ordering::Relaxed);
            if cur < cache.inter_end.load(Ordering::Relaxed) {
                cache.inter_cur.store(cur + 1, Ordering::Relaxed);
                Some(LEAF_POOL as NodeId + cur)
            } else {
                match Self::claim_window(&self.inter_central, INTER_POOL as u32) {
                    None => None,
                    Some((base, end)) => {
                        cur = base;
                        cache.inter_cur.store(cur + 1, Ordering::Relaxed);
                        cache.inter_end.store(end, Ordering::Relaxed);
                        Some(LEAF_POOL as NodeId + cur)
                    }
                }
            }
        };
        if shared {
            self.shared_lock_release();
        }
        id
    }

    fn free_inter(&self, cpu: usize, id: NodeId) {
        let shared = Self::is_shared(cpu);
        if shared {
            self.shared_lock_acquire();
        }
        let cache = &self.caches[if shared { MAX_CPUS } else { cpu }];
        let head = cache.inter_free.load(Ordering::Relaxed);
        self.node(id).parent.store(head, Ordering::Relaxed);
        cache.inter_free.store(id, Ordering::Relaxed);
        if shared {
            self.shared_lock_release();
        }
    }

    fn reserve_internodes(&self, cpu: usize) -> Option<Reserve> {
        let mut r = Reserve {
            ids: [NULL_ID; CASCADE_MAX],
            n: 0,
        };
        while r.n < CASCADE_MAX {
            match self.alloc_inter(cpu) {
                Some(id) => {
                    r.ids[r.n] = id;
                    r.n += 1;
                }
                None => {
                    self.return_reserve(cpu, r);
                    return None;
                }
            }
        }
        Some(r)
    }

    fn return_reserve(&self, cpu: usize, r: Reserve) {
        for i in 0..r.n {
            self.free_inter(cpu, r.ids[i]);
        }
    }

    fn init_lock_acquire(&self) {
        while self
            .init_lock
            .compare_exchange_weak(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            core::hint::spin_loop();
        }
    }

    fn init_lock_release(&self) {
        self.init_lock.store(0, Ordering::Release);
    }

    // ---- root management -------------------------------------------------

    /// basic_table::initialize, lazily: install the first (empty) root leaf.
    /// Serialized so concurrent first-put contenders allocate exactly ONE
    /// leaf between them: an un-serialized CAS race would leak every loser's
    /// leaf into the bump pool (reclamation is deferred), and enough startup
    /// contenders could falsely exhaust the fixed leaf pool.
    fn ensure_root(&self, cpu: usize) -> Result<NodeId, InsertError> {
        let r = self.root.load(Ordering::Acquire);
        if r != NULL_ID {
            return Ok(r);
        }
        // Double-checked under the startup-only init lock, so only the first
        // thread allocates + installs the root leaf (no loser-leaf leak). The
        // init lock is off the allocation hot path.
        self.init_lock_acquire();
        let r = self.root.load(Ordering::Acquire);
        if r != NULL_ID {
            self.init_lock_release();
            return Ok(r);
        }
        let id = match self.alloc_leaf(cpu) {
            Some(id) => id,
            None => {
                self.init_lock_release();
                return Err(InsertError::PoolExhausted);
            }
        };
        // make_root: next/prev already NULL, ikey0[0] already 0 (the
        // reference zeroes it "to avoid undefined behavior"), parent NULL.
        self.node(id).version.mark_root();
        self.root.store(id, Ordering::Release);
        self.init_lock_release();
        Ok(id)
    }

    /// basic_table::fix_root: one step toward the real root; reach_leaf
    /// tolerates any remaining staleness.
    fn fix_root(&self, root: NodeId) -> NodeId {
        if self.node(root).version.is_root() {
            return root;
        }
        let p = self.node(root).parent.load(Ordering::Acquire);
        if p == NULL_ID {
            return root;
        }
        let _ = self
            .root
            .compare_exchange(root, p, Ordering::Release, Ordering::Relaxed);
        p
    }

    // ---- searches within one node ----------------------------------------

    /// key_bound::lower for leaves (linear, as the reference picks for
    /// width 15): logical position + slot of the match, slot -1 on miss.
    fn leaf_lower(&self, n: &Node, perm: Permuter, ika: u64) -> Kx {
        let size = perm.size();
        let mut l = 0;
        while l < size {
            let lp = perm.get(l);
            let c = cmp3(ika, n.ikey(lp));
            if c < 0 {
                break;
            } else if c == 0 {
                return Kx { i: l, p: lp };
            }
            l += 1;
        }
        Kx { i: l, p: -1 }
    }

    /// key_bound::upper for internodes: index of the child to descend into.
    fn inter_upper(&self, n: &Node, ika: u64) -> i32 {
        let size = n.isize();
        let mut l = 0;
        while l < size {
            let c = cmp3(ika, n.ikey(l));
            if c < 0 {
                break;
            } else if c == 0 {
                return l + 1;
            }
            l += 1;
        }
        l
    }

    /// leaf::stable_last_key_compare: compare ika against the node's last
    /// key, rerun until the comparison is stable. On an empty node the
    /// probed slot is harmless garbage and the result is only a routing
    /// hint the caller re-checks (the reference documents the same).
    fn leaf_stable_last_key_compare(&self, id: NodeId, ika: u64, mut v: Version) -> i32 {
        let n = self.node(id);
        loop {
            let perm = n.leaf_perm();
            let size = perm.size();
            let p = perm.get(if size > 0 { size - 1 } else { 0 });
            let c = cmp3(ika, n.ikey(p));
            if !n.version.has_changed(v) {
                return c;
            }
            v = n.version.stable();
        }
    }

    /// internode::stable_last_key_compare.
    fn inter_stable_last_key_compare(&self, id: NodeId, ika: u64, mut v: Version) -> i32 {
        let n = self.node(id);
        loop {
            let size = n.isize();
            let c = if size > 0 { cmp3(ika, n.ikey(size - 1)) } else { 1 };
            if !n.version.has_changed(v) {
                return c;
            }
            v = n.version.stable();
        }
    }

    // ---- lock-free descent ------------------------------------------------

    /// node_base::reach_leaf: return the leaf responsible for ika and its
    /// stable version. Staleness of the starting node is detected through
    /// the root bit (the true root has never split) and repaired by walking
    /// parent links; the two-slot `sense` buffer validates each parent
    /// against the child version it handed out.
    fn reach_leaf(&self, start: NodeId, ika: u64) -> (NodeId, Version) {
        let mut n: [NodeId; 2] = [NULL_ID; 2];
        let mut v: [Version; 2];
        'retry: loop {
            let mut sense = 0usize;
            n[0] = start;
            let seed = self.node(start).version.stable();
            v = [seed, seed];
            while !v[sense].is_root() {
                // maybe_parent: stay put if the parent link is not up yet
                // (a mid-publication window); the loop re-stabilizes.
                let p = self.node(n[sense]).parent.load(Ordering::Acquire);
                if p != NULL_ID {
                    n[sense] = p;
                }
                v[sense] = self.node(n[sense]).version.stable();
            }

            while !v[sense].is_leaf() {
                let in_id = n[sense];
                let kp = self.inter_upper(self.node(in_id), ika);
                // ordering: Acquire pairs with the Release publication of a
                // fresh child so its contents are visible before we read
                // them below.
                let next = self.node(in_id).child[kp as usize].load(Ordering::Acquire);
                if next == NULL_ID {
                    continue 'retry;
                }
                n[sense ^ 1] = next;
                v[sense ^ 1] = self.node(next).version.stable();

                if !self.node(in_id).version.has_changed(v[sense]) {
                    sense ^= 1;
                    continue;
                }

                let oldv = v[sense];
                v[sense] = self.node(in_id).version.stable();
                if split_delta(oldv, v[sense])
                    && self.inter_stable_last_key_compare(in_id, ika, v[sense]) > 0
                {
                    // The node split and ika now belongs to its right
                    // sibling, which only the ancestors know about:
                    // restart from the top.
                    continue 'retry;
                }
                // Otherwise the change stayed within this subtree; retry
                // the same internode with the fresher version.
            }
            return (n[sense], v[sense]);
        }
    }

    /// leaf::advance_to_key: if the leaf split since version v, ride the
    /// B-link next pointers right until the leaf whose bound covers ika,
    /// restabilizing v on each hop.
    fn advance_to_key(&self, start: NodeId, ika: u64, v: &mut Version) -> NodeId {
        let mut n = start;
        let oldv = *v;
        *v = self.node(n).version.stable();
        if split_delta(*v, oldv) && self.leaf_stable_last_key_compare(n, ika, *v) > 0 {
            loop {
                if v.deleted() {
                    break;
                }
                let next = self.node(n).next.load(Ordering::Acquire);
                if next == NULL_ID || ika < self.node(next).ikey_bound() {
                    break;
                }
                n = next;
                *v = self.node(n).version.stable();
            }
        }
        n
    }

    // ---- read path ----------------------------------------------------

    /// unlocked_tcursor::find_unlocked, minus the layer descent (fixed
    /// 8-byte keys have exactly one layer, so a slot match IS the answer).
    pub fn get(&self, key: [u8; 8]) -> Option<u64> {
        let ika = u64::from_be_bytes(key);
        let root = self.root.load(Ordering::Acquire);
        if root == NULL_ID {
            return None;
        }
        'retry: loop {
            let (mut n, mut v) = self.reach_leaf(root, ika);
            loop {
                if v.deleted() {
                    continue 'retry;
                }
                let node = self.node(n);
                let perm = node.leaf_perm();
                let kx = self.leaf_lower(node, perm, ika);
                let mut value = 0u64;
                if kx.p >= 0 {
                    value = node.lv[kx.p as usize].load(Ordering::Relaxed);
                }
                if node.version.has_changed(v) {
                    n = self.advance_to_key(n, ika, &mut v);
                    continue;
                }
                return if kx.p >= 0 { Some(value) } else { None };
            }
        }
    }

    // ---- write path -----------------------------------------------------

    /// tcursor::find_locked, minus layers and deleted_layer: return the
    /// locked leaf responsible for ika and the key's position in it.
    fn find_locked(&self, root: NodeId, ika: u64) -> (NodeId, Kx) {
        'retry: loop {
            let (mut n, mut v) = self.reach_leaf(root, ika);
            loop {
                if v.deleted() {
                    continue 'retry;
                }
                let node = self.node(n);
                let perm = node.leaf_perm();
                let kx = self.leaf_lower(node, perm, ika);
                node.version.lock();
                // The permutation check catches fresh-slot inserts, which
                // deliberately do not bump the version word.
                if node.version.has_changed(v)
                    || node.permutation.load(Ordering::Relaxed) != perm.value()
                {
                    node.version.unlock();
                    n = self.advance_to_key(n, ika, &mut v);
                    continue;
                }
                return (n, kx);
            }
        }
    }

    /// tcursor::finish_insert: expose the assigned slot at logical position
    /// kxi. The Release publish is the only step readers can observe.
    fn finish_insert(&self, n: NodeId, kxi: i32, kxp: i32) {
        let node = self.node(n);
        let mut perm = Permuter::from_value(node.permutation.load(Ordering::Relaxed));
        debug_assert!(perm.back() == kxp);
        let _ = kxp;
        perm.insert_from_back(kxi);
        node.publish_perm(perm);
    }

    /// basic_table::put with no dedicated cpu: allocates from the shared
    /// central cursor. Correct under concurrency, just without the per-cpu
    /// allocation fast path -- use [`put_on`](Self::put_on) with a unique cpu
    /// index to get that.
    pub fn put(&self, key: [u8; 8], value: u64) -> Result<Inserted, InsertError> {
        self.put_on(SHARED_CPU, key, value)
    }

    /// basic_table::put == find_insert + finish, allocating any new node from
    /// `cpu`'s private pool. `cpu` must be unique to the calling thread for
    /// the duration (or `SHARED_CPU`); see [`SHARED_CPU`]. Value updates on an
    /// existing key are a single atomic store into the slot (the reference
    /// hands the value cell to its caller; row-level concurrency control is
    /// out of tree scope there and here).
    pub fn put_on(&self, cpu: usize, key: [u8; 8], value: u64) -> Result<Inserted, InsertError> {
        self.put_inner(cpu, key, value, true).map(|(ins, _)| ins)
    }

    /// Get-or-insert: if `key` is present, return its EXISTING value without
    /// overwriting; otherwise insert `value` and return it. This is the
    /// primitive the engine's slot allocator needs -- a key must keep its
    /// first-assigned record slot even if two threads race to bind it. `cpu`
    /// contract as [`put_on`](Self::put_on).
    pub fn get_or_put_on(&self, cpu: usize, key: [u8; 8], value: u64) -> Result<u64, InsertError> {
        self.put_inner(cpu, key, value, false).map(|(_, v)| v)
    }

    /// Shared body of put_on/get_or_put_on. Returns `(outcome, mapped_value)`:
    /// on an existing key, `overwrite` decides whether the slot takes `value`
    /// (put) or keeps its current value (get-or-insert); either way the
    /// returned value is what the key now maps to.
    fn put_inner(
        &self,
        cpu: usize,
        key: [u8; 8],
        value: u64,
        overwrite: bool,
    ) -> Result<(Inserted, u64), InsertError> {
        let ika = u64::from_be_bytes(key);
        let root = self.fix_root(self.ensure_root(cpu)?);
        let (n, kx) = self.find_locked(root, ika);
        let node = self.node(n);

        if kx.p >= 0 {
            if overwrite {
                // Publish the new value through the version protocol. The
                // reference treats value concurrency as outside the tree (it
                // hands the caller the value cell), but here the value IS the
                // tree's payload, so an update must bump the version:
                // mark_insert() forces an in-flight reader to retry and its
                // release on unlock() pairs with a later reader's stable()
                // acquire, so no one reads a stale value behind a clean word.
                node.version.mark_insert();
                node.lv[kx.p as usize].store(value, Ordering::Relaxed);
                node.version.unlock();
                return Ok((Inserted::Updated, value));
            }
            // Get-or-insert: leave the value untouched, return the existing one.
            let existing = node.lv[kx.p as usize].load(Ordering::Relaxed);
            node.version.unlock();
            return Ok((Inserted::Updated, existing));
        }

        // find_insert: try the leaf's free slots first. (modstate_ dance
        // dropped: without remove the state is always modstate_insert.)
        if node.leaf_size() < WIDTH {
            let perm = Permuter::from_value(node.permutation.load(Ordering::Relaxed));
            let kxp = perm.back();
            // Don't inappropriately reuse slot 0, which holds ikey_bound.
            if kxp != 0
                || node.prev.load(Ordering::Relaxed) == NULL_ID
                || node.ikey_bound() == ika
            {
                node.assign(kxp, ika, value);
                self.finish_insert(n, kx.i, kxp);
                node.version.unlock();
                return Ok((Inserted::New, value));
            }
        }

        self.make_split(cpu, n, kx.i, ika, value)?;
        Ok((Inserted::New, value))
    }

    /// leaf::split_into: move the upper half of nl into the fresh locked
    /// leaf nr, place the new key's slot (but never its permuter entry --
    /// that waits for finish_insert), and publish nr through the B-link.
    /// Returns (split_ikey, split_type 0|1|2).
    fn leaf_split_into(&self, nl_id: NodeId, nr_id: NodeId, kxi: i32, ika: u64, value: u64) -> (u64, i32) {
        let nl = self.node(nl_id);
        let nr = self.node(nr_id);
        let perml = Permuter::from_value(nl.permutation.load(Ordering::Relaxed));
        let width = perml.size(); // WIDTH, or WIDTH-1 when slot 0 was blocked
        let mut mid = WIDTH / 2 + 1;
        let p = kxi;
        if p == 0 && nl.prev.load(Ordering::Relaxed) == NULL_ID {
            mid = 1; // reverse-sequential optimization
        } else if p == width && nl.next.load(Ordering::Relaxed) == NULL_ID {
            mid = width; // sequential optimization
        }

        // The reference now walks mid outward so equal ikeys stay in one
        // leaf; with fixed 8-byte keys every ikey in a leaf is distinct
        // (duplicates update in place), so that loop is dead and dropped.

        // Move post-insertion positions [mid, width] into nr slots [0, ..).
        // The reference computes value_from(width) when mid == width == 15,
        // a shift by 64 whose result the loop never consumes (only the new
        // key moves); Rust checks shifts, so give the dead case a 0.
        let from = mid - (p < mid) as i32;
        let mut pv = if from < WIDTH { perml.value_from(from) } else { 0 };
        for x in mid..=width {
            if x == p {
                nr.assign(x - mid, ika, value);
            } else {
                let slot = (pv & 15) as i32;
                nr.assign(x - mid, nl.ikey(slot), nl.lv[slot as usize].load(Ordering::Relaxed));
                pv >>= 4;
            }
        }
        let mut permr = Permuter::from_value(Permuter::make_sorted(width + 1 - mid));
        if p >= mid {
            // Park the new key's slot at the back: not reader-visible
            // until finish_insert, after the value is in place.
            permr.remove_to_back(p - mid);
        }
        nr.permutation.store(permr.value(), Ordering::Relaxed);
        let split_ikey = nr.ikey(0);

        // btree_leaflink::link_split. The reference's marked-pointer CAS
        // loop only defends against concurrent unlink; remove is deferred,
        // and nl's lock already excludes concurrent splits of nl, so plain
        // stores suffice. The Release on nl.next is nr's publication.
        nr.prev.store(nl_id, Ordering::Relaxed);
        let nxt = nl.next.load(Ordering::Relaxed);
        nr.next.store(nxt, Ordering::Relaxed);
        if nxt != NULL_ID {
            self.node(nxt).prev.store(nr_id, Ordering::Release);
        }
        nl.next.store(nr_id, Ordering::Release);

        let split_type = if p < mid { 0 } else { 1 + (mid == width) as i32 };
        (split_ikey, split_type)
    }

    /// internode::split_into: split the locked-full parent p into the fresh
    /// locked internode nr while inserting (ika, value_child) at kp.
    /// Returns the new kp if the pending child still lands in p, -1 if it
    /// went into nr or became the separator.
    // Mirrors the reference internode::split_into signature; the split
    // point, the pending (ika, child), and the returned separator all have
    // to cross this boundary together.
    #[allow(clippy::too_many_arguments)]
    fn internode_split_into(
        &self,
        p_id: NodeId,
        nr_id: NodeId,
        kp: i32,
        ika: u64,
        value_child: NodeId,
        split_ikey: &mut u64,
        split_type: i32,
    ) -> i32 {
        let p = self.node(p_id);
        let nr = self.node(nr_id);
        let mid = if split_type == 2 { IWIDTH } else { (IWIDTH + 1) / 2 };
        let nr_keys = IWIDTH + 1 - (mid + 1);
        nr.nkeys.store(nr_keys as u32, Ordering::Relaxed);
        nr.height
            .store(p.height.load(Ordering::Relaxed), Ordering::Relaxed);

        let shift_from = |dst_base: i32, src_base: i32, count: i32| {
            for i in 0..count {
                nr.ikey0[(dst_base + i) as usize]
                    .store(p.ikey(src_base + i), Ordering::Relaxed);
                nr.child[(dst_base + i + 1) as usize].store(
                    p.child[(src_base + i + 1) as usize].load(Ordering::Relaxed),
                    Ordering::Release,
                );
            }
        };
        if kp < mid {
            nr.child[0].store(p.child[mid as usize].load(Ordering::Relaxed), Ordering::Release);
            shift_from(0, mid, IWIDTH - mid);
            *split_ikey = p.ikey(mid - 1);
        } else if kp == mid {
            nr.child[0].store(value_child, Ordering::Release);
            shift_from(0, mid, IWIDTH - mid);
            *split_ikey = ika;
        } else {
            nr.child[0]
                .store(p.child[(mid + 1) as usize].load(Ordering::Relaxed), Ordering::Release);
            shift_from(0, mid + 1, kp - (mid + 1));
            // internode::assign inlined: set the pending child + its key.
            self.node(value_child).parent.store(nr_id, Ordering::Release);
            nr.child[(kp - mid) as usize].store(value_child, Ordering::Release);
            nr.ikey0[(kp - (mid + 1)) as usize].store(ika, Ordering::Relaxed);
            shift_from(kp + 1 - (mid + 1), kp, IWIDTH - kp);
            *split_ikey = p.ikey(mid);
        }

        // Reparent everything nr took. nr is locked, so a concurrent
        // locked_parent() that chases one of these fresh links spins until
        // the cascade publishes nr and unlocks it.
        for i in 0..=nr_keys {
            let c = nr.child[i as usize].load(Ordering::Relaxed);
            self.node(c).parent.store(nr_id, Ordering::Release);
        }

        // Readers of p spin or retry from here until p unlocks.
        p.version.mark_split();
        if kp < mid {
            p.nkeys.store((mid - 1) as u32, Ordering::Relaxed);
            kp
        } else {
            p.nkeys.store(mid as u32, Ordering::Relaxed);
            -1
        }
    }

    /// node_base::locked_parent: chase the parent link until we lock a node
    /// that still is the parent. NULL means n is (currently) a root.
    fn locked_parent(&self, n: NodeId) -> NodeId {
        debug_assert!(self.node(n).version.locked());
        loop {
            let p = self.node(n).parent.load(Ordering::Acquire);
            if p == NULL_ID {
                return NULL_ID;
            }
            self.node(p).version.lock();
            if p == self.node(n).parent.load(Ordering::Acquire) {
                debug_assert!(!self.node(p).version.is_leaf());
                return p;
            }
            self.node(p).version.unlock();
            core::hint::spin_loop();
        }
    }

    /// tcursor::make_split: called with the leaf n locked and the key not
    /// insertable in place. First retries the free-slot rearrangement, then
    /// runs the full split + hand-over-hand parent cascade. Owns every
    /// unlock on every path.
    fn make_split(
        &self,
        cpu: usize,
        n_orig: NodeId,
        kxi_in: i32,
        ika: u64,
        value: u64,
    ) -> Result<(), InsertError> {
        let nl = self.node(n_orig);
        // Maybe rearrange the permuter so back() is a usable slot: the swap
        // touches only free positions, invisible to readers.
        if nl.leaf_size() < WIDTH {
            let mut perm = Permuter::from_value(nl.permutation.load(Ordering::Relaxed));
            perm.exchange(perm.size(), WIDTH - 1);
            let kxp = perm.back();
            if kxp != 0 {
                nl.publish_perm(perm);
                nl.assign(kxp, ika, value);
                self.finish_insert(n_orig, kxi_in, kxp);
                nl.version.unlock();
                return Ok(());
            }
        }

        // Reserve the cascade's worst-case internodes up front: once an
        // internode splits there is no consistent way to stop, so all
        // allocation failures must surface before the first mutation.
        let mut reserve = match self.reserve_internodes(cpu) {
            Some(r) => r,
            None => {
                nl.version.unlock();
                return Err(InsertError::PoolExhausted);
            }
        };
        let child0 = match self.alloc_leaf(cpu) {
            Some(id) => id,
            None => {
                self.return_reserve(cpu, reserve);
                nl.version.unlock();
                return Err(InsertError::PoolExhausted);
            }
        };
        // assign_version(*n_) stand-in: NodeVersion has no raw copy, so the
        // fresh child is simply locked. Only version DELTAS matter to
        // validators (readers first meet the child after publication), and
        // the lock bit is the part that is load-bearing: writers arriving
        // over the B-link spin until the cascade completes. The root bit is
        // deliberately not inherited (the reference's leaf child can carry
        // a stale root bit; benign there, absent here).
        self.node(child0).version.lock();

        let (xikey0, split_type) = self.leaf_split_into(n_orig, child0, kxi_in, ika, value);
        let mut xikey: [u64; 2] = [xikey0, 0];
        let mut sense = 0usize;
        let mut n = n_orig;
        let mut child = child0;
        let mut height = 0u32;
        // n_ may be reassigned to the right leaf; the new key's final
        // logical position rides along.
        let mut n_cursor = n_orig;
        let mut kxi = kxi_in;
        let mut kxp = -1i32; // always set by the height-0 iteration below

        loop {
            let mut next_child = NULL_ID;
            let p = self.locked_parent(n);

            let mut kp = -1i32;
            if p != NULL_ID {
                kp = self.inter_upper(self.node(p), xikey[sense]);
                self.node(p).version.mark_insert();
            }

            if kp < 0 || self.node(p).height.load(Ordering::Relaxed) > height + 1 {
                // No parent (grow a new root) or the parent is more than
                // one level up (a racing split left a height gap): bridge
                // with a fresh internode over just {n, child}.
                let nn_id = reserve.take();
                if nn_id == NULL_ID {
                    // Statically unreachable (reserve covers the deepest
                    // cascade); bail keeping every lock consistent.
                    if p != NULL_ID {
                        self.node(p).version.unlock();
                    }
                    if n != n_cursor {
                        self.node(n).version.unlock();
                    }
                    if child != n_cursor {
                        self.node(child).version.unlock();
                    }
                    self.node(n_cursor).version.unlock();
                    self.return_reserve(cpu, reserve);
                    return Err(InsertError::PoolExhausted);
                }
                let nn = self.node(nn_id);
                nn.height.store(height + 1, Ordering::Relaxed);
                nn.nkeys.store(1, Ordering::Relaxed);
                nn.ikey0[0].store(xikey[sense], Ordering::Relaxed);
                nn.child[0].store(n, Ordering::Release);
                nn.child[1].store(child, Ordering::Release);
                self.node(child).parent.store(nn_id, Ordering::Release);
                if kp < 0 {
                    // make_layer_root: parent already NULL on a fresh node.
                    nn.version.mark_root();
                } else {
                    nn.parent.store(p, Ordering::Release);
                    self.node(p).child[kp as usize].store(nn_id, Ordering::Release);
                }
                // Publication for the no-parent case: stale-root walkers
                // find the new root through this link. The reference's
                // fence() before set_parent is this Release.
                self.node(n).parent.store(nn_id, Ordering::Release);
            } else {
                if self.node(p).isize() >= IWIDTH {
                    next_child = reserve.take();
                    if next_child == NULL_ID {
                        if n != n_cursor {
                            self.node(n).version.unlock();
                        }
                        if child != n_cursor {
                            self.node(child).version.unlock();
                        }
                        self.node(n_cursor).version.unlock();
                        self.node(p).version.unlock();
                        self.return_reserve(cpu, reserve);
                        return Err(InsertError::PoolExhausted);
                    }
                    // assign_version(*p) + mark_nonroot stand-in: fresh
                    // internode, locked; see the leaf child note above.
                    self.node(next_child).version.lock();
                    let xi = xikey[sense];
                    kp = self.internode_split_into(
                        p,
                        next_child,
                        kp,
                        xi,
                        child,
                        &mut xikey[sense ^ 1],
                        split_type,
                    );
                }
                if kp >= 0 {
                    // shift_up + assign under p's inserting mark: readers
                    // that could observe the torn window are spinning in
                    // stable() or will fail has_changed().
                    let pn = self.node(p);
                    let size = pn.isize();
                    let mut i = size;
                    while i > kp {
                        pn.ikey0[i as usize].store(pn.ikey(i - 1), Ordering::Relaxed);
                        pn.child[(i + 1) as usize]
                            .store(pn.child[i as usize].load(Ordering::Relaxed), Ordering::Relaxed);
                        i -= 1;
                    }
                    self.node(child).parent.store(p, Ordering::Release);
                    pn.child[(kp + 1) as usize].store(child, Ordering::Release);
                    pn.ikey0[kp as usize].store(xikey[sense], Ordering::Relaxed);
                    // The reference fence()s before ++nkeys_; Release keeps
                    // the new slot's stores ahead of the size that admits it.
                    pn.nkeys.store((size + 1) as u32, Ordering::Release);
                }
            }

            if height == 0 {
                // Complete the leaf split, delayed until both halves are
                // reachable: strip the moved keys from nl and place the new
                // key on whichever side owns it now.
                let nl = self.node(n_orig);
                let mut perml = Permuter::from_value(nl.permutation.load(Ordering::Relaxed));
                let width = perml.size();
                perml.set_size(width - self.node(child0).leaf_size());
                if width != WIDTH {
                    // The removed (blocked slot-0) item must sit at
                    // perml.size(): swap it into place.
                    perml.exchange(perml.size(), WIDTH - 1);
                }
                nl.version.mark_split();
                nl.publish_perm(perml);
                if split_type == 0 {
                    kxp = perml.back();
                    nl.assign(kxp, ika, value);
                } else {
                    kxi -= perml.size();
                    kxp = kxi;
                    n_cursor = child0;
                }
            }

            // Hand-over-hand: everything but the cursor leaf unlocks here;
            // the cursor leaf stays locked for finish_insert below.
            if n != n_cursor {
                self.node(n).version.unlock();
            }
            if child != n_cursor {
                self.node(child).version.unlock();
            }
            if next_child != NULL_ID {
                n = p;
                child = next_child;
                sense ^= 1;
                height += 1;
            } else {
                if p != NULL_ID {
                    self.node(p).version.unlock();
                }
                break;
            }
        }

        // tcursor::finish: expose the new key and release the cursor leaf.
        self.finish_insert(n_cursor, kxi, kxp);
        self.node(n_cursor).version.unlock();
        self.return_reserve(cpu, reserve);
        Ok(())
    }

    // TODO(next round): remove(), with modstate_, leaflink unlink and EBR
    // reclamation of retired nodes.
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::boxed::Box;
    use std::sync::atomic::{AtomicBool, AtomicU64 as StdAtomicU64, Ordering as O};
    use std::thread;
    use std::vec::Vec;

    fn k(x: u64) -> [u8; 8] {
        x.to_be_bytes()
    }

    /// Fresh tree with a 'static lifetime so scoped threads can borrow it.
    /// (Leaked on purpose; each test owns its own.)
    fn tree() -> &'static Masstree {
        Box::leak(Box::new(Masstree::new()))
    }

    #[test]
    fn const_new_supports_a_static() {
        // The whole point of const fn new(): a bare-metal static.
        static T: Masstree = Masstree::new();
        assert_eq!(T.get(k(1)), None);
        assert_eq!(T.put(k(1), 10), Ok(Inserted::New));
        assert_eq!(T.get(k(1)), Some(10));
    }

    #[test]
    fn empty_tree_returns_none() {
        let t = tree();
        assert_eq!(t.get(k(0)), None);
        assert_eq!(t.get(k(u64::MAX)), None);
    }

    #[test]
    fn insert_lookup_and_miss_within_one_leaf() {
        let t = tree();
        for x in [5u64, 1, 9, 3, 7, 2, 8, 4, 6] {
            assert_eq!(t.put(k(x), x * 100), Ok(Inserted::New));
        }
        for x in 1..=9u64 {
            assert_eq!(t.get(k(x)), Some(x * 100));
        }
        assert_eq!(t.get(k(0)), None, "below min");
        assert_eq!(t.get(k(10)), None, "above max");
    }

    #[test]
    fn update_replaces_value_in_place() {
        let t = tree();
        assert_eq!(t.put(k(4), 400), Ok(Inserted::New));
        assert_eq!(t.put(k(4), 4000), Ok(Inserted::Updated));
        assert_eq!(t.get(k(4)), Some(4000));
        // Updates keep working after the leaf splits.
        for x in 0..40u64 {
            let _ = t.put(k(x), x + 1);
        }
        assert_eq!(t.put(k(4), 44), Ok(Inserted::Updated));
        assert_eq!(t.get(k(4)), Some(44));
    }

    #[test]
    fn leaf_split_keeps_every_key() {
        let t = tree();
        // One past the leaf width forces the first split.
        for x in 0..=WIDTH as u64 {
            assert_eq!(t.put(k(x), x + 1), Ok(Inserted::New));
        }
        for x in 0..=WIDTH as u64 {
            assert_eq!(t.get(k(x)), Some(x + 1), "key {x} lost across split");
        }
        assert_eq!(t.get(k(999)), None);
    }

    #[test]
    fn insert_below_ikey_bound_exercises_slot0_guard() {
        // Fill 1..=WIDTH so slot 0 holds key 1 (the ikey_bound), then
        // insert 0: the reverse-sequential split path (mid = 1) runs and
        // the slot-0 reuse guard is what keeps ikey_bound sane.
        let t = tree();
        for x in 1..=WIDTH as u64 {
            assert_eq!(t.put(k(x), x + 1), Ok(Inserted::New));
        }
        assert_eq!(t.put(k(0), 1), Ok(Inserted::New));
        for x in 0..=WIDTH as u64 {
            assert_eq!(t.get(k(x)), Some(x + 1), "key {x} lost");
        }
    }

    #[test]
    fn descending_inserts_take_reverse_sequential_splits() {
        let t = tree();
        let n = 200u64;
        for x in (0..n).rev() {
            assert_eq!(t.put(k(x), x + 1), Ok(Inserted::New), "put {x}");
        }
        for x in 0..n {
            assert_eq!(t.get(k(x)), Some(x + 1), "key {x} lost");
        }
    }

    #[test]
    fn ascending_inserts_take_sequential_splits() {
        let t = tree();
        let n = 400u64;
        for x in 0..n {
            assert_eq!(t.put(k(x), x + 1), Ok(Inserted::New), "put {x}");
        }
        for x in 0..n {
            assert_eq!(t.get(k(x)), Some(x + 1), "key {x} lost");
        }
        assert_eq!(t.get(k(n)), None);
    }

    #[test]
    fn scrambled_inserts_build_multiple_internode_levels() {
        let t = tree();
        // 400 keys across ~27+ leaves guarantees at least two internode
        // levels; the full-period LCG scrambles the order so both halves
        // of every leaf and internode split get exercised.
        let n = 400u64;
        for i in 0..n {
            let x = (i.wrapping_mul(97).wrapping_add(41)) % n;
            assert_eq!(t.put(k(x), x + 1), Ok(Inserted::New), "put {x}");
        }
        for x in 0..n {
            assert_eq!(t.get(k(x)), Some(x + 1), "key {x} lost or wrong");
        }
        assert_eq!(t.get(k(n)), None);
        assert_eq!(t.get(k(u64::MAX)), None);
        // Re-put every key as an update; nothing may move or vanish.
        for x in 0..n {
            assert_eq!(t.put(k(x), x + 2), Ok(Inserted::Updated), "re-put {x}");
        }
        for x in 0..n {
            assert_eq!(t.get(k(x)), Some(x + 2));
        }
    }

    #[test]
    fn sparse_and_extreme_keys() {
        let t = tree();
        let keys = [0u64, 1, u64::MAX, u64::MAX - 1, 1 << 32, 1 << 63, 12345];
        for (i, &x) in keys.iter().enumerate() {
            assert_eq!(t.put(k(x), i as u64), Ok(Inserted::New));
        }
        for (i, &x) in keys.iter().enumerate() {
            assert_eq!(t.get(k(x)), Some(i as u64));
        }
        assert_eq!(t.get(k(2)), None);
    }

    #[test]
    fn pool_exhaustion_reports_error_and_keeps_tree_consistent() {
        let t = tree();
        let mut inserted = 0u64;
        loop {
            match t.put(k(inserted), inserted + 1) {
                Ok(Inserted::New) => inserted += 1,
                Err(InsertError::PoolExhausted) => break,
                other => panic!("unexpected {other:?} at {inserted}"),
            }
            assert!(inserted < 500_000, "pool never exhausted");
        }
        assert!(inserted > 1_000, "exhausted implausibly early ({inserted})");
        // Everything inserted before the failure stays readable, and the
        // tree still serves updates.
        for x in (0..inserted).step_by(7) {
            assert_eq!(t.get(k(x)), Some(x + 1), "key {x} lost at exhaustion");
        }
        assert_eq!(t.put(k(0), 42), Ok(Inserted::Updated));
        assert_eq!(t.get(k(0)), Some(42));
    }

    // ---- concurrent hazards (modeled on the cbptree suite) --------------

    #[test]
    fn reader_sees_consistent_values_across_splits() {
        // A reader hammers get() while one writer splits leaves; any value
        // observed must match its key (stale-route + reader-during-split).
        let t = tree();
        let done = AtomicBool::new(false);
        thread::scope(|s| {
            s.spawn(|| {
                while !done.load(O::Acquire) {
                    for x in 0..(4 * WIDTH as u64) {
                        if let Some(v) = t.get(k(x)) {
                            assert_eq!(v, x + 1, "key {x} read mismatched value {v}");
                        }
                    }
                }
            });
            for x in 0..(4 * WIDTH as u64) {
                assert_eq!(t.put(k(x), x + 1), Ok(Inserted::New));
            }
            done.store(true, O::Release);
        });
    }

    #[test]
    fn committed_keys_never_vanish_during_leaf_splits() {
        // The stale-route hazard: a reader routes to a leaf, the writer
        // splits it moving the key right; the B-link advance must carry the
        // reader over. Every committed key stays continuously visible.
        let t = tree();
        let committed = StdAtomicU64::new(0);
        let done = AtomicBool::new(false);
        thread::scope(|s| {
            for _ in 0..2 {
                s.spawn(|| {
                    while !done.load(O::Acquire) {
                        let hi = committed.load(O::Acquire);
                        for x in 0..hi {
                            assert_eq!(t.get(k(x)), Some(x + 1), "committed key {x} vanished");
                        }
                    }
                });
            }
            for x in 0..300u64 {
                t.put(k(x), x + 1).expect("pool is large enough");
                committed.store(x + 1, O::Release);
            }
            done.store(true, O::Release);
        });
        for x in 0..300u64 {
            assert_eq!(t.get(k(x)), Some(x + 1));
        }
    }

    #[test]
    fn committed_keys_survive_cascading_internode_splits() {
        // Push far past one internode's fan-out so internode splits (and a
        // root split growing the tree) run under concurrent readers.
        let t = tree();
        let committed = StdAtomicU64::new(0);
        let done = AtomicBool::new(false);
        thread::scope(|s| {
            for _ in 0..2 {
                s.spawn(|| {
                    while !done.load(O::Acquire) {
                        let hi = committed.load(O::Acquire);
                        // Sweep from the top so reads race the freshest
                        // splits first.
                        for x in (0..hi).rev() {
                            assert_eq!(t.get(k(x)), Some(x + 1), "committed key {x} vanished");
                        }
                    }
                });
            }
            let n = 1200u64; // several internode levels
            for x in 0..n {
                t.put(k(x), x + 1).expect("pool is large enough");
                committed.store(x + 1, O::Release);
            }
            done.store(true, O::Release);
        });
        for x in 0..1200u64 {
            assert_eq!(t.get(k(x)), Some(x + 1));
        }
    }

    #[test]
    fn concurrent_writers_on_one_leaf_all_land() {
        // WIDTH disjoint keys from several writers race into the same
        // (initially single) leaf: the per-leaf lock serializes them.
        let t = tree();
        thread::scope(|s| {
            for w in 0..5u64 {
                s.spawn(move || {
                    for j in 0..3u64 {
                        let key = w * 3 + j;
                        assert_eq!(t.put(k(key), key + 1), Ok(Inserted::New));
                    }
                });
            }
        });
        for key in 0..15u64 {
            assert_eq!(t.get(k(key)), Some(key + 1), "key {key} missing");
        }
    }

    #[test]
    fn concurrent_writers_interleaved_ranges_with_readers() {
        // 4 writers insert interleaved key ranges (maximal lock-coupling
        // contention: they keep landing in the same leaves and racing the
        // same splits) while readers verify their own committed prefixes.
        const WRITERS: u64 = 4;
        const PER: u64 = 250;
        let t = tree();
        let committed: Vec<StdAtomicU64> =
            (0..WRITERS).map(|_| StdAtomicU64::new(0)).collect();
        let committed = &committed;
        let done = AtomicBool::new(false);
        thread::scope(|s| {
            for w in 0..WRITERS {
                s.spawn(move || {
                    for j in 0..PER {
                        let key = j * WRITERS + w;
                        assert_eq!(t.put(k(key), key + 1), Ok(Inserted::New));
                        committed[w as usize].store(j + 1, O::Release);
                    }
                });
            }
            for _ in 0..2 {
                s.spawn(|| {
                    while !done.load(O::Acquire) {
                        for w in 0..WRITERS {
                            let hi = committed[w as usize].load(O::Acquire);
                            for j in 0..hi {
                                let key = j * WRITERS + w;
                                assert_eq!(
                                    t.get(k(key)),
                                    Some(key + 1),
                                    "writer {w} key {key} vanished"
                                );
                            }
                        }
                    }
                });
            }
            // Writers finish, then release the readers.
            // (scope join happens at block end; flag first so readers exit
            // after one more full verification pass.)
            for w in 0..WRITERS {
                while committed[w as usize].load(O::Acquire) < PER {
                    thread::yield_now();
                }
            }
            done.store(true, O::Release);
        });
        for key in 0..WRITERS * PER {
            assert_eq!(t.get(k(key)), Some(key + 1), "key {key} missing");
        }
    }

    #[test]
    fn concurrent_updates_never_show_torn_or_foreign_values() {
        // Writers rewrite existing keys with generation-tagged values while
        // readers check every observed value is one the key legitimately
        // held (value = key * 1000 + generation).
        let t = tree();
        let n = 64u64;
        for x in 0..n {
            t.put(k(x), x * 1000).unwrap();
        }
        let done = AtomicBool::new(false);
        thread::scope(|s| {
            s.spawn(|| {
                for generation in 1..=200u64 {
                    for x in 0..n {
                        assert_eq!(t.put(k(x), x * 1000 + generation), Ok(Inserted::Updated));
                    }
                }
                done.store(true, O::Release);
            });
            for _ in 0..2 {
                s.spawn(|| {
                    while !done.load(O::Acquire) {
                        for x in 0..n {
                            let v = t.get(k(x)).expect("existing key vanished");
                            assert_eq!(v / 1000, x, "key {x} read foreign value {v}");
                            assert!(v % 1000 <= 200, "key {x} read torn value {v}");
                        }
                    }
                });
            }
        });
    }

    #[test]
    fn per_cpu_pools_concurrent_disjoint_inserts_all_land() {
        // Exercises the per-cpu allocation fast path (put_on with a UNIQUE cpu
        // per thread) directly: each worker allocates from its own pool with
        // no global lock, splitting its own subtrees. Every key must survive,
        // and concurrent readers must never lose a committed key -- the same
        // invariants as the SHARED path, but over the lock-free allocator.
        let t = tree();
        let threads = 6usize;
        let per = 120u64;
        let committed: &'static [StdAtomicU64] = Box::leak(
            (0..threads)
                .map(|_| StdAtomicU64::new(0))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        thread::scope(|s| {
            for c in 0..threads {
                s.spawn(move || {
                    let base = c as u64 * per;
                    for i in 0..per {
                        let key = base + i;
                        assert_eq!(t.put_on(c, k(key), key + 1), Ok(Inserted::New));
                        committed[c].store(i + 1, O::Release);
                    }
                });
            }
            // A reader hammering every committed key of every worker: none may
            // vanish or read a foreign value while the per-cpu writers split.
            for _ in 0..2 {
                s.spawn(|| {
                    let mut done = false;
                    while !done {
                        done = true;
                        for (c, prog) in committed.iter().enumerate() {
                            let hi = prog.load(O::Acquire);
                            if hi < per {
                                done = false;
                            }
                            let base = c as u64 * per;
                            for i in 0..hi {
                                let key = base + i;
                                assert_eq!(t.get(k(key)), Some(key + 1), "committed key {key} vanished");
                            }
                        }
                    }
                });
            }
        });
        for c in 0..threads {
            for i in 0..per {
                let key = c as u64 * per + i;
                assert_eq!(t.get(k(key)), Some(key + 1), "key {key} lost");
            }
        }
    }

    // ---- concurrency scaling microbench (run on demand) ------------------
    //
    // Measures how insert throughput scales with cores for this Masstree
    // (fine-grained per-node locking) against the sequential-writer-locked
    // cbptree. Both resolve reads lock-free; the difference is the write
    // path, so the workload is concurrent insertion of DISJOINT new keys --
    // exactly the case a single tree-wide writer lock serializes and
    // per-node locking does not.
    //
    // Run with: cargo test -p bmdb-core --release masstree::tests::scaling
    //           -- --ignored --nocapture
    #[test]
    #[ignore = "microbench; run explicitly with --ignored --nocapture"]
    fn scaling_insert_throughput_masstree_vs_cbptree() {
        use crate::cbptree::Tree as CbTree;
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as SO};
        use std::sync::Arc;
        use std::time::Instant;

        // Fixed work PER THREAD: each worker inserts its own disjoint block
        // of PER keys, so total work grows with the thread count and the
        // timed wall clock stays flat only if the writers actually run in
        // parallel. PER * max_threads stays inside cbptree's 256-node pool.
        const PER: u64 = 200;
        const TRIALS: usize = 25;
        let thread_counts = [1usize, 2, 4, 8];

        // Pin worker `cpu` to a distinct physical core of socket 0 (logical
        // CPUs 0..16 on this box, one per physical core, same NUMA node). This
        // dev box is a busy 2-socket NUMA machine; without pinning the
        // scheduler migrates workers across sockets and the measurement
        // reports migration/coherence noise rather than the tree's scaling.
        fn pin_to_core(cpu: usize) {
            #[cfg(target_os = "linux")]
            unsafe {
                let mut set: libc::cpu_set_t = core::mem::zeroed();
                libc::CPU_SET(cpu, &mut set);
                libc::sched_setaffinity(0, core::mem::size_of::<libc::cpu_set_t>(), &set);
            }
        }

        fn cb_ins(t: &CbTree, _cpu: usize, key: [u8; 8], v: u64) {
            let _ = t.insert(key, v);
        }
        fn mt_ins(t: &Masstree, cpu: usize, key: [u8; 8], v: u64) {
            // Each worker owns a unique cpu index, so it allocates from its
            // private per-cpu pool -- no global alloc lock, the whole point.
            let _ = t.put_on(cpu, key, v);
        }

        // Best (min) wall time over TRIALS for `threads` workers, each
        // inserting PER disjoint keys, released together by a spin barrier so
        // thread-spawn cost is outside the timed region.
        fn best<T: Send + Sync + 'static>(
            threads: usize,
            make: fn() -> T,
            ins: fn(&T, usize, [u8; 8], u64),
        ) -> core::time::Duration {
            let mut best = core::time::Duration::MAX;
            for _ in 0..TRIALS {
                let tree = Arc::new(make());
                let ready = Arc::new(AtomicUsize::new(0));
                let go = Arc::new(AtomicBool::new(false));
                let mut handles = std::vec::Vec::new();
                for t in 0..threads {
                    let (tr, rd, gg) = (tree.clone(), ready.clone(), go.clone());
                    handles.push(std::thread::spawn(move || {
                        pin_to_core(t);
                        rd.fetch_add(1, SO::Release);
                        while !gg.load(SO::Acquire) {
                            core::hint::spin_loop();
                        }
                        let base = t as u64 * PER;
                        for i in 0..PER {
                            ins(&tr, t, (base + i).to_be_bytes(), base + i + 1);
                        }
                    }));
                }
                while ready.load(SO::Acquire) < threads {
                    core::hint::spin_loop();
                }
                let start = Instant::now();
                go.store(true, SO::Release);
                for h in handles {
                    h.join().unwrap();
                }
                let dt = start.elapsed();
                if dt < best {
                    best = dt;
                }
            }
            best
        }

        std::eprintln!(
            "\nconcurrent insert, {} disjoint keys PER thread (best of {} trials):",
            PER, TRIALS
        );
        std::eprintln!(
            "{:>7} | {:>16} {:>7} | {:>16} {:>7}",
            "threads", "cbptree keys/s", "scaling", "masstree keys/s", "scaling"
        );
        let mut cb_base = 0.0f64;
        let mut mt_base = 0.0f64;
        for &n in &thread_counts {
            let total = PER * n as u64;
            let cb = best(n, CbTree::new, cb_ins);
            let mt = best(n, Masstree::new, mt_ins);
            let cb_rate = total as f64 / cb.as_secs_f64();
            let mt_rate = total as f64 / mt.as_secs_f64();
            if n == 1 {
                cb_base = cb_rate;
                mt_base = mt_rate;
            }
            std::eprintln!(
                "{:>7} | {:>16.0} {:>6.2}x | {:>16.0} {:>6.2}x",
                n,
                cb_rate,
                cb_rate / cb_base,
                mt_rate,
                mt_rate / mt_base
            );
        }
        std::eprintln!(
            "(cbptree serializes writers on one lock; masstree locks per node)\n"
        );
    }

    /// YCSB A/B/C scaling on a warm, steady-state tree -- the honest
    /// measurement the top-down study prescribes (.note/topdown-scaling-
    /// verdict.md). Unlike the insert microbench (which times the slowest
    /// wakeup building a tiny tree from empty), this LOADS a tree once, then
    /// times a fixed DEADLINE WINDOW during which every pinned worker runs
    /// get/update ops (no allocation -- run phase never inserts), so throughput
    /// is Σ ops / identical window and stragglers can't set the number.
    ///
    ///   YCSB-C: 100% read   YCSB-B: 95% read / 5% update   YCSB-A: 50/50
    ///
    /// Run: cargo test -p bmdb-core --release masstree::tests::ycsb
    ///      -- --ignored --nocapture
    #[test]
    #[ignore = "microbench; run explicitly with --ignored --nocapture"]
    fn ycsb_scaling_on_a_warm_tree() {
        use std::sync::atomic::{AtomicBool, AtomicU64 as A64, AtomicUsize, Ordering as SO};
        use std::time::{Duration, Instant};

        // Steady-state tree size: as large as the leaf pool holds so upper
        // splits are rare and the tree is >~3 levels. Read-only run phase, so
        // no allocation happens after load.
        const KEYS: u64 = 6000;
        const WINDOW: Duration = Duration::from_millis(150);
        let thread_counts = [1usize, 2, 4, 6];

        fn pin(cpu: usize) {
            #[cfg(target_os = "linux")]
            unsafe {
                let mut set: libc::cpu_set_t = core::mem::zeroed();
                libc::CPU_SET(cpu, &mut set);
                libc::sched_setaffinity(0, core::mem::size_of::<libc::cpu_set_t>(), &set);
            }
        }

        let tree = tree();
        for key in 0..KEYS {
            tree.put(k(key), key + 1).expect("load must fit the pool");
        }
        // Confirm the warm tree is correct before measuring.
        for key in [0u64, KEYS / 2, KEYS - 1] {
            assert_eq!(tree.get(k(key)), Some(key + 1));
        }

        std::eprintln!(
            "\nYCSB on a warm {}-key tree, {}ms window, pinned to socket-0 cores:",
            KEYS,
            WINDOW.as_millis()
        );
        std::eprintln!(
            "{:>9} | {:>7} | {:>14} {:>8} {:>12}",
            "workload", "threads", "ops/s", "scaling", "ops/s/core"
        );

        for &(name, read_pct) in &[("C(100r)", 100u64), ("B(95/5)", 95), ("A(50/50)", 50)] {
            let mut base = 0.0f64;
            for &n in &thread_counts {
                let ready = std::sync::Arc::new(AtomicUsize::new(0));
                let go = std::sync::Arc::new(AtomicBool::new(false));
                let total = std::sync::Arc::new(A64::new(0));
                std::thread::scope(|s| {
                    for t in 0..n {
                        let (rd, gg, tot) = (ready.clone(), go.clone(), total.clone());
                        s.spawn(move || {
                            pin(t);
                            // Distinct nonzero xorshift seed per worker.
                            let mut rng = 0x9E37_79B9_7F4A_7C15u64 ^ (t as u64 + 1);
                            let mut xorshift = || {
                                rng ^= rng << 13;
                                rng ^= rng >> 7;
                                rng ^= rng << 17;
                                rng
                            };
                            rd.fetch_add(1, SO::Release);
                            while !gg.load(SO::Acquire) {
                                core::hint::spin_loop();
                            }
                            let deadline = Instant::now() + WINDOW;
                            let mut ops = 0u64;
                            // Check the clock every 256 ops to keep it off the
                            // hot path.
                            loop {
                                for _ in 0..256 {
                                    let key = xorshift() % KEYS;
                                    if xorshift() % 100 < read_pct {
                                        let _ = tree.get(k(key));
                                    } else {
                                        let _ = tree.put_on(t, k(key), key + 1);
                                    }
                                    ops += 1;
                                }
                                if Instant::now() >= deadline {
                                    break;
                                }
                            }
                            tot.fetch_add(ops, SO::Relaxed);
                        });
                    }
                    while ready.load(SO::Acquire) < n {
                        core::hint::spin_loop();
                    }
                    go.store(true, SO::Release);
                });
                let ops = total.load(SO::Relaxed);
                let rate = ops as f64 / WINDOW.as_secs_f64();
                if n == 1 {
                    base = rate;
                }
                std::eprintln!(
                    "{:>9} | {:>7} | {:>14.0} {:>7.2}x {:>12.0}",
                    name,
                    n,
                    rate,
                    rate / base,
                    rate / n as f64
                );
            }
        }
        std::eprintln!(
            "(flat ops/s/core == linear scaling; steady-state read+update, no alloc)\n"
        );
    }
}
