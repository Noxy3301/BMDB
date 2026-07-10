// Node version word for Masstree-style optimistic concurrency control.
//
// Faithful port of nodeversion<nodeversion_parameters<uint64_t>> from
// masstree-beta nodeversion.hh (c) 2012-2013 President and Fellows of
// Harvard College and Massachusetts Institute of Technology. Semantics are
// meant to match the reference exactly so its correctness carries over.
//
// One 64-bit word packs a spinlock, two "dirty" flags and two update
// counters so readers can validate lock-free traversals seqlock-style:
//
//   bit  63      isleaf     (immutable after construction)
//   bit  62      root
//   bit  61      deleted
//   bit  60      unused
//   bits 27..60  split counter
//   bits 11..27  insert counter
//   bit  10      splitting
//   bit   9      inserting
//   bit   8      lock
//   bits  0..8   unused
//
// The C++ original keeps v_ as a PLAIN integer and orders accesses with
// compiler-only fences (acquire_fence/release_fence/fence are all just
// `asm volatile("" ::: "memory")`), leaning on x86-TSO for the hardware
// ordering. Rust has no such escape hatch: a word read and written
// concurrently must be an atomic or the program has undefined behavior. So
// v_ becomes an AtomicU64, every plain access becomes a Relaxed access, and
// each fence maps to the C++11-model fence that forbids the same reordering
// the reference relied on -- no more, no less. See the "ordering:" comments.
//
// The two subtle mappings (both verified by porting this file twice, with
// two independent models, and reconciling): the reference's misleadingly
// named `acquire_fence()` after a mark sets a dirty bit needs a StoreStore
// edge (the mark must be visible no later than the content stores that
// follow), which is `fence(Release)` -- an acquire fence orders nothing
// store-to-store. And the `fence()` before a validation re-read needs only
// a LoadLoad edge, which is `fence(Acquire)`; SeqCst would add a StoreLoad
// barrier (an mfence) the reference never issued.
//
// Caller obligation: these fences only order node CONTENTS if those contents
// are themselves accessed with atomic operations. Plain non-atomic contents
// shared with concurrent readers are a data race in Rust regardless of this
// word's correctness.

use core::sync::atomic::{fence, AtomicU64, Ordering};

pub const LOCK_BIT: u64 = 1 << 8;
pub const INSERTING_SHIFT: u32 = 9;
pub const INSERTING_BIT: u64 = 1 << 9;
pub const SPLITTING_BIT: u64 = 1 << 10;
pub const DIRTY_MASK: u64 = INSERTING_BIT | SPLITTING_BIT;
/* == INSERTING_BIT << 2; insert counter occupies [11..27) */
pub const VINSERT_LOWBIT: u64 = 1 << 11;
/* split counter occupies [27..60) */
pub const VSPLIT_LOWBIT: u64 = 1 << 27;
pub const UNUSED1_BIT: u64 = 1 << 60;
pub const DELETED_BIT: u64 = 1 << 61;
pub const ROOT_BIT: u64 = 1 << 62;
pub const ISLEAF_BIT: u64 = 1 << 63;
pub const SPLIT_UNLOCK_MASK: u64 = !(ROOT_BIT | UNUSED1_BIT | (VSPLIT_LOWBIT - 1));
pub const UNLOCK_MASK: u64 = !(UNUSED1_BIT | (VINSERT_LOWBIT - 1));

/// Immutable snapshot of a version word, as returned by `stable()` and
/// `lock()`. Mirrors the C++ idiom of returning `nodeversion` by value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Version(u64);

impl Version {
    pub fn value(self) -> u64 {
        self.0
    }
    pub fn unlocked_value(self) -> u64 {
        self.0 & UNLOCK_MASK
    }
    pub fn is_leaf(self) -> bool {
        self.0 & ISLEAF_BIT != 0
    }
    pub fn locked(self) -> bool {
        self.0 & LOCK_BIT != 0
    }
    pub fn inserting(self) -> bool {
        self.0 & INSERTING_BIT != 0
    }
    pub fn splitting(self) -> bool {
        self.0 & SPLITTING_BIT != 0
    }
    pub fn deleted(self) -> bool {
        self.0 & DELETED_BIT != 0
    }
    pub fn is_root(self) -> bool {
        self.0 & ROOT_BIT != 0
    }
}

/// The live, shared version word. Locking is manual, as in the reference:
/// `lock()` returns the locked snapshot and the holder must call `unlock()`.
pub struct NodeVersion {
    v: AtomicU64,
}

impl NodeVersion {
    pub const fn new(is_leaf: bool) -> Self {
        NodeVersion {
            v: AtomicU64::new(if is_leaf { ISLEAF_BIT } else { 0 }),
        }
    }

    // Plain reads in the reference; nothing is ordered around them.
    fn snapshot(&self) -> Version {
        Version(self.v.load(Ordering::Relaxed))
    }

    pub fn is_leaf(&self) -> bool {
        self.snapshot().is_leaf()
    }
    pub fn locked(&self) -> bool {
        self.snapshot().locked()
    }
    pub fn inserting(&self) -> bool {
        self.snapshot().inserting()
    }
    pub fn splitting(&self) -> bool {
        self.snapshot().splitting()
    }
    pub fn deleted(&self) -> bool {
        self.snapshot().deleted()
    }
    pub fn is_root(&self) -> bool {
        self.snapshot().is_root()
    }

    /// Spin until the word is not dirty, then return it as a snapshot the
    /// caller may validate against later with `has_changed()`/`has_split()`.
    pub fn stable(&self) -> Version {
        let mut x = self.v.load(Ordering::Relaxed);
        while x & DIRTY_MASK != 0 {
            core::hint::spin_loop(); /* relax_fence(): pause */
            x = self.v.load(Ordering::Relaxed);
        }
        // ordering: acquire_fence() in the reference. Content loads that
        // follow must not be hoisted above the version load, or the reader
        // would see contents older than the snapshot claims and the later
        // has_changed() check could not tell. An acquire fence after the
        // relaxed load orders prior loads before everything after it, and
        // synchronizes with the release store in unlock().
        fence(Ordering::Acquire);
        Version(x)
    }

    /// True unless the word still matches the snapshot (the lock bit alone
    /// does not count as a change).
    pub fn has_changed(&self, x: Version) -> bool {
        // ordering: fence() in the reference is a compiler barrier whose
        // job under TSO is LoadLoad: the content loads being validated must
        // complete before the version word is re-read. An acquire fence is
        // exactly that edge; SeqCst would add a StoreLoad barrier (mfence)
        // the reference never issued.
        fence(Ordering::Acquire);
        (x.0 ^ self.v.load(Ordering::Relaxed)) > LOCK_BIT
    }

    /// True if the split counter advanced since the snapshot.
    pub fn has_split(&self, x: Version) -> bool {
        // ordering: same LoadLoad edge as has_changed().
        fence(Ordering::Acquire);
        (x.0 ^ self.v.load(Ordering::Relaxed)) >= VSPLIT_LOWBIT
    }

    /// `has_split()` without the fence, for callers that already ordered
    /// their loads.
    pub fn simple_has_split(&self, x: Version) -> bool {
        (x.0 ^ self.v.load(Ordering::Relaxed)) >= VSPLIT_LOWBIT
    }

    /// Spin until the lock bit is won; returns the locked snapshot.
    pub fn lock(&self) -> Version {
        let mut expected = self.v.load(Ordering::Relaxed);
        loop {
            // ordering: the reference wins a plain `lock cmpxchg` and then
            // issues acquire_fence(). Acquire on success folds that fence
            // into the CAS: critical-section accesses cannot float above
            // the lock acquisition. Failure needs nothing -- the loop
            // re-reads. Strong CAS, as x86 cmpxchg never fails spuriously.
            if expected & LOCK_BIT == 0
                && self
                    .v
                    .compare_exchange(
                        expected,
                        expected | LOCK_BIT,
                        Ordering::Acquire,
                        Ordering::Relaxed,
                    )
                    .is_ok()
            {
                break;
            }
            core::hint::spin_loop(); /* relax_fence(): pause */
            expected = self.v.load(Ordering::Relaxed);
        }
        // An unlocked word is always clean: dirty bits are set only under
        // the lock and cleared by unlock().
        debug_assert!(expected & DIRTY_MASK == 0);
        let locked = expected | LOCK_BIT;
        // Nobody else may write the word while we hold the lock.
        debug_assert!(locked == self.v.load(Ordering::Relaxed));
        Version(locked)
    }

    /// One-shot lock attempt; never spins waiting for the holder.
    pub fn try_lock(&self) -> bool {
        let expected = self.v.load(Ordering::Relaxed);
        // ordering: as in lock(). Strong CAS matters here: a spurious weak
        // failure would make try_lock() fail on a free, uncontended word,
        // which the reference's x86 cmpxchg cannot do.
        if expected & LOCK_BIT == 0
            && self
                .v
                .compare_exchange(
                    expected,
                    expected | LOCK_BIT,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                )
                .is_ok()
        {
            debug_assert!(expected & DIRTY_MASK == 0);
            debug_assert!(expected | LOCK_BIT == self.v.load(Ordering::Relaxed));
            true
        } else {
            core::hint::spin_loop(); /* relax_fence(): pause */
            false
        }
    }

    /// Publish a clean word: bump the counter for whatever was marked,
    /// clear the lock and dirty bits, and release the lock.
    pub fn unlock(&self) {
        // C++ unlock() forwards *this, i.e. the live word, as the snapshot.
        self.unlock_version(self.snapshot());
    }

    pub fn unlock_version(&self, x: Version) {
        // masstree_invariant((fence(), x.v_ == v_)): debug-only, so the
        // fence exists only in debug builds there too.
        debug_assert!({
            fence(Ordering::Acquire);
            x.0 == self.v.load(Ordering::Relaxed)
        });
        debug_assert!(x.0 & LOCK_BIT != 0);
        let v = if x.0 & SPLITTING_BIT != 0 {
            // A split invalidates every outstanding snapshot: bump the
            // split counter; the mask clears the insert counter, the root
            // bit and the lock/dirty bits in one stroke.
            x.0.wrapping_add(VSPLIT_LOWBIT) & SPLIT_UNLOCK_MASK
        } else {
            // inserting_bit << 2 == vinsert_lowbit, so this bumps the
            // insert counter iff an insert was marked; the mask clears the
            // lock and dirty bits.
            x.0.wrapping_add((x.0 & INSERTING_BIT) << 2) & UNLOCK_MASK
        };
        // ordering: release_fence() + plain store in the reference. The
        // store that publishes the clean version must come after every
        // content store made under the lock; a release store is exactly
        // that edge, and it is what stable()'s acquire fence pairs with.
        self.v.store(v, Ordering::Release);
    }

    // The mark_* family runs under the lock (except mark_root), so the
    // plain |= of the reference is race-free on the write side; a Relaxed
    // load/store pair mirrors it without inventing an atomic RMW.
    fn mark(&self, bits: u64) {
        let v = self.v.load(Ordering::Relaxed);
        self.v.store(v | bits, Ordering::Relaxed);
        // ordering: acquire_fence() in the reference, but the edge it buys
        // under TSO is StoreStore: the dirty mark must become visible no
        // later than the content stores that follow, or a validating
        // reader could see torn contents behind a clean-looking version.
        // In the C++11 model that store->store edge is a release fence; an
        // acquire fence orders nothing store-to-store. This pairs with the
        // acquire fence in has_changed()/has_split().
        fence(Ordering::Release);
    }

    pub fn mark_insert(&self) {
        debug_assert!(self.locked());
        self.mark(INSERTING_BIT);
    }

    pub fn mark_split(&self) {
        debug_assert!(self.locked());
        self.mark(SPLITTING_BIT);
    }

    pub fn mark_change(&self, is_split: bool) {
        debug_assert!(self.locked());
        self.mark((is_split as u64 + 1) << INSERTING_SHIFT);
    }

    /// Deletion is announced as a split so every concurrent reader retries.
    pub fn mark_deleted(&self) -> Version {
        debug_assert!(self.locked());
        self.mark(DELETED_BIT | SPLITTING_BIT);
        self.snapshot()
    }

    // The reference sets the root bit with a plain `v_ |= root_bit` and no
    // locked() invariant, relying on mark_root() only ever running on a node
    // not yet published to other threads. We diverge to a single atomic
    // fetch_or so that, unlike a relaxed load/store pair, it cannot clobber a
    // concurrent lock bit if the node is already visible -- a footgun the C++
    // avoids by convention alone. The fetch_or is Relaxed and the release
    // fence follows it, exactly as mark() does: a Release RMW would order
    // only what precedes it, but the edge we need is StoreStore from the mark
    // to the content stores that come AFTER.
    pub fn mark_root(&self) {
        self.v.fetch_or(ROOT_BIT, Ordering::Relaxed);
        fence(Ordering::Release);
    }

    pub fn mark_nonroot(&self) {
        debug_assert!(self.locked());
        let v = self.v.load(Ordering::Relaxed);
        self.v.store(v & !ROOT_BIT, Ordering::Relaxed);
        // ordering: same StoreStore edge as mark() above.
        fence(Ordering::Release);
    }

    pub fn version_value(&self) -> u64 {
        self.v.load(Ordering::Relaxed)
    }

    pub fn unlocked_version_value(&self) -> u64 {
        self.v.load(Ordering::Relaxed) & UNLOCK_MASK
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::AtomicBool;
    use std::thread;
    use std::vec::Vec;

    #[test]
    fn fresh_nodes_are_clean() {
        let leaf = NodeVersion::new(true);
        assert!(leaf.is_leaf());
        assert!(!leaf.is_root());
        assert!(!leaf.locked());
        assert!(!leaf.inserting());
        assert!(!leaf.splitting());
        assert!(!leaf.deleted());
        assert_eq!(leaf.version_value(), ISLEAF_BIT);

        let internal = NodeVersion::new(false);
        assert!(!internal.is_leaf());
        assert_eq!(internal.version_value(), 0);

        // stable() on a clean word returns it verbatim.
        assert_eq!(leaf.stable().value(), ISLEAF_BIT);
    }

    #[test]
    fn lock_unlock_without_marks_changes_nothing() {
        let nv = NodeVersion::new(true);
        let before = nv.stable();
        let locked = nv.lock();
        assert!(locked.locked());
        assert!(nv.locked());
        assert_eq!(locked.value(), before.value() | LOCK_BIT);
        nv.unlock();
        assert_eq!(nv.version_value(), before.value());
        assert!(!nv.has_changed(before));
        assert!(!nv.has_split(before));
    }

    #[test]
    fn insert_cycle_bumps_insert_counter_only() {
        let nv = NodeVersion::new(true);
        let before = nv.stable();
        nv.lock();
        nv.mark_insert();
        assert!(nv.inserting());
        nv.unlock();
        assert_eq!(nv.version_value(), before.value() + VINSERT_LOWBIT);
        assert!(nv.has_changed(before));
        assert!(!nv.has_split(before));
        assert!(!nv.simple_has_split(before));
        assert!(!nv.locked());
        assert!(!nv.inserting());
    }

    #[test]
    fn split_cycle_bumps_split_counter_and_resets() {
        let nv = NodeVersion::new(true);
        nv.mark_root();
        assert!(nv.is_root());
        // Dirty the insert counter first so the split unlock provably
        // clears it.
        nv.lock();
        nv.mark_insert();
        nv.unlock();
        assert_eq!(
            nv.version_value() & !(VINSERT_LOWBIT - 1) & (VSPLIT_LOWBIT - 1),
            VINSERT_LOWBIT
        );

        let before = nv.stable();
        nv.lock();
        nv.mark_split();
        assert!(nv.splitting());
        nv.unlock();

        let v = nv.version_value();
        assert_eq!(v, (before.value() + VSPLIT_LOWBIT) & SPLIT_UNLOCK_MASK);
        // Split counter bumped by exactly one step; flags, low bits and the
        // insert counter all cleared; root bit cleared; leaf bit kept.
        assert_eq!(v & (VSPLIT_LOWBIT - 1), 0);
        assert!(!nv.is_root());
        assert!(nv.is_leaf());
        assert!(!nv.locked());
        assert!(!nv.splitting());
        assert!(nv.has_changed(before));
        assert!(nv.has_split(before));
        assert!(nv.simple_has_split(before));
    }

    #[test]
    fn mark_change_selects_insert_or_split() {
        let nv = NodeVersion::new(false);
        nv.lock();
        nv.mark_change(false);
        assert!(nv.inserting());
        assert!(!nv.splitting());
        nv.unlock();

        nv.lock();
        nv.mark_change(true);
        assert!(nv.splitting());
        assert!(!nv.inserting());
        nv.unlock();
    }

    #[test]
    fn mark_deleted_reads_as_split() {
        let nv = NodeVersion::new(true);
        let before = nv.stable();
        nv.lock();
        let snap = nv.mark_deleted();
        assert!(snap.deleted());
        assert!(snap.splitting());
        nv.unlock();
        // deleted survives the split-style unlock and readers retry.
        assert!(nv.deleted());
        assert!(nv.has_split(before));
    }

    #[test]
    fn try_lock_fails_only_when_held() {
        let nv = NodeVersion::new(true);
        assert!(nv.try_lock());
        assert!(nv.locked());
        assert!(!nv.try_lock());
        assert_eq!(nv.unlocked_version_value(), ISLEAF_BIT);
        nv.unlock();
        assert!(!nv.locked());
        assert!(nv.try_lock());
        nv.unlock();
    }

    #[test]
    fn stable_never_returns_dirty_across_insert() {
        let nv = NodeVersion::new(true);
        let marked = AtomicBool::new(false);
        thread::scope(|s| {
            s.spawn(|| {
                nv.lock();
                nv.mark_insert();
                marked.store(true, Ordering::Release);
                // Hold the dirty window open long enough for the reader to
                // land in it.
                for _ in 0..100_000 {
                    core::hint::spin_loop();
                }
                nv.unlock();
            });
            while !marked.load(Ordering::Acquire) {
                core::hint::spin_loop();
            }
            // The dirty bit is (or was) up: stable() must spin past it and
            // hand back the post-unlock word, never a dirty one.
            let snap = nv.stable();
            assert_eq!(snap.value() & DIRTY_MASK, 0);
            assert_eq!(snap.value(), ISLEAF_BIT + VINSERT_LOWBIT);
        });
    }

    #[test]
    fn stable_never_returns_dirty_across_split() {
        // A concurrent splitter must also block stable(); on release the
        // reader sees the split counter advanced and no dirty bit.
        let nv = NodeVersion::new(true);
        let marked = AtomicBool::new(false);
        thread::scope(|s| {
            s.spawn(|| {
                nv.lock();
                nv.mark_split();
                marked.store(true, Ordering::Release);
                for _ in 0..100_000 {
                    core::hint::spin_loop();
                }
                nv.unlock();
            });
            while !marked.load(Ordering::Acquire) {
                core::hint::spin_loop();
            }
            let snap = nv.stable();
            // Clean, one split step recorded, immutable leaf bit intact.
            assert_eq!(snap.value() & DIRTY_MASK, 0);
            assert_eq!(snap.value() & (VSPLIT_LOWBIT - 1), 0);
            assert_eq!(snap.value(), ISLEAF_BIT + VSPLIT_LOWBIT);
        });
    }

    #[test]
    fn concurrent_inserts_accumulate_exactly() {
        const N: u64 = 8;
        const M: u64 = 1000;
        let nv = NodeVersion::new(true);
        let stop = AtomicBool::new(false);
        thread::scope(|s| {
            let writers: Vec<_> = (0..N)
                .map(|_| {
                    s.spawn(|| {
                        for _ in 0..M {
                            nv.lock();
                            nv.mark_insert();
                            nv.unlock();
                        }
                    })
                })
                .collect();
            // Validating readers race the writers: a stable snapshot must
            // never be dirty and never lose the immutable leaf bit.
            for _ in 0..2 {
                s.spawn(|| {
                    while !stop.load(Ordering::Relaxed) {
                        let snap = nv.stable();
                        assert_eq!(snap.value() & DIRTY_MASK, 0);
                        assert!(snap.is_leaf());
                        let _ = nv.has_changed(snap);
                    }
                });
            }
            for w in writers {
                w.join().unwrap();
            }
            stop.store(true, Ordering::Relaxed);
        });
        // N*M inserts fit in the 16-bit insert counter without wrapping.
        assert_eq!(nv.version_value(), ISLEAF_BIT + N * M * VINSERT_LOWBIT);
        assert!(!nv.locked());
        assert!(!nv.inserting());
        assert!(!nv.splitting());
    }
}
