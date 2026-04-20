//! Epoch-Based Reclamation (EBR).
//!
//! Fixed-size, `no_std`-friendly reclamation for data structures whose
//! writers retire nodes that concurrent lock-free readers may still
//! dereference. Pattern is classical (Fraser 2004, Harris 2005,
//! Crossbeam's `crossbeam-epoch` — this crate is the same algorithm
//! with per-worker arrays instead of heap-allocated linked lists).
//!
//! Usage:
//! ```ignore
//! let guard = ebr.enter(cpu_index);   // reader pins its epoch
//! let node = tree.lookup(key);         // may follow retired pointers
//! // ... still valid while `guard` lives
//! drop(guard);                         // reader becomes quiescent
//!
//! ebr.retire(cpu_index, node_id);      // writer schedules for free
//! ebr.try_advance(cpu_index);          // periodically drain buckets
//! ```
//!
//! Invariant: a `retire` at epoch `E` is never freed while any CPU
//! has `active_epoch <= E`. Three buckets (`current`, `current - 1`,
//! `current - 2`) are sufficient because a reader that entered at
//! epoch `E` cannot hold a snapshot older than `E`, and the global
//! epoch cannot advance past `E + 2` without every reader having
//! left and re-entered.

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU32, Ordering};

use crate::silo::current_epoch;

/// Upper bound on CPUs we track. Matches kernel::acpi::MAX_CPUS (64)
/// and fits into a u64 bitmap if a bench wants one, but EBR itself
/// doesn't need the bitmap — the per-slot `active_epoch` is the
/// authoritative signal.
pub const MAX_CPUS: usize = 64;

/// How many retirable entities each worker can park in flight before
/// blocking on `try_advance`. 256 * 8 B = 2 KiB per bucket per CPU,
/// 6 KiB per CPU across the three buckets — fits in per-CPU state.
pub const RETIRE_CAP: usize = 256;

/// Epoch sentinel meaning "this CPU is quiescent — not inside any
/// reader critical section". A retire stamped at epoch `E` ignores
/// sentinel CPUs when computing the safe horizon.
const QUIESCENT: u32 = 0;

/// An opaque handle to a retired datum. The caller decides what `u64`
/// means (a pointer, a `NodeId`, an index, etc.); the reclamation
/// callback receives the same bits back.
pub type Retired = u64;

/// Per-CPU retire buckets. Interior-mutable so `retire` and
/// `try_advance` can take `&Ebr` — the safety contract is that only
/// the CPU whose `cpu_index` matches the slot writes to it. Cross-CPU
/// reads go through `Acquire` on `active_epoch` in `try_advance`.
struct SlotInner {
    buckets: [[Retired; RETIRE_CAP]; 3],
    counts: [u32; 3],
}

impl SlotInner {
    const EMPTY: Self = Self {
        buckets: [[0; RETIRE_CAP]; 3],
        counts: [0, 0, 0],
    };
}

#[repr(C, align(64))]
struct Slot {
    active_epoch: AtomicU32,
    inner: UnsafeCell<SlotInner>,
}

impl Slot {
    const EMPTY: Self = Self {
        active_epoch: AtomicU32::new(QUIESCENT),
        inner: UnsafeCell::new(SlotInner::EMPTY),
    };
}

// Safety: single-producer per slot. Cross-CPU access goes through the
// Acquire/Release dance on `active_epoch`. Callers must uphold the
// "owning CPU is the sole writer of `inner`" invariant, same as for
// `WorkerSlot` in silo_bench.
unsafe impl Sync for Slot {}

/// Fixed-size EBR. Callers pass a reclamation closure to
/// `try_advance`; EBR itself doesn't know what `Retired` means (a
/// pointer, a `NodeId`, an arena index — all fit in a `u64`).
pub struct Ebr {
    slots: [Slot; MAX_CPUS],
}

impl Ebr {
    pub const fn new() -> Self {
        const EMPTY: Slot = Slot::EMPTY;
        Self {
            slots: [EMPTY; MAX_CPUS],
        }
    }

    /// Pin the calling CPU to the current global epoch. Must be
    /// matched by dropping the returned `Guard` (or calling `exit`)
    /// before the CPU does anything that may race with a writer's
    /// `retire`.
    pub fn enter(&self, cpu: usize) -> Guard<'_> {
        let epoch = current_epoch();
        self.slots[cpu].active_epoch.store(epoch, Ordering::Release);
        Guard { ebr: self, cpu }
    }

    fn exit(&self, cpu: usize) {
        self.slots[cpu]
            .active_epoch
            .store(QUIESCENT, Ordering::Release);
    }

    /// Schedule a value for reclamation at the current epoch.
    /// Returns `Err(retired)` if the bucket is full — caller must
    /// `try_advance` and retry (or block).
    ///
    /// # Safety
    /// `cpu` must be the caller's owning index. Two calls with the
    /// same `cpu` must not overlap — the slot is single-producer.
    pub unsafe fn retire(&self, cpu: usize, retired: Retired) -> Result<(), Retired> {
        let epoch = current_epoch();
        let bucket = (epoch % 3) as usize;
        let inner = unsafe { &mut *self.slots[cpu].inner.get() };
        let count = inner.counts[bucket] as usize;
        if count >= RETIRE_CAP {
            return Err(retired);
        }
        inner.buckets[bucket][count] = retired;
        inner.counts[bucket] = (count + 1) as u32;
        Ok(())
    }

    /// Attempt to free tombstones whose epoch is now provably safe.
    /// Walks every CPU's `active_epoch` and computes
    /// `horizon = min(global - 1, min_active - 1)`. Buckets in this
    /// CPU's slot with epoch <= `horizon` are drained into `freer`.
    ///
    /// # Safety
    /// Single-producer per slot — `cpu` is the caller's own index;
    /// no concurrent `retire` on the same slot.
    pub unsafe fn try_advance(&self, cpu: usize, mut freer: impl FnMut(Retired)) {
        let global = current_epoch();
        let mut horizon = global.saturating_sub(1);

        for i in 0..MAX_CPUS {
            let ae = self.slots[i].active_epoch.load(Ordering::Acquire);
            if ae == QUIESCENT {
                continue;
            }
            let ae_minus_one = ae.saturating_sub(1);
            if ae_minus_one < horizon {
                horizon = ae_minus_one;
            }
        }

        let inner = unsafe { &mut *self.slots[cpu].inner.get() };
        for off in 0..3u32 {
            let epoch = global.saturating_sub(off);
            if epoch > horizon {
                continue;
            }
            let bucket = (epoch % 3) as usize;
            let count = inner.counts[bucket] as usize;
            for i in 0..count {
                freer(inner.buckets[bucket][i]);
            }
            inner.counts[bucket] = 0;
        }
    }
}

impl Default for Ebr {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII guard produced by [`Ebr::enter`]. Drop re-publishes the CPU
/// as quiescent.
pub struct Guard<'a> {
    ebr: &'a Ebr,
    cpu: usize,
}

impl Drop for Guard<'_> {
    fn drop(&mut self) {
        self.ebr.exit(self.cpu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::silo::advance_epoch;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn retire_is_deferred_until_readers_leave() {
        // A reader pinned at epoch E must prevent reclamation of
        // anything retired at epoch E. Once the reader leaves and the
        // global epoch moves past E+1, the tombstone becomes safe.
        let ebr = Ebr::new();
        let freed = Rc::new(RefCell::new(std::vec::Vec::<Retired>::new()));

        let reader_guard = ebr.enter(1); // cpu 1 pins current epoch
        // Safety: test is single-threaded; cpu 0 and cpu 1 slots are
        // disjoint, and the guard lives on cpu 1's slot only.
        unsafe { ebr.retire(0, 0xDEAD).unwrap() };

        // Advance with reader still pinned — tombstone stays.
        let f = freed.clone();
        unsafe { ebr.try_advance(0, |r| f.borrow_mut().push(r)) };
        assert!(freed.borrow().is_empty(), "must not free while reader is pinned");

        // Reader releases; advance past its epoch.
        drop(reader_guard);
        advance_epoch();
        advance_epoch();

        let f = freed.clone();
        unsafe { ebr.try_advance(0, |r| f.borrow_mut().push(r)) };
        assert_eq!(freed.borrow().as_slice(), &[0xDEAD]);
    }

    #[test]
    fn retire_overflow_returns_error() {
        let ebr = Ebr::new();
        for i in 0..RETIRE_CAP {
            unsafe { ebr.retire(0, i as u64).unwrap() };
        }
        assert_eq!(unsafe { ebr.retire(0, 9999) }, Err(9999));
    }

    #[test]
    fn quiescent_cpu_is_ignored_by_horizon() {
        // A never-entered CPU must not pin the horizon. Otherwise the
        // first worker to retire on boot would stall reclamation
        // forever (since most CPUs never enter a reader section).
        let ebr = Ebr::new();
        let freed = Rc::new(RefCell::new(std::vec::Vec::<Retired>::new()));

        unsafe { ebr.retire(0, 0xBEEF).unwrap() };
        advance_epoch();
        advance_epoch();

        let f = freed.clone();
        unsafe { ebr.try_advance(0, |r| f.borrow_mut().push(r)) };
        assert_eq!(freed.borrow().as_slice(), &[0xBEEF]);
    }

    #[test]
    fn entering_and_exiting_restores_quiescence() {
        let ebr = Ebr::new();
        {
            let _g = ebr.enter(2);
            assert!(ebr.slots[2].active_epoch.load(Ordering::Acquire) != QUIESCENT);
        }
        assert_eq!(
            ebr.slots[2].active_epoch.load(Ordering::Acquire),
            QUIESCENT,
            "Guard drop must re-quiesce the slot",
        );
    }
}
