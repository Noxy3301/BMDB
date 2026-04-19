//! Non-reentrant spinlock for short critical sections on bare metal.
//!
//! Uses a test-then-test-and-set loop: threads spin on a relaxed load until
//! the lock looks free, *then* attempt a single acquire via
//! `compare_exchange_weak`. This keeps the lock cache line in the shared
//! state while contended, which is the standard trick for avoiding the
//! cache-line ping-pong caused by plain test-and-set under contention.
//!
//! The wait loop uses `core::hint::spin_loop`, which compiles to `PAUSE` on
//! x86_64. PAUSE tells the CPU the current iteration is a spin-wait, which
//! both reduces power draw and lets the memory pipeline drain in time for
//! the eventual successful load — so the instant the lock frees, we see it.
//!
//! Non-reentrant: a thread that already holds the lock will deadlock if it
//! calls `lock()` again. Phase 3 has no re-entrant critical sections.
//!
//! Interrupt safety: this lock does not mask interrupts. A critical section
//! on the same CPU that can be entered from both normal flow and an
//! interrupt handler needs a separate primitive that disables interrupts
//! around the acquire/release. Added when the interrupt paths reach the
//! data protected by one of these locks.

use core::cell::UnsafeCell;
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};
use core::sync::atomic::{AtomicBool, Ordering};

pub struct SpinLock<T> {
    // True when held. Acquire is CAS from false to true with Acquire
    // ordering so the critical section sees writes that happened before
    // the prior release.
    locked: AtomicBool,
    value: UnsafeCell<T>,
}

/// Required because the default `Sync` impl for a type containing
/// `UnsafeCell<T>` is absent. The atomic serialization makes concurrent
/// access through `&SpinLock<T>` sound as long as `T: Send` (a value
/// produced on thread A may be read by thread B once the handoff is via
/// the lock acquire/release fence).
unsafe impl<T: Send> Send for SpinLock<T> {}
unsafe impl<T: Send> Sync for SpinLock<T> {}

impl<T> SpinLock<T> {
    pub const fn new(value: T) -> Self {
        Self {
            locked: AtomicBool::new(false),
            value: UnsafeCell::new(value),
        }
    }

    /// Block until the lock is acquired. Returns a RAII guard that releases
    /// on drop.
    pub fn lock(&self) -> SpinLockGuard<'_, T> {
        loop {
            // Acquire ordering: a successful exchange has to happen-before
            // reads of the guarded data.
            if self
                .locked
                .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                return SpinLockGuard {
                    lock: self,
                    _not_sync_for_t_not_sync: PhantomData,
                };
            }
            // Back off with a relaxed-load spin until the lock looks free
            // again, avoiding the cache-line ping-pong that a bare CAS loop
            // would generate on a contended lock.
            while self.locked.load(Ordering::Relaxed) {
                core::hint::spin_loop();
            }
        }
    }

    /// Non-blocking lock acquisition. Returns `None` if the lock is held by
    /// another party.
    pub fn try_lock(&self) -> Option<SpinLockGuard<'_, T>> {
        self.locked
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .ok()
            .map(|_| SpinLockGuard {
                lock: self,
                _not_sync_for_t_not_sync: PhantomData,
            })
    }

    /// Observe the current held/free state. Useful only for tests and
    /// diagnostics — a `false` observation can be stale the next cycle.
    pub fn is_locked(&self) -> bool {
        self.locked.load(Ordering::Relaxed)
    }
}

pub struct SpinLockGuard<'a, T> {
    lock: &'a SpinLock<T>,
    // Without this marker, the guard auto-derives `Sync` whenever
    // `&SpinLock<T>` is `Sync`, which only requires `T: Send`. That is
    // unsound for `T: !Sync` (e.g. `Cell`): sharing a
    // `&SpinLockGuard<Cell<_>>` across threads would expose concurrent
    // `&Cell<_>`. `PhantomData<&'a mut T>` adds `T: Sync` to the composed
    // auto-bound, so the final bound becomes `Sync iff T: Send + Sync` —
    // stricter than `std::sync::MutexGuard` (which is `Sync iff T: Sync`),
    // and strictly safer. The extra `T: Send` requirement only matters
    // for the rare Sync-but-not-Send types (e.g. another `MutexGuard`),
    // which we are not expected to wrap in a SpinLock.
    _not_sync_for_t_not_sync: PhantomData<&'a mut T>,
}

impl<T> Deref for SpinLockGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // The lock is held, so no other reference can exist.
        unsafe { &*self.lock.value.get() }
    }
}

impl<T> DerefMut for SpinLockGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.lock.value.get() }
    }
}

impl<T> Drop for SpinLockGuard<'_, T> {
    fn drop(&mut self) {
        // Release ordering pairs with the Acquire on the successful
        // compare-exchange in `lock` / `try_lock`, publishing every write
        // made through the guard to the next acquirer.
        self.lock.locked.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn uncontended_lock_and_unlock_round_trip() {
        let lock = SpinLock::new(0u64);
        assert!(!lock.is_locked());
        {
            let mut guard = lock.lock();
            assert!(lock.is_locked());
            *guard = 42;
        }
        assert!(!lock.is_locked());
        assert_eq!(*lock.lock(), 42);
    }

    #[test]
    fn try_lock_fails_while_held_and_succeeds_once_released() {
        let lock = SpinLock::new(7u64);
        let guard = lock.lock();
        assert!(lock.try_lock().is_none(), "should fail while held");
        drop(guard);
        assert!(lock.try_lock().is_some(), "should succeed after release");
    }

    #[test]
    fn drop_releases_without_explicit_unlock() {
        let lock = SpinLock::new(());
        {
            let _g = lock.lock();
            assert!(lock.is_locked());
        }
        // Guard dropped; lock must be free.
        assert!(!lock.is_locked());
    }

    #[test]
    fn concurrent_increments_sum_to_expected_value() {
        // Four threads each increment the shared counter 25_000 times;
        // final value must equal 100_000 regardless of interleaving.
        let lock = Arc::new(SpinLock::new(0u64));
        let mut handles = std::vec::Vec::new();
        let threads = 4;
        let iters_per_thread = 25_000u64;
        for _ in 0..threads {
            let lock = Arc::clone(&lock);
            handles.push(thread::spawn(move || {
                for _ in 0..iters_per_thread {
                    let mut g = lock.lock();
                    *g += 1;
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(*lock.lock(), threads * iters_per_thread);
    }

    #[test]
    fn concurrent_pair_updates_are_never_observed_mid_write() {
        // Two-field invariant under contention: writers increment both
        // fields under the lock, readers sample both fields under the
        // lock. A broken Acquire/Release (or broken mutual exclusion)
        // could let a reader observe `a != b` mid-write. Unlike a
        // `join`-based test, the reader and writers run concurrently, so
        // the lock's publish/observe edge is the only thing preventing
        // torn reads.
        let lock = Arc::new(SpinLock::new((0u64, 0u64)));
        let iters_per_writer = 5_000u64;
        let writer_count = 4;

        let mut writers = std::vec::Vec::new();
        for _ in 0..writer_count {
            let lock = Arc::clone(&lock);
            writers.push(thread::spawn(move || {
                for _ in 0..iters_per_writer {
                    let mut g = lock.lock();
                    g.0 = g.0.wrapping_add(1);
                    g.1 = g.1.wrapping_add(1);
                }
            }));
        }

        let reader = {
            let lock = Arc::clone(&lock);
            thread::spawn(move || {
                // Sample until we've seen roughly as many reads as total
                // writes; detects any tear the writers may produce.
                for _ in 0..(iters_per_writer * writer_count as u64) {
                    let g = lock.lock();
                    assert_eq!(g.0, g.1, "torn pair observed under lock");
                }
            })
        };

        for h in writers {
            h.join().unwrap();
        }
        reader.join().unwrap();
        let g = lock.lock();
        assert_eq!(g.0, iters_per_writer * writer_count as u64);
        assert_eq!(g.0, g.1);
    }
}
