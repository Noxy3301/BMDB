//! Concurrent Silo microbench.
//!
//! Every Application Processor runs a transaction loop over a shared
//! `Record` pool; each attempt reads a few records and writes a few,
//! then commits under the real Silo OCC protocol. Per-worker commit /
//! abort counters live in a static array indexed by `cpu_index`, so
//! aggregation is lock-free. After every AP parks itself the BSP
//! drains every worker's log buffer through the group-commit
//! [`bmdb_core::silo::persist`] path, flushing the WAL exactly once —
//! this exercises the durability gate the next boot's recovery will
//! replay.
//!
//! The output is deliberately coarse: TSC cycles, commit/abort counts
//! by reason, and the peak epoch that was made durable. QEMU's TSC is
//! not a real clock so absolute numbers are only meaningful against
//! each other; bare metal replaces that with a monotonic host TSC.

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU64, AtomicU32, Ordering};

use bmdb_core::silo::{
    self, CommitOutcome, LogBuffer, Record, Tid, TxnState, current_epoch, durable_epoch,
};
use bmdb_core::wal::Wal;
use bmdb_serial::serial_println;

use crate::acpi::MAX_CPUS;

/// Workload parameters. Tuned for QEMU -smp 4 + a ~256-record pool so
/// each worker sees a blend of uncontended fast paths and contended
/// abort paths. Larger `RECORDS` reduces contention; smaller `TXNS`
/// keeps the total runtime inside QEMU's default patience.
pub const RECORDS: usize = 16;
pub const TXNS_PER_WORKER: usize = 500;
pub const READS_PER_TXN: usize = 4;
pub const WRITES_PER_TXN: usize = 2;

/// Shared record pool. Every worker indexes into this slice; the
/// concurrency story is Silo's (Record's `tid` and `value` are both
/// atomics, and the commit protocol owns the lock bit). Storing the
/// pool as a `static` with const initialization keeps it out of the
/// bootloader's dynamic page tables.
static RECORDS_POOL: [Record; RECORDS] = {
    const EMPTY: Record = Record::new(Tid::new(0, 0), 0);
    [EMPTY; RECORDS]
};

/// Per-worker scratch: a mutable TxnState and a LogBuffer, both owned
/// exclusively by the worker whose `cpu_index` matches the slot index.
/// `UnsafeCell` because we hand out raw mutable pointers via
/// `slot_mut()`; the safety argument is single-writer per slot.
struct WorkerSlot {
    txn: UnsafeCell<TxnState>,
    log: UnsafeCell<LogBuffer>,
}

// Safety: each slot is mutated by exactly one CPU (identified by the
// matching `cpu_index`). Cross-CPU access only happens in `run` after
// every worker has parked in `hlt`, which is a happens-before edge
// published by `WORKERS_ONLINE.fetch_add(Release)` / `load(Acquire)`.
unsafe impl Sync for WorkerSlot {}

static WORKERS: [WorkerSlot; MAX_CPUS] = {
    const EMPTY: WorkerSlot = WorkerSlot {
        txn: UnsafeCell::new(TxnState::new(0)),
        log: UnsafeCell::new(LogBuffer::new()),
    };
    [EMPTY; MAX_CPUS]
};

/// Per-worker outcome counters. Single producer per slot (the worker
/// whose `cpu_index` owns the row), so `Relaxed` RMW is enough.
/// Aggregation on the BSP reads with `Acquire` after the worker has
/// published `WORKERS_ONLINE`.
struct WorkerStats {
    commits: AtomicU64,
    aborts_lock: AtomicU64,
    aborts_read: AtomicU64,
    aborts_seq: AtomicU64,
    commit_cycles: AtomicU64,
    /// Log-buffer overflow events. Each one means an OCC-committed
    /// transaction whose write set could not be recorded — the Record
    /// atomics reflect the commit, but recovery would not replay it.
    log_overflows: AtomicU64,
}

impl WorkerStats {
    const EMPTY: Self = Self {
        commits: AtomicU64::new(0),
        aborts_lock: AtomicU64::new(0),
        aborts_read: AtomicU64::new(0),
        aborts_seq: AtomicU64::new(0),
        commit_cycles: AtomicU64::new(0),
        log_overflows: AtomicU64::new(0),
    };
}

static STATS: [WorkerStats; MAX_CPUS] = {
    const EMPTY: WorkerStats = WorkerStats::EMPTY;
    [EMPTY; MAX_CPUS]
};

/// Workers that have finished their run and parked themselves. The
/// BSP polls this counter, then drains every buffer and prints stats.
static WORKERS_ONLINE: AtomicU32 = AtomicU32::new(0);

/// Start barrier. Each worker ticks `WORKERS_READY` on entry, then
/// spins until `GO` turns true — the BSP releases the barrier after
/// `smp::init` has signalled every AP, so every worker begins its
/// commit loop within a few hundred cycles of the others rather than
/// staggered over the SMP bring-up timeline.
static WORKERS_READY: AtomicU32 = AtomicU32::new(0);
static GO: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// AP entry for the Silo bench. Replaces the SMP-f contention loop
/// under `--features silo-bench`.
pub fn ap_worker(cpu_index: usize) {
    // Distinct xorshift seed per CPU. Seed must be nonzero — combining
    // a constant golden-ratio mask with the cpu_index keeps CPU 0 out
    // of the degenerate state.
    let mut rng = 0x9E37_79B9_7F4A_7C15u64 ^ ((cpu_index as u64).wrapping_add(1));

    let txn_ptr = WORKERS[cpu_index].txn.get();
    let log_ptr = WORKERS[cpu_index].log.get();
    // Safety: single-writer-per-slot invariant — this CPU is the only
    // one that ever dereferences its own slot.
    let txn = unsafe { &mut *txn_ptr };
    let log = unsafe { &mut *log_ptr };

    let stats = &STATS[cpu_index];

    // Start barrier: wait for every worker to reach this point and
    // the BSP to flip `GO`. Without it, workers woken earlier would
    // finish their commit loop before workers woken later had even
    // started, making the bench a sequence of single-CPU runs.
    WORKERS_READY.fetch_add(1, Ordering::Release);
    while !GO.load(Ordering::Acquire) {
        core::hint::spin_loop();
    }

    for _ in 0..TXNS_PER_WORKER {
        txn.reset(current_epoch());
        let t0 = rdtsc();
        let outcome = run_one_txn(txn, &mut rng);
        let elapsed = rdtsc().wrapping_sub(t0);

        match outcome {
            CommitOutcome::Committed { new_tid } => {
                stats.commits.fetch_add(1, Ordering::Relaxed);
                stats.commit_cycles.fetch_add(elapsed, Ordering::Relaxed);
                // Log path. If the buffer fills, count the overflow so
                // the report can flag durability loss. A production
                // path would either block the worker or hand off to a
                // logger; the feasibility bench treats the overflow as
                // an OCC-committed-but-not-durable event.
                if log.record_commit(new_tid, txn.write_entries()).is_err() {
                    stats.log_overflows.fetch_add(1, Ordering::Relaxed);
                }
            }
            CommitOutcome::AbortedLockConflict => {
                stats.aborts_lock.fetch_add(1, Ordering::Relaxed);
            }
            CommitOutcome::AbortedReadChanged => {
                stats.aborts_read.fetch_add(1, Ordering::Relaxed);
            }
            CommitOutcome::AbortedSequenceExhausted => {
                stats.aborts_seq.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    // Release so the BSP's `Acquire` load of WORKERS_ONLINE sees all
    // updates to this worker's stats / buffer.
    WORKERS_ONLINE.fetch_add(1, Ordering::Release);
}

/// One OCC attempt: pick a few random records to read and a few to
/// write, issue the Silo commit protocol, return the outcome. Reads
/// and writes can overlap — commit resolves the read-set/write-set
/// intersection by ignoring the lock bit when the caller is the one
/// holding it.
fn run_one_txn(txn: &mut TxnState, rng: &mut u64) -> CommitOutcome {
    for _ in 0..READS_PER_TXN {
        let idx = (xorshift64(rng) as usize) % RECORDS;
        let record = &RECORDS_POOL[idx];
        if let Some((observed_tid, _value)) = record.read_snapshot() {
            if txn.add_read(record, observed_tid).is_err() {
                // Read-set overflow; the bench keeps MAX_RW_SET wide
                // enough (64) that this shouldn't fire at these
                // parameters, but treat it as a read-abort rather than
                // panicking.
                return CommitOutcome::AbortedReadChanged;
            }
        }
        // read_snapshot == None means a concurrent writer holds the
        // lock. Skip adding to the read set; commit's read-validation
        // won't see it and the read "result" is effectively discarded.
    }

    for _ in 0..WRITES_PER_TXN {
        let idx = (xorshift64(rng) as usize) % RECORDS;
        let record = &RECORDS_POOL[idx];
        let key = (idx as u64).to_be_bytes();
        // Value encodes worker-identifying bits via the rng state so
        // different commits of the same record produce distinct values
        // (a zero always-write would make the bench nothing but
        // version bumps).
        let new_value = *rng;
        if txn.add_write(record, key, new_value).is_err() {
            return CommitOutcome::AbortedLockConflict;
        }
    }

    // Safety: every `record_addr` in `txn` points into `RECORDS_POOL`,
    // which is a `'static` region. The commit call's pointer contract
    // is therefore upheld for the lifetime of the bench.
    unsafe { silo::commit(txn) }
}

#[inline(always)]
fn rdtsc() -> u64 {
    unsafe { core::arch::x86_64::_rdtsc() }
}

#[inline(always)]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// BSP entry for the Silo bench. Call after [`crate::smp::init`] has
/// returned — every non-BSP AP has already finished `ap_worker` by
/// then because `smp::init` polls `ONLINE_APS` until all APs report
/// in. Drives durability (one `persist` across every buffer), prints
/// aggregates, parks.
pub fn run(nvme: &mut bmdb_nvme::Controller, expected_workers: u32) {
    // `expected_workers` is the number of APs that entered
    // `ap_worker`; the BSP itself is also a worker in this bench, so
    // we add 1. On QEMU -smp 4 with 3 APs online, expected_workers=3
    // and we wait for WORKERS_ONLINE == 4.
    let total_workers = expected_workers + 1;
    let my_cpu = unsafe { crate::percpu::current().cpu_index } as usize;

    // BSP also counts toward `WORKERS_READY`. Without this tick the
    // barrier release condition would miss the BSP and all APs would
    // hang on `GO`. The subsequent loop is the BSP's referee role:
    // once every worker (including the BSP) is ready, flip `GO` so
    // everyone starts their commit loop within a handful of cycles of
    // one another.
    WORKERS_READY.fetch_add(1, Ordering::Release);
    while WORKERS_READY.load(Ordering::Acquire) < total_workers {
        core::hint::spin_loop();
    }
    GO.store(true, Ordering::Release);

    ap_worker(my_cpu);

    // Spin until every worker has published its stats. APs are already
    // hlt'd so this is a quick bounded wait.
    while WORKERS_ONLINE.load(Ordering::Acquire) < total_workers {
        core::hint::spin_loop();
    }

    // Aggregate per-worker numbers.
    let mut total_commits: u64 = 0;
    let mut total_lock: u64 = 0;
    let mut total_read: u64 = 0;
    let mut total_seq: u64 = 0;
    let mut total_cycles: u64 = 0;
    let mut total_overflows: u64 = 0;
    for i in 0..(total_workers as usize) {
        let s = &STATS[i];
        let c = s.commits.load(Ordering::Acquire);
        let al = s.aborts_lock.load(Ordering::Acquire);
        let ar = s.aborts_read.load(Ordering::Acquire);
        let aseq = s.aborts_seq.load(Ordering::Acquire);
        let cyc = s.commit_cycles.load(Ordering::Acquire);
        let lov = s.log_overflows.load(Ordering::Acquire);
        serial_println!(
            "SILO-BENCH cpu{} commits={} aborts(lock/read/seq)={}/{}/{} log_overflows={} cycles={}",
            i,
            c,
            al,
            ar,
            aseq,
            lov,
            cyc,
        );
        total_commits += c;
        total_lock += al;
        total_read += ar;
        total_seq += aseq;
        total_cycles += cyc;
        total_overflows += lov;
    }

    let total_attempts =
        total_commits + total_lock + total_read + total_seq;
    let mean_commit_cycles = if total_commits > 0 {
        total_cycles / total_commits
    } else {
        0
    };
    serial_println!(
        "SILO-BENCH total attempts={} commits={} aborts={} (lock={} read={} seq={}) log_overflows={} mean_commit_cycles={}",
        total_attempts,
        total_commits,
        total_attempts - total_commits,
        total_lock,
        total_read,
        total_seq,
        total_overflows,
        mean_commit_cycles,
    );

    // Group commit pass: drain every worker's buffer through one flush.
    // Every worker is parked, so the `persist` contract (no live
    // commit buffering a log entry for an epoch about to be published)
    // is satisfied trivially.
    let mut wal = match Wal::recover(nvme) {
        Ok(w) => w,
        Err(e) => {
            serial_println!("SILO-BENCH Wal::recover failed: {:?}", e);
            return;
        }
    };
    serial_println!(
        "SILO-BENCH wal recovered: next_lsn={}, starting persist",
        wal.next_lsn(),
    );

    let t0 = rdtsc();
    let mut buffer_refs = [core::ptr::null_mut::<LogBuffer>(); MAX_CPUS];
    for i in 0..(total_workers as usize) {
        buffer_refs[i] = WORKERS[i].log.get();
    }
    let (persist_result, logged) = drain_and_persist(nvme, &mut wal, &buffer_refs[..total_workers as usize]);
    let persist_cycles = rdtsc().wrapping_sub(t0);

    match persist_result {
        Ok(durable) => serial_println!(
            "SILO-BENCH persist OK durable_epoch={} records_logged={} cycles={}",
            durable,
            logged,
            persist_cycles,
        ),
        Err(e) => serial_println!("SILO-BENCH persist failed: {:?}", e),
    }
    serial_println!(
        "SILO-BENCH global durable_epoch={} current_epoch={}",
        durable_epoch(),
        current_epoch(),
    );
}

/// Thin wrapper that turns the raw pointer array into the `&mut [&mut
/// LogBuffer]` slice expected by `silo::persist`. Split out so the
/// `unsafe` isolation is visible.
fn drain_and_persist(
    nvme: &mut bmdb_nvme::Controller,
    wal: &mut Wal,
    buffer_ptrs: &[*mut LogBuffer],
) -> (Result<u32, bmdb_nvme::IoError>, u64) {
    // Build a `[&mut LogBuffer; N]` on the stack. MAX_CPUS is fixed at
    // 64, so this array is always large enough.
    let mut refs: [Option<&mut LogBuffer>; MAX_CPUS] = [const { None }; MAX_CPUS];
    let mut total_records: u64 = 0;
    for (i, ptr) in buffer_ptrs.iter().enumerate() {
        // Safety: the BSP is the sole writer at this point; every AP
        // has parked in `hlt` and published its buffer state via
        // WORKERS_ONLINE's Release.
        let buf = unsafe { &mut **ptr };
        total_records += buf.len() as u64;
        refs[i] = Some(buf);
    }
    // Flatten to the exact slice length persist expects.
    let mut flattened: [*mut LogBuffer; MAX_CPUS] = [core::ptr::null_mut(); MAX_CPUS];
    for (i, opt) in refs.iter_mut().enumerate() {
        if let Some(b) = opt.take() {
            flattened[i] = b as *mut LogBuffer;
        }
    }
    // Final cast back to `&mut [&mut LogBuffer]` — safe because we
    // just built these mutable references from unique buffers.
    let n = buffer_ptrs.len();
    let slice = unsafe {
        core::slice::from_raw_parts_mut(
            flattened.as_mut_ptr() as *mut &mut LogBuffer,
            n,
        )
    };
    (silo::persist(nvme, wal, slice), total_records)
}
