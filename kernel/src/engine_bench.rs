//! Concurrent engine microbench.
//!
//! Every Application Processor runs full key-addressable transactions
//! against one shared [`Engine`] whose index is the lock-free concurrent
//! B+tree. Each attempt reads a few keys and writes a few, then commits
//! under the real Silo OCC protocol — but unlike `silo_bench`, every key
//! access goes through `Engine::slot_for`, i.e. a concurrent B+tree
//! lookup (and, on a cold key, a slot allocation). This measures the
//! end-to-end cost the index adds on top of bare OCC, and whether the
//! index scales with cores instead of throttling on a global lock.
//!
//! The BSP pre-populates the whole key space before releasing the start
//! barrier, so the timed loop measures steady-state behaviour: lock-free
//! lookups that hit an already-built tree, plus contended commits. The
//! durable group-commit path is measured separately by `silo_bench`; this
//! bench stays in memory to isolate the index and OCC.
//!
//! Output mirrors `silo_bench`: per-CPU and total commit/abort counts,
//! mean commit/abort cycles, latency percentiles, and a 1 GHz-normalized
//! TPS. QEMU's TSC is not a real clock, so treat absolute cycle counts as
//! comparable only against each other; bare metal uses a monotonic TSC.

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use bmdb_core::bench::compute_stats;
use bmdb_core::engine::{Engine, CbTreeIndex};
use bmdb_core::silo::{CommitOutcome, durable_epoch};
use bmdb_serial::serial_println;

use crate::acpi::MAX_CPUS;

/// Workload parameters. `KEYSPACE` distinct keys build a multi-level
/// concurrent B+tree (each leaf holds up to 15 keys), and are small
/// enough to stay inside both the index node pool and the engine's record
/// pool. `READS`/`WRITES` per transaction keep the read/write sets far
/// under `MAX_RW_SET`, so an abort always reflects a real conflict.
pub const KEYSPACE: u64 = 200;
pub const TXNS_PER_WORKER: usize = 500;
pub const READS_PER_TXN: usize = 4;
pub const WRITES_PER_TXN: usize = 2;

/// The shared store. `const` construction keeps it in the kernel image's
/// BSS rather than the bootloader's dynamic page tables, matching the
/// other benches' static pools.
static ENGINE: Engine<CbTreeIndex> = Engine::concurrent();

/// Per-worker outcome counters. Single producer per slot (the worker
/// whose `cpu_index` owns the row), so `Relaxed` RMW is enough; the BSP
/// reads with `Acquire` after the worker publishes `WORKERS_ONLINE`.
///
/// `align(64)` so each worker's counters own their cache line(s) and
/// adjacent workers do not false-share their per-transaction `fetch_add`s.
#[repr(C, align(64))]
struct WorkerStats {
    commits: AtomicU64,
    aborts_lock: AtomicU64,
    aborts_read: AtomicU64,
    aborts_seq: AtomicU64,
    commit_cycles: AtomicU64,
    abort_cycles: AtomicU64,
    loop_start_tsc: AtomicU64,
    loop_end_tsc: AtomicU64,
}

impl WorkerStats {
    const EMPTY: Self = Self {
        commits: AtomicU64::new(0),
        aborts_lock: AtomicU64::new(0),
        aborts_read: AtomicU64::new(0),
        aborts_seq: AtomicU64::new(0),
        commit_cycles: AtomicU64::new(0),
        abort_cycles: AtomicU64::new(0),
        loop_start_tsc: AtomicU64::new(0),
        loop_end_tsc: AtomicU64::new(0),
    };
}

static STATS: [WorkerStats; MAX_CPUS] = {
    const EMPTY: WorkerStats = WorkerStats::EMPTY;
    [EMPTY; MAX_CPUS]
};

/// Per-CPU latency sample buffer, one entry per attempted commit in TSC
/// cycles. Single-writer-per-slot (only the owning `cpu_index` writes);
/// the BSP reads after every worker publishes `WORKERS_ONLINE`.
#[repr(C, align(64))]
struct LatencySlot {
    samples: UnsafeCell<[u64; TXNS_PER_WORKER]>,
}

// Safety: only the owning CPU writes its slot, and the BSP only reads
// after the `WORKERS_ONLINE` happens-before edge — same invariant as
// silo_bench's per-CPU slots.
unsafe impl Sync for LatencySlot {}

static LATENCY_SAMPLES: [LatencySlot; MAX_CPUS] = {
    const EMPTY: LatencySlot = LatencySlot {
        samples: UnsafeCell::new([0u64; TXNS_PER_WORKER]),
    };
    [EMPTY; MAX_CPUS]
};

/// Workers that finished their run and parked. The BSP polls this, then
/// aggregates and prints.
static WORKERS_ONLINE: AtomicU32 = AtomicU32::new(0);

/// Bitmap of live `cpu_index` values (bit `i` set when worker `i` ran).
/// `smp::init` assigns `cpu_index` in wake order and skips timed-out APs,
/// so the live set may be sparse; iterating the bitmap is the only
/// correct way to visit every worker's slot.
static LIVE_CPU_MASK: AtomicU64 = AtomicU64::new(0);

/// Start barrier: each worker ticks `WORKERS_READY` on entry, then spins
/// on `GO`. The BSP pre-populates the key space and flips `GO` once every
/// worker is ready, so all workers start their timed loop together rather
/// than staggered across SMP bring-up.
static WORKERS_READY: AtomicU32 = AtomicU32::new(0);
static GO: AtomicBool = AtomicBool::new(false);

/// AP entry for the engine bench. Runs under `--features engine-bench`.
pub fn ap_worker(cpu_index: usize) {
    // Distinct nonzero xorshift seed per CPU.
    let mut rng = 0x9E37_79B9_7F4A_7C15u64 ^ ((cpu_index as u64).wrapping_add(1));

    let stats = &STATS[cpu_index];
    // Safety: single-writer-per-cpu_index invariant; see LatencySlot.
    let samples = unsafe { &mut *LATENCY_SAMPLES[cpu_index].samples.get() };

    assert!(cpu_index < 64, "engine-bench bitmap is u64-wide; MAX_CPUS must fit");
    LIVE_CPU_MASK.fetch_or(1u64 << cpu_index, Ordering::Release);

    WORKERS_READY.fetch_add(1, Ordering::Release);
    while !GO.load(Ordering::Acquire) {
        core::hint::spin_loop();
    }

    stats.loop_start_tsc.store(rdtsc(), Ordering::Relaxed);

    for slot in samples.iter_mut().take(TXNS_PER_WORKER) {
        let t0 = rdtsc();
        let outcome = run_one_txn(&mut rng);
        let elapsed = rdtsc().wrapping_sub(t0);
        *slot = elapsed;

        match outcome {
            CommitOutcome::Committed { .. } => {
                stats.commits.fetch_add(1, Ordering::Relaxed);
                stats.commit_cycles.fetch_add(elapsed, Ordering::Relaxed);
            }
            CommitOutcome::AbortedLockConflict => {
                stats.aborts_lock.fetch_add(1, Ordering::Relaxed);
                stats.abort_cycles.fetch_add(elapsed, Ordering::Relaxed);
            }
            CommitOutcome::AbortedReadChanged => {
                stats.aborts_read.fetch_add(1, Ordering::Relaxed);
                stats.abort_cycles.fetch_add(elapsed, Ordering::Relaxed);
            }
            CommitOutcome::AbortedSequenceExhausted => {
                stats.aborts_seq.fetch_add(1, Ordering::Relaxed);
                stats.abort_cycles.fetch_add(elapsed, Ordering::Relaxed);
            }
        }
    }

    stats.loop_end_tsc.store(rdtsc(), Ordering::Relaxed);
    // Release so the BSP's `Acquire` load of WORKERS_ONLINE sees this
    // worker's stats and latency samples.
    WORKERS_ONLINE.fetch_add(1, Ordering::Release);
}

/// One OCC attempt through the full engine: pick a few keys to read and a
/// few to write, all resolved through the concurrent index, then commit.
/// Every key is pre-populated, so `get`/`put` never fail for space here;
/// a returned error still aborts the attempt rather than panicking.
fn run_one_txn(rng: &mut u64) -> CommitOutcome {
    let mut txn = ENGINE.begin();

    for _ in 0..READS_PER_TXN {
        let key = (xorshift64(rng) % KEYSPACE).to_be_bytes();
        if txn.get(key).is_err() {
            return CommitOutcome::AbortedReadChanged;
        }
    }
    for _ in 0..WRITES_PER_TXN {
        let key = (xorshift64(rng) % KEYSPACE).to_be_bytes();
        let value = xorshift64(rng);
        if txn.put(key, value).is_err() {
            return CommitOutcome::AbortedLockConflict;
        }
    }
    txn.commit()
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

/// BSP entry for the engine bench. Call after `smp::init` returns (every
/// AP has entered `ap_worker` and is spinning on `GO`). Pre-populates the
/// key space, releases the barrier, runs the BSP's own worker, then
/// aggregates and prints. `expected_workers` is the count of APs that
/// entered `ap_worker`; the BSP is also a worker, hence `+ 1`.
pub fn run(nvme: &mut bmdb_nvme::Controller, expected_workers: u32) {
    // `expected_workers` is the live AP set `smp::init` brought up and
    // quiesced; every one already signalled `ONLINE_APS` before entering
    // `ap_worker`, so this count is stable here. Like `silo_bench`, this
    // bench assumes no AP that `smp::init` timed out on wakes up late and
    // joins after this snapshot — such an AP would race the aggregation.
    // The whole-worker barrier and `WORKERS_ONLINE` gate below cover every
    // counted worker; a post-timeout straggler is out of scope by design.
    let total_workers = expected_workers + 1;
    let my_cpu = unsafe { crate::percpu::current().cpu_index } as usize;

    // Build the whole index up front on the BSP while the APs spin on
    // `GO`. This makes the timed loop measure steady-state lookups on an
    // already-grown tree rather than the one-time slot-allocation cost.
    for key in 0..KEYSPACE {
        let ok = ENGINE.transaction(8, |txn| txn.put(key.to_be_bytes(), key + 1));
        assert!(ok.is_some(), "engine-bench pre-populate must commit");
    }

    // Barrier: the BSP counts too. Once every worker is ready, flip `GO`
    // so all timed loops start within a few cycles of one another. The
    // `GO` release also publishes the pre-populated store to every worker.
    WORKERS_READY.fetch_add(1, Ordering::Release);
    while WORKERS_READY.load(Ordering::Acquire) < total_workers {
        core::hint::spin_loop();
    }
    GO.store(true, Ordering::Release);

    ap_worker(my_cpu);

    while WORKERS_ONLINE.load(Ordering::Acquire) < total_workers {
        core::hint::spin_loop();
    }

    // Aggregate per-worker numbers over the live-CPU bitmap.
    let live_mask = LIVE_CPU_MASK.load(Ordering::Acquire);
    let mut total_commits: u64 = 0;
    let mut total_lock: u64 = 0;
    let mut total_read: u64 = 0;
    let mut total_seq: u64 = 0;
    let mut total_commit_cycles: u64 = 0;
    let mut total_abort_cycles: u64 = 0;
    let mut wall_start: u64 = u64::MAX;
    let mut wall_end: u64 = 0;

    for i in 0..MAX_CPUS {
        if live_mask & (1u64 << i) == 0 {
            continue;
        }
        let s = &STATS[i];
        let c = s.commits.load(Ordering::Acquire);
        let al = s.aborts_lock.load(Ordering::Acquire);
        let ar = s.aborts_read.load(Ordering::Acquire);
        let aseq = s.aborts_seq.load(Ordering::Acquire);
        let ccyc = s.commit_cycles.load(Ordering::Acquire);
        let acyc = s.abort_cycles.load(Ordering::Acquire);
        let lst = s.loop_start_tsc.load(Ordering::Acquire);
        let lend = s.loop_end_tsc.load(Ordering::Acquire);

        // Safety: samples owned by this CPU's slot; every worker is
        // parked. `compute_stats` sorts in place (consume-once).
        let samples = unsafe { &mut *LATENCY_SAMPLES[i].samples.get() };
        let per_cpu = compute_stats(samples);

        let mean_commit = if c > 0 { ccyc / c } else { 0 };
        let abort_n = al + ar + aseq;
        let mean_abort = if abort_n > 0 { acyc / abort_n } else { 0 };

        serial_println!(
            "ENGINE-BENCH cpu{} commits={} aborts(lock/read/seq)={}/{}/{} \
             mean_commit_cycles={} mean_abort_cycles={}",
            i, c, al, ar, aseq, mean_commit, mean_abort,
        );
        serial_println!("ENGINE-BENCH cpu{} latency {}", i, per_cpu);

        total_commits += c;
        total_lock += al;
        total_read += ar;
        total_seq += aseq;
        total_commit_cycles += ccyc;
        total_abort_cycles += acyc;
        if lst != 0 && lst < wall_start {
            wall_start = lst;
        }
        if lend > wall_end {
            wall_end = lend;
        }
    }

    let total_attempts = total_commits + total_lock + total_read + total_seq;
    let total_aborts = total_attempts - total_commits;
    let mean_commit_cycles = if total_commits > 0 {
        total_commit_cycles / total_commits
    } else {
        0
    };
    let mean_abort_cycles = if total_aborts > 0 {
        total_abort_cycles / total_aborts
    } else {
        0
    };
    let wall_cycles = if wall_end > wall_start { wall_end - wall_start } else { 0 };
    // Commits per second assuming a 1 GHz TSC; multiply by the real GHz
    // rate to rescale. QEMU's TCG TSC is not a true clock.
    let tps_at_1ghz = if wall_cycles > 0 {
        total_commits.saturating_mul(1_000_000_000) / wall_cycles
    } else {
        0
    };

    serial_println!(
        "ENGINE-BENCH total attempts={} commits={} aborts={} (lock={} read={} seq={}) \
         mean_commit_cycles={} mean_abort_cycles={}",
        total_attempts,
        total_commits,
        total_aborts,
        total_lock,
        total_read,
        total_seq,
        mean_commit_cycles,
        mean_abort_cycles,
    );
    serial_println!(
        "ENGINE-BENCH keyspace={} wall_cycles={} tps_at_1ghz={} (scale by actual GHz)",
        KEYSPACE, wall_cycles, tps_at_1ghz,
    );

    // Durability capstone: prove the concurrent-index engine also persists.
    // Every worker has parked, so these durable commits run uncontended;
    // each appends its write set plus a commit boundary to the WAL and
    // flushes it to the real device before returning. This closes the loop
    // on "concurrency control + index + durability" in one workload.
    let t0 = rdtsc();
    let mut durable_commits: u64 = 0;
    for key in 0..DURABLE_KEYS {
        let r = ENGINE.transaction_durable(nvme, 8, |txn| {
            let prev = txn.get(key.to_be_bytes())?.unwrap_or(0);
            txn.put(key.to_be_bytes(), prev.wrapping_add(1))?;
            Ok(())
        });
        match r {
            Ok(Some(())) => durable_commits += 1,
            Ok(None) => {}
            Err(e) => {
                serial_println!("ENGINE-BENCH durable commit failed: {:?}", e);
                break;
            }
        }
    }
    let durable_cycles = rdtsc().wrapping_sub(t0);
    serial_println!(
        "ENGINE-BENCH durable commits={} cycles={} durable_epoch={}",
        durable_commits, durable_cycles, durable_epoch(),
    );
}

/// Keys touched by the durability capstone. A small slice of the key
/// space — enough to exercise the WAL append + flush path without a long
/// synchronous run.
const DURABLE_KEYS: u64 = 16;
