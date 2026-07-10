//! Bare-metal YCSB scaling bench for the concurrent index.
//!
//! This is the authoritative scaling surface the top-down study calls for:
//! one worker permanently bound to each physical core (the SMP harness boots
//! one AP per core and never migrates), a WARM steady-state tree loaded once
//! before timing, and a fixed DEADLINE WINDOW (equal TSC-cycle budget per
//! core) over which each worker runs YCSB ops -- so throughput is Σ per-core
//! ops over one common interval, not the makespan of the slowest thread.
//!
//! Three standard workloads run back-to-back, phase-synchronized:
//!   C = 100% read   B = 95% read / 5% update   A = 50% read / 50% update
//! Reads take no locks and write nothing (the version-word read path), so C
//! should scale linearly; the update fraction adds the leaf-lock cache-line
//! hand-off. Flat ops/s/core across cores == linear scaling.
//!
//! QEMU's TSC/CPUID are synthetic, so the numbers are only meaningful on the
//! real M920q; on QEMU this validates correctness (no crash, every key still
//! reads back) and the harness, nothing more.

use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use bmdb_core::masstree::Masstree;
use bmdb_serial::serial_println;

use crate::acpi::MAX_CPUS;
use crate::timing;

/// Warm tree size: fills the leaf pool so the tree is a few levels deep and
/// upper splits are rare. The read/update run phase never inserts, so no
/// allocation happens after load.
pub const KEYS: u64 = 6000;
/// Per-core cycle budget per workload. ~200M cycles is tens of ms on real
/// hardware -- long enough to dwarf barrier skew, short enough to stay inside
/// QEMU's patience.
pub const WINDOW_CYCLES: u64 = 200_000_000;

const N_WORKLOADS: usize = 3;
/// Read percentage per workload: C, B, A.
const READ_PCT: [u64; N_WORKLOADS] = [100, 95, 50];
const WORKLOAD_NAME: [&str; N_WORKLOADS] = ["C(100r)", "B(95/5)", "A(50/50)"];

static TREE: Masstree = Masstree::new();

/// Per-core, per-workload op counts. Single-writer per cpu_index; the BSP
/// reads after the final phase barrier.
#[repr(C, align(64))]
struct OpSlot {
    ops: [AtomicU64; N_WORKLOADS],
}
impl OpSlot {
    const EMPTY: Self = Self {
        ops: [const { AtomicU64::new(0) }; N_WORKLOADS],
    };
}
static OPS: [OpSlot; MAX_CPUS] = {
    const E: OpSlot = OpSlot::EMPTY;
    [E; MAX_CPUS]
};

static LIVE_CPU_MASK: AtomicU64 = AtomicU64::new(0);
static WORKERS_READY: AtomicU32 = AtomicU32::new(0);
/// BSP flips `PHASE_GO[p]` to release workload phase `p`; workers bump
/// `PHASE_DONE[p]` when they finish it, so phases stay non-overlapping and
/// each workload's aggregate is measured cleanly.
static PHASE_GO: [AtomicBool; N_WORKLOADS] = [const { AtomicBool::new(false) }; N_WORKLOADS];
static PHASE_DONE: [AtomicU32; N_WORKLOADS] = [const { AtomicU32::new(0) }; N_WORKLOADS];

#[inline(always)]
fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Run one workload for this core's cycle budget; return ops completed.
fn run_window(cpu: usize, read_pct: u64) -> u64 {
    let mut rng = 0x9E37_79B9_7F4A_7C15u64 ^ ((cpu as u64).wrapping_add(1));
    let start = timing::start();
    let mut ops = 0u64;
    loop {
        // Amortize the fenced clock read over a batch.
        for _ in 0..256 {
            let key = (xorshift(&mut rng) % KEYS).to_be_bytes();
            if xorshift(&mut rng) % 100 < read_pct {
                let _ = TREE.get(key);
            } else {
                let _ = TREE.put_on(cpu, key, u64::from_be_bytes(key) + 1);
            }
            ops += 1;
        }
        if timing::end().wrapping_sub(start) >= WINDOW_CYCLES {
            break;
        }
    }
    ops
}

/// Published by the BSP before releasing phase 0 so APs know the barrier size.
static TOTAL_WORKERS: AtomicU32 = AtomicU32::new(0);

/// AP entry, run under `--features ycsb-bench`. Runs all three workloads,
/// phase-synchronized with the BSP.
pub fn ap_worker(cpu_index: usize) {
    assert!(cpu_index < 64, "ycsb-bench bitmap is u64-wide; MAX_CPUS must fit");
    LIVE_CPU_MASK.fetch_or(1u64 << cpu_index, Ordering::Release);
    WORKERS_READY.fetch_add(1, Ordering::Release);

    // Wait for phase 0's release; the BSP publishes TOTAL_WORKERS (Release)
    // before it (Release), so an Acquire read of PHASE_GO[0]==true lets us
    // safely read the published worker count.
    while !PHASE_GO[0].load(Ordering::Acquire) {
        core::hint::spin_loop();
    }
    let total = TOTAL_WORKERS.load(Ordering::Acquire);

    for p in 0..N_WORKLOADS {
        if p > 0 {
            while !PHASE_GO[p].load(Ordering::Acquire) {
                core::hint::spin_loop();
            }
        }
        let ops = run_window(cpu_index, READ_PCT[p]);
        OPS[cpu_index].ops[p].store(ops, Ordering::Relaxed);
        PHASE_DONE[p].fetch_add(1, Ordering::Release);
        // Barrier before the next phase so workloads don't overlap.
        while PHASE_DONE[p].load(Ordering::Acquire) < total {
            core::hint::spin_loop();
        }
    }
}

/// BSP entry. `expected_workers` counts the APs in `ap_worker`; the BSP is
/// also a worker, hence `+ 1`. `_nvme` is unused (this is a pure in-memory
/// index bench) but taken so the caller's handle stays live.
pub fn run(_nvme: &mut bmdb_nvme::Controller, expected_workers: u32) {
    // `expected_workers` is the live AP set smp::init quiesced; every one
    // already signalled ONLINE_APS before entering ap_worker, so this count
    // is stable. Like engine_bench, this assumes no AP that smp::init timed
    // out on wakes late and joins after this snapshot (which would over-count
    // PHASE_DONE and overlap a phase); a post-timeout straggler is out of
    // scope.
    let total = expected_workers + 1;
    let my_cpu = unsafe { crate::percpu::current().cpu_index } as usize;

    let hz = timing::tsc_hz();
    serial_println!(
        "YCSB-BENCH invariant_tsc={} tsc_hz={} keys={} window_cycles={}",
        timing::has_invariant_tsc(),
        hz,
        KEYS,
        WINDOW_CYCLES,
    );

    // Load the warm tree single-threaded before any timing.
    for key in 0..KEYS {
        TREE.put(key.to_be_bytes(), key + 1)
            .expect("ycsb-bench load must fit the pool");
    }
    // Correctness spot-check on a driverless OS: a lost key here is a bug.
    for key in [0u64, KEYS / 2, KEYS - 1] {
        assert_eq!(
            TREE.get(key.to_be_bytes()),
            Some(key + 1),
            "ycsb-bench: warm-load lost a key"
        );
    }

    // Barrier: wait for every AP to be ready, publish the worker count, then
    // release each phase and run it as a worker too.
    LIVE_CPU_MASK.fetch_or(1u64 << my_cpu, Ordering::Release);
    WORKERS_READY.fetch_add(1, Ordering::Release);
    while WORKERS_READY.load(Ordering::Acquire) < total {
        core::hint::spin_loop();
    }
    TOTAL_WORKERS.store(total, Ordering::Release);

    for p in 0..N_WORKLOADS {
        PHASE_GO[p].store(true, Ordering::Release);
        let ops = run_window(my_cpu, READ_PCT[p]);
        OPS[my_cpu].ops[p].store(ops, Ordering::Relaxed);
        PHASE_DONE[p].fetch_add(1, Ordering::Release);
        while PHASE_DONE[p].load(Ordering::Acquire) < total {
            core::hint::spin_loop();
        }
    }

    // Aggregate: for each workload, sum per-core ops (all ran the same
    // WINDOW_CYCLES budget) and report per-core throughput. Flat per-core ==
    // linear scaling.
    let live = LIVE_CPU_MASK.load(Ordering::Acquire);
    let ncores = (live.count_ones()) as u64;
    for p in 0..N_WORKLOADS {
        let mut total_ops = 0u64;
        let mut min_core = u64::MAX;
        let mut max_core = 0u64;
        for i in 0..MAX_CPUS {
            if live & (1u64 << i) == 0 {
                continue;
            }
            let o = OPS[i].ops[p].load(Ordering::Acquire);
            total_ops += o;
            min_core = min_core.min(o);
            max_core = max_core.max(o);
        }
        let agg = timing::ops_per_sec(total_ops, WINDOW_CYCLES, hz);
        let per_core = timing::ops_per_sec(total_ops / ncores.max(1), WINDOW_CYCLES, hz);
        serial_println!(
            "YCSB-BENCH {} cores={} total_ops={} agg_ops_s={} per_core_ops_s={} \
             core_ops(min/max)={}/{}",
            WORKLOAD_NAME[p],
            ncores,
            total_ops,
            agg,
            per_core,
            min_core,
            max_core,
        );
    }
    serial_println!("YCSB-BENCH done (flat per_core_ops_s across core counts == linear scaling)");
}
