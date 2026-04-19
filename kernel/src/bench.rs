//! KV microbenchmark. Runs a put phase followed by a random-probe get
//! phase against the recovered KV store and prints TSC-cycle latency
//! summaries over the serial line.
//!
//! Numbers reported on QEMU are useful only as harness validation — the
//! guest TSC is not a real clock. The same code path produces the real
//! numbers on bare metal.

use bmdb_core::bench::{BenchStats, compute_stats};
use bmdb_core::kv::Kv;
use bmdb_core::storage::BlockStorage;
use bmdb_serial::serial_println;

// Samples are stack-allocated arrays of u64. Keep counts moderate: put is
// bounded by B+tree pool capacity (POOL_SIZE = 64), get has no such cap
// but we read-back only the keys we inserted.
const PUT_COUNT: usize = 128;
const GET_ITERS: usize = 10_000;

#[inline(always)]
fn rdtsc() -> u64 {
    // `_rdtsc` is non-serializing. For a coarse microbench against a
    // single-threaded bare-metal kernel the ordering imprecision is
    // dominated by the op cost itself (hundreds of cycles minimum for a
    // tree descent).
    unsafe { core::arch::x86_64::_rdtsc() }
}

fn key_from_index(i: u64) -> [u8; 8] {
    i.to_be_bytes()
}

fn value_from_index(i: u64) -> [u8; 8] {
    // Trivial derivation; the bench does not care about content, only that
    // writes are distinguishable so upserts register as such.
    (i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).to_be_bytes()
}

/// Run `n` unique-key puts starting at `start_key`. Returns the populated
/// key range so the subsequent read phase probes only keys that actually
/// landed in the tree (put may stop early on `TreeFull`).
fn bench_puts<S: BlockStorage>(
    kv: &mut Kv,
    storage: &mut S,
    start_key: u64,
    n: usize,
    samples: &mut [u64],
) -> (BenchStats, u64) {
    let cap = core::cmp::min(n, samples.len());
    let mut written: u64 = 0;
    for i in 0..cap {
        let key = key_from_index(start_key + i as u64);
        let value = value_from_index(start_key + i as u64);
        let t0 = rdtsc();
        let result = kv.put(storage, key, value);
        let t1 = rdtsc();
        samples[i] = t1.wrapping_sub(t0);
        if result.is_err() {
            // Truncate samples at the first failure; the failed op's time
            // is not representative of a successful put.
            return (compute_stats(&mut samples[..i]), written);
        }
        written += 1;
    }
    (compute_stats(&mut samples[..cap]), written)
}

/// xorshift64: pure pseudo-random sequence with period `2^64 - 1`, cheap
/// enough that its overhead sits well below any tree-descent cost. Used in
/// place of a linear stride so the probe index distribution is not biased
/// by `gcd(stride, population)` for unlucky populations.
#[inline(always)]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Run `iters` pseudo-random gets against `population` keys starting at
/// `start_key`. Each probe index is drawn from a xorshift64 stream modulo
/// `population`, giving approximately uniform coverage even when the
/// population is small or shares factors with any fixed stride.
fn bench_gets(
    kv: &Kv,
    start_key: u64,
    population: u64,
    iters: usize,
    samples: &mut [u64],
) -> BenchStats {
    let cap = core::cmp::min(iters, samples.len());
    if population == 0 {
        return BenchStats::EMPTY;
    }
    let mut rng = 0x9E37_79B9_7F4A_7C15u64; // nonzero seed; xorshift needs nonzero state
    for i in 0..cap {
        let idx = xorshift64(&mut rng) % population;
        let key = key_from_index(start_key + idx);
        let t0 = rdtsc();
        let _got = kv.get(key);
        let t1 = rdtsc();
        samples[i] = t1.wrapping_sub(t0);
    }
    compute_stats(&mut samples[..cap])
}

pub fn run_bench<S: BlockStorage>(storage: &mut S) {
    let mut kv = Kv::recover(storage).expect("bench: Kv::recover failed");
    // Existing population from prior boots (gate test writes key=lsn as
    // big-endian u64, starting at lsn=1). bench_puts extends the same
    // contiguous range, so the get phase can probe every key in [1, end).
    let existing = kv.next_lsn().saturating_sub(1);
    let new_key_base = kv.next_lsn();
    serial_println!(
        "BENCH start: next_lsn={}, existing={}, put_count={}, get_iters={}",
        new_key_base,
        existing,
        PUT_COUNT,
        GET_ITERS,
    );

    // Put latency samples. 128 * 8 = 1 KB on the stack.
    let mut put_samples = [0u64; PUT_COUNT];
    let (put_stats, written) =
        bench_puts(&mut kv, storage, new_key_base, PUT_COUNT, &mut put_samples);
    serial_println!("BENCH put {} written={}", put_stats, written);

    // Get latency samples. 10_000 * 8 = 80 KB on the stack; still well
    // below the bootloader's default stack of several MB.
    let mut get_samples = [0u64; GET_ITERS];
    let total_population = existing + written;
    // Probe the whole populated range [1, 1 + total_population). With a
    // near-full pool and `written == 0`, this still produces meaningful
    // get numbers against the recovered keys.
    let get_stats = bench_gets(&kv, 1, total_population, GET_ITERS, &mut get_samples);
    serial_println!("BENCH get {}", get_stats);

    serial_println!("BENCH done");
}
