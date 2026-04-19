//! Benchmark summary statistics.
//!
//! The caller collects per-operation latency samples into a `u64` slice
//! (TSC cycles, nanoseconds — the module does not interpret the unit) and
//! passes it to [`compute_stats`]. The buffer is sorted in place so the
//! caller can reuse it without reallocating.
//!
//! Percentiles use the canonical ceiling-based nearest-rank method
//! (NIST / Wikipedia): `rank_1based = ceil(n * k / 1000)`, then map to a
//! zero-based index. For small `n`, high-percentile ranks collapse toward
//! `max` (not `min`), which is the intuitive behavior for bench summaries.

use core::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BenchStats {
    pub n: u64,
    pub min: u64,
    pub max: u64,
    pub mean: u64,
    pub p50: u64,
    pub p90: u64,
    pub p99: u64,
    pub p999: u64,
}

impl BenchStats {
    pub const EMPTY: BenchStats = BenchStats {
        n: 0,
        min: 0,
        max: 0,
        mean: 0,
        p50: 0,
        p90: 0,
        p99: 0,
        p999: 0,
    };
}

/// Compute summary stats over latency samples. Sorts `samples` in place.
pub fn compute_stats(samples: &mut [u64]) -> BenchStats {
    let n = samples.len();
    if n == 0 {
        return BenchStats::EMPTY;
    }
    samples.sort_unstable();

    // Widen to u128 so a long bench over multi-billion-cycle samples cannot
    // overflow the accumulator.
    let sum: u128 = samples.iter().map(|&x| x as u128).sum();
    let mean = (sum / n as u128) as u64;

    let rank = |per_mille: u64| -> u64 {
        // Canonical ceiling nearest-rank: 1-based rank = ceil(n * pm / 1000),
        // then convert to 0-based. Integer-math ceil: (a + b - 1) / b.
        let rank_1based = ((n as u64) * per_mille + 999) / 1000;
        let idx = rank_1based.saturating_sub(1).min((n as u64) - 1);
        samples[idx as usize]
    };

    BenchStats {
        n: n as u64,
        min: samples[0],
        max: samples[n - 1],
        mean,
        p50: rank(500),
        p90: rank(900),
        p99: rank(990),
        p999: rank(999),
    }
}

impl fmt::Display for BenchStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "n={} min={} p50={} p90={} p99={} p999={} max={} mean={}",
            self.n, self.min, self.p50, self.p90, self.p99, self.p999, self.max, self.mean
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_samples_yield_empty_stats() {
        let mut samples: [u64; 0] = [];
        assert_eq!(compute_stats(&mut samples), BenchStats::EMPTY);
    }

    #[test]
    fn single_sample_collapses_all_fields() {
        let mut samples = [42u64];
        let s = compute_stats(&mut samples);
        assert_eq!(s.n, 1);
        assert_eq!(s.min, 42);
        assert_eq!(s.max, 42);
        assert_eq!(s.mean, 42);
        assert_eq!(s.p50, 42);
        assert_eq!(s.p99, 42);
        assert_eq!(s.p999, 42);
    }

    #[test]
    fn stats_over_one_to_hundred() {
        let mut samples: std::vec::Vec<u64> = (1..=100u64).collect();
        let s = compute_stats(&mut samples);
        assert_eq!(s.n, 100);
        assert_eq!(s.min, 1);
        assert_eq!(s.max, 100);
        // sum = 5050, mean = 50 (integer division of 50.5).
        assert_eq!(s.mean, 50);
        assert_eq!(s.p50, 50);
        assert_eq!(s.p90, 90);
        assert_eq!(s.p99, 99);
    }

    #[test]
    fn stats_over_one_to_thousand() {
        let mut samples: std::vec::Vec<u64> = (1..=1000u64).collect();
        let s = compute_stats(&mut samples);
        assert_eq!(s.n, 1000);
        assert_eq!(s.min, 1);
        assert_eq!(s.max, 1000);
        assert_eq!(s.p50, 500);
        assert_eq!(s.p90, 900);
        assert_eq!(s.p99, 990);
        assert_eq!(s.p999, 999);
    }

    #[test]
    fn compute_stats_sorts_buffer_in_place() {
        let mut samples = [3u64, 1, 4, 1, 5, 9, 2, 6, 5, 3];
        compute_stats(&mut samples);
        assert!(samples.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn mean_does_not_overflow_on_large_samples() {
        // Mean of [u64::MAX / 2; 4] is u64::MAX / 2, but the sum is
        // (u64::MAX / 2) * 4 which would wrap a u64 accumulator.
        let big = u64::MAX / 2;
        let mut samples = [big; 4];
        let s = compute_stats(&mut samples);
        assert_eq!(s.mean, big);
        assert_eq!(s.max, big);
    }

    #[test]
    fn stats_on_two_samples_split_min_and_max() {
        // With n=2, rank(500) = 1*500/1000 = 0 → samples[0] = min, and
        // every higher percentile lands on samples[1] = max.
        let mut samples = [10u64, 20];
        let s = compute_stats(&mut samples);
        assert_eq!(s.n, 2);
        assert_eq!(s.min, 10);
        assert_eq!(s.max, 20);
        assert_eq!(s.p50, 10);
        assert_eq!(s.p90, 20);
        assert_eq!(s.p99, 20);
        assert_eq!(s.p999, 20);
    }

    #[test]
    fn stats_on_three_samples_percentile_buckets() {
        // rank(500) = 2*500/1000 = 1 → median at index 1.
        let mut samples = [5u64, 15, 25];
        let s = compute_stats(&mut samples);
        assert_eq!(s.p50, 15);
        assert_eq!(s.p90, 25);
        assert_eq!(s.p99, 25);
    }

    #[test]
    fn stats_on_all_equal_samples_collapse_every_percentile() {
        let mut samples = [7u64; 50];
        let s = compute_stats(&mut samples);
        assert_eq!(s.min, 7);
        assert_eq!(s.max, 7);
        assert_eq!(s.mean, 7);
        assert_eq!(s.p50, 7);
        assert_eq!(s.p99, 7);
        assert_eq!(s.p999, 7);
    }

    #[test]
    fn stats_respect_duplicates_in_mixed_data() {
        // Five 1s, five 2s, five 3s → p50 should land in the 2s.
        let mut samples = [1u64, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3];
        let s = compute_stats(&mut samples);
        assert_eq!(s.min, 1);
        assert_eq!(s.max, 3);
        // rank(500) = 14*500/1000 = 7 → samples[7] = 2 after sort.
        assert_eq!(s.p50, 2);
    }

    #[test]
    fn display_format_is_single_line_and_includes_all_fields() {
        let stats = BenchStats {
            n: 10,
            min: 100,
            max: 900,
            mean: 500,
            p50: 450,
            p90: 800,
            p99: 850,
            p999: 890,
        };
        let s = std::format!("{}", stats);
        assert!(!s.contains('\n'));
        for needle in ["n=10", "min=100", "p50=450", "p99=850", "p999=890", "max=900", "mean=500"] {
            assert!(s.contains(needle), "missing {} in {:?}", needle, s);
        }
    }
}
