//! Fenced, calibrated TSC timing for bare-metal benchmarks.
//!
//! The old benches read a bare `_rdtsc()` and reported "TPS at 1 GHz" -- both
//! wrong for a scaling study. RDTSC is not serializing, so out-of-order
//! execution smears sub-microsecond samples, and the invariant TSC counts a
//! fixed *reference* frequency, not core cycles, so "cycles / 1 GHz" is
//! fiction. This module gives the two things a trustworthy measurement needs:
//! a fenced read (so a timed region's boundaries are where they look), and
//! the real TSC frequency (so cycles convert to seconds).
//!
//! QEMU/TCG synthesizes the TSC and its CPUID leaves, so these numbers are
//! only meaningful on real hardware; on the M920q the invariant TSC and
//! CPUID.15H are present (Coffee Lake).

use core::arch::x86_64::{__cpuid, __rdtscp, _mm_lfence, _rdtsc};

/// Invariant TSC: constant rate across P/C-states, so a delta is a valid wall
/// clock. CPUID.80000007H:EDX[8]. Without it, RDTSC deltas are not a clock.
pub fn has_invariant_tsc() -> bool {
    let max_ext = __cpuid(0x8000_0000).eax;
    if max_ext < 0x8000_0007 {
        return false;
    }
    (__cpuid(0x8000_0007).edx & (1 << 8)) != 0
}

/// TSC frequency in Hz. Prefers CPUID.15H (crystal clock * TSC/crystal ratio);
/// falls back to CPUID.16H (base MHz). Returns 0 when neither is usable, in
/// which case the caller must treat cycle counts as self-relative only.
pub fn tsc_hz() -> u64 {
    let max_leaf = __cpuid(0).eax;

    // CPUID.15H: EAX = ratio denominator, EBX = ratio numerator,
    // ECX = crystal Hz (0 on some parts -> use the known crystal).
    if max_leaf >= 0x15 {
        let l = __cpuid(0x15);
        if l.eax != 0 && l.ebx != 0 {
            let crystal = if l.ecx != 0 {
                l.ecx as u64
            } else {
                // Coffee Lake / Kaby Lake client crystal is 24 MHz when the
                // leaf reports 0. (Server SKUs differ; recheck per part.)
                24_000_000
            };
            return crystal * (l.ebx as u64) / (l.eax as u64);
        }
    }

    // CPUID.16H: EAX = processor base frequency in MHz.
    if max_leaf >= 0x16 {
        let l = __cpuid(0x16);
        if l.eax != 0 {
            return (l.eax as u64) * 1_000_000;
        }
    }

    0
}

/// Timestamp at the START of a timed region: LFENCE drains earlier
/// instructions so none are counted inside, RDTSC takes the stamp, and a
/// second LFENCE keeps the region's first instructions from starting before
/// the stamp is read (fully closing the start edge).
#[inline(always)]
pub fn start() -> u64 {
    // SSE2 baseline on x86_64, so lfence/rdtsc are always present.
    unsafe {
        _mm_lfence();
        let t = _rdtsc();
        _mm_lfence();
        t
    }
}

/// Timestamp at the END of a timed region: RDTSCP waits for earlier
/// instructions to retire (so all the work is counted), then LFENCE stops
/// later instructions from being pulled in before the timestamp is taken.
#[inline(always)]
pub fn end() -> u64 {
    unsafe {
        let mut aux = 0u32;
        let t = __rdtscp(&mut aux);
        _mm_lfence();
        t
    }
}

/// Ops/second given `ops` completed over a `cycles` window at `hz`.
pub fn ops_per_sec(ops: u64, cycles: u64, hz: u64) -> u64 {
    if hz == 0 || cycles == 0 {
        0
    } else {
        (ops as u128 * hz as u128 / cycles as u128) as u64
    }
}
