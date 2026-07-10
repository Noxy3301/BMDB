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
    // Under KVM (`-cpu host`) the hardware CPUID.15H usually reports 0, but the
    // hypervisor exposes a paravirt leaf 0x4000_0010 whose EAX is the virtual
    // TSC frequency in kHz. Prefer it so KVM runs on the dev host still convert
    // cycles to seconds. (The `hypervisor` bit, CPUID.1:ECX[31], is set under
    // any VMM; the 0x4000_0000 leaf's EAX is the max hypervisor leaf.)
    let hv_max = __cpuid(0x4000_0000).eax;
    if hv_max >= 0x4000_0010 {
        let l = __cpuid(0x4000_0010);
        if l.eax != 0 {
            return (l.eax as u64) * 1_000;
        }
    }

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

/// Diagnostic: raw CPUID values that decide TSC-frequency detection, so a run
/// can show exactly what the (possibly virtualized) guest exposes.
pub struct CpuidDiag {
    pub hypervisor: bool,
    pub hv_max_leaf: u32,
    pub hv_sig: [u32; 3],
    pub hv_tsc_khz: u32,
    pub invariant_tsc: bool,
    pub leaf15: [u32; 3],
    pub leaf16_mhz: u32,
}

pub fn cpuid_diag() -> CpuidDiag {
    let f1 = __cpuid(1);
    let hv = __cpuid(0x4000_0000);
    let hv10 = if hv.eax >= 0x4000_0010 {
        __cpuid(0x4000_0010).eax
    } else {
        0
    };
    let max = __cpuid(0).eax;
    let l15 = if max >= 0x15 {
        let c = __cpuid(0x15);
        [c.eax, c.ebx, c.ecx]
    } else {
        [0, 0, 0]
    };
    let l16 = if max >= 0x16 { __cpuid(0x16).eax } else { 0 };
    CpuidDiag {
        hypervisor: (f1.ecx & (1 << 31)) != 0,
        hv_max_leaf: hv.eax,
        hv_sig: [hv.ebx, hv.ecx, hv.edx],
        hv_tsc_khz: hv10,
        invariant_tsc: has_invariant_tsc(),
        leaf15: l15,
        leaf16_mhz: l16,
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
