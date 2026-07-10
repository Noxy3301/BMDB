//! Per-core hardware performance counters (Intel architectural PMU) via RDPMC.
//!
//! On bare metal we are always ring 0, so RDPMC is free -- no syscall, no
//! vmexit -- which makes it the right way to attribute a scaling wall. The
//! headline event is XSNP_HITM: a load that hit a line another core held
//! Modified, i.e. a cross-core cache-line hand-off (contention / false
//! sharing). HITM-per-op rising with core count is exactly the "throughput
//! flattened, coherence-bound" signature the scaling study is hunting; a flat
//! HITM/op with linear throughput means the index genuinely scales.
//!
//! QEMU/TCG returns garbage (or zero) from RDPMC and does not model
//! coherence, so these numbers are only meaningful on the real M920q; the
//! event codes below are Skylake/Coffee-Lake family and should be confirmed
//! against the exact SKU on first boot.

use core::arch::asm;

use x86_64::registers::model_specific::Msr;

// Architectural PMU MSRs (Intel SDM Vol. 3, ch. 20/21).
const IA32_PMC0: u32 = 0x0C1;
const IA32_PERFEVTSEL0: u32 = 0x186;
const IA32_FIXED_CTR0: u32 = 0x309;
const IA32_FIXED_CTR_CTRL: u32 = 0x38D;
const IA32_PERF_GLOBAL_CTRL: u32 = 0x38F;

// PERFEVTSEL layout: event | umask<<8 | USR(16) | OS(17) | EDGE(18) | EN(22)
// | CMASK<<24.
const EVT_USR: u64 = 1 << 16;
const EVT_OS: u64 = 1 << 17;
const EVT_EDGE: u64 = 1 << 18;
const EVT_EN: u64 = 1 << 22;

/// Base PERFEVTSEL value counting `event`/`umask` in both ring 0 and ring 3.
const fn evtsel(event: u8, umask: u8) -> u64 {
    (event as u64) | ((umask as u64) << 8) | EVT_USR | EVT_OS | EVT_EN
}

/// The four general-purpose events, in RDPMC index order, as full PERFEVTSEL
/// values (Skylake/Coffee-Lake perfmon):
///   GP0 XSNP_HITM: MEM_LOAD_L3_HIT_RETIRED.XSNP_HITM  (0xD2 / 0x04)
///   GP1 L3_MISS:   MEM_LOAD_RETIRED.L3_MISS            (0xD1 / 0x20)
///   GP2 L2_MISS:   MEM_LOAD_RETIRED.L2_MISS            (0xD1 / 0x10)
///   GP3 MACHINE_CLEARS.COUNT (0xC3 / 0x01) -- a COUNT event, so it needs
///       EdgeDetect + CounterMask=1 to count rising edges, not busy cycles.
const GP_EVENTS: [u64; 4] = [
    evtsel(0xD2, 0x04),
    evtsel(0xD1, 0x20),
    evtsel(0xD1, 0x10),
    evtsel(0xC3, 0x01) | EVT_EDGE | (1 << 24),
];

fn wrmsr(addr: u32, val: u64) {
    // Safety: PMU MSRs; ring 0. Writing a disabled/zeroed PMU is harmless.
    unsafe { Msr::new(addr).write(val) };
}

/// True if the CPU exposes an architectural PMU (CPUID.0AH version >= 1).
pub fn available() -> bool {
    let max = core::arch::x86_64::__cpuid(0).eax;
    if max < 0x0A {
        return false;
    }
    (core::arch::x86_64::__cpuid(0x0A).eax & 0xFF) >= 1
}

/// Program this core's PMU: 4 GP events + 3 fixed counters, all counting
/// OS+USR, then globally enable. Each core must call this for its own
/// counters before a timed window.
pub fn program() {
    // Freeze everything while (re)programming.
    wrmsr(IA32_PERF_GLOBAL_CTRL, 0);
    for (i, &sel) in GP_EVENTS.iter().enumerate() {
        wrmsr(IA32_PERFEVTSEL0 + i as u32, sel);
        wrmsr(IA32_PMC0 + i as u32, 0);
    }
    // Fixed 0 = INST_RETIRED.ANY, 1 = CPU_CLK_UNHALTED.CORE, 2 = .REF.
    // FIXED_CTR_CTRL packs 4 bits per counter: bit0 OS, bit1 USR. 0x3 each.
    wrmsr(IA32_FIXED_CTR_CTRL, 0x333);
    for i in 0..3u32 {
        wrmsr(IA32_FIXED_CTR0 + i, 0);
    }
    // Enable GP0..3 (bits 0..3) and fixed 0..2 (bits 32..34).
    wrmsr(IA32_PERF_GLOBAL_CTRL, 0xF | (0x7u64 << 32));
}

#[inline(always)]
fn rdpmc(idx: u32) -> u64 {
    let lo: u32;
    let hi: u32;
    // Safety: ring 0, counters programmed by program(). RDPMC of an unenabled
    // index returns 0/garbage but does not fault at CPL 0.
    unsafe {
        asm!(
            "rdpmc",
            in("ecx") idx,
            out("eax") lo,
            out("edx") hi,
            options(nomem, nostack, preserves_flags),
        );
    }
    ((hi as u64) << 32) | (lo as u64)
}

/// A snapshot of all counters at one instant.
#[derive(Clone, Copy, Default)]
pub struct Counters {
    pub gp: [u64; 4],
    pub inst: u64,
    pub core_cycles: u64,
}

impl Counters {
    /// All-zero, usable in `const` (static array) initializers.
    pub const ZERO: Counters = Counters {
        gp: [0; 4],
        inst: 0,
        core_cycles: 0,
    };
}

/// Fixed-counter RDPMC indices carry bit 30 set.
const RDPMC_FIXED: u32 = 1 << 30;

/// Read all counters now (GP by index, fixed by index | 1<<30).
pub fn read() -> Counters {
    let mut c = Counters::default();
    for (i, slot) in c.gp.iter_mut().enumerate() {
        *slot = rdpmc(i as u32);
    }
    c.inst = rdpmc(RDPMC_FIXED);
    c.core_cycles = rdpmc(RDPMC_FIXED | 1);
    c
}

impl Counters {
    /// Counter deltas over a window (`self` = end, `start` = begin).
    pub fn delta(&self, start: &Counters) -> Counters {
        let mut d = Counters::default();
        for i in 0..4 {
            d.gp[i] = self.gp[i].wrapping_sub(start.gp[i]);
        }
        d.inst = self.inst.wrapping_sub(start.inst);
        d.core_cycles = self.core_cycles.wrapping_sub(start.core_cycles);
        d
    }

    /// Cycles-per-instruction ×100 (integer, no float in no_std). 0 if no
    /// instructions retired.
    pub fn cpi_x100(&self) -> u64 {
        if self.inst == 0 {
            0
        } else {
            self.core_cycles.saturating_mul(100) / self.inst
        }
    }

    /// Events per 1000 ops for GP counter `i` (the per-op contention rate).
    pub fn per_kop(&self, i: usize, ops: u64) -> u64 {
        if ops == 0 {
            0
        } else {
            self.gp[i].saturating_mul(1000) / ops
        }
    }
}
