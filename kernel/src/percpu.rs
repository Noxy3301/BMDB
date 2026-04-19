//! Per-CPU storage indexed by bring-up order, accessed through the GS
//! base. Mirrors Atmosphere's `kernel/src/cpu.rs`: one static slot per
//! logical CPU, `self_ptr` at offset zero so `gs:[0]` recovers a live
//! `&mut PerCpu` without a table lookup.
//!
//! `IA32_GS_BASE` (MSR 0xC0000101) is written once per CPU from Rust —
//! by the BSP in `kernel_main`, and by each AP as the first statement
//! inside `ap_main`. Until that write lands, no code on that CPU may
//! read per-CPU state, which is already true of the trampoline because
//! nothing between SIPI and `ap_main` touches `gs:`.

use core::cell::UnsafeCell;
use core::ptr;
use core::sync::atomic::AtomicU64;

use crate::acpi::MAX_CPUS;

/// Layout the AP trampoline and `gs:[0]` accessor both rely on:
/// `self_ptr` must live at offset 0 of every `PerCpu`.
#[repr(C, align(64))]
pub struct PerCpu {
    /// Self-reference. `mov reg, gs:[0]` recovers `&PerCpu` — matches
    /// the trick Atmosphere and Linux use to get O(1) access to the
    /// current CPU's slot without a table lookup.
    self_ptr: *mut PerCpu,
    /// Bring-up-order index. BSP is always 0; APs get 1..=N in the
    /// order they come online. Matches the `AP_STACKS` index used by
    /// `smp::init`.
    pub cpu_index: u32,
    /// Reserved for the Silo OCC TID generator. Initialized to 0; each
    /// worker increments its own counter, so concurrent access stays
    /// strictly local to the owning CPU.
    pub tid_counter: AtomicU64,
}

impl PerCpu {
    const NEW: PerCpu = PerCpu {
        self_ptr: ptr::null_mut(),
        cpu_index: 0,
        tid_counter: AtomicU64::new(0),
    };
}

/// Static pool — one slot per possible CPU. `UnsafeCell` because we
/// hand out aliasing `*mut PerCpu` pointers at init time (self_ptr
/// setup) and `gs:[0]` expects the same address afterward.
#[repr(C, align(64))]
struct PerCpuSlot(UnsafeCell<PerCpu>);

// Safety: the slots are only accessed through per-CPU local views
// after `init` publishes `self_ptr`. Cross-CPU access must go through
// the lock-free atomics inside `PerCpu`, never through the raw cell.
unsafe impl Sync for PerCpuSlot {}

static PERCPU: [PerCpuSlot; MAX_CPUS] = {
    const EMPTY: PerCpuSlot = PerCpuSlot(UnsafeCell::new(PerCpu::NEW));
    [EMPTY; MAX_CPUS]
};

/// MSR number for `IA32_GS_BASE`. Writing this MSR directly updates
/// the hidden GS base register used by `mov reg, gs:[imm]`, so it is
/// the safest primitive across vendors and modes.
const IA32_GS_BASE: u32 = 0xC000_0101;

/// Install the per-CPU slot for `cpu_index` and point `GS` base at it.
///
/// # Safety
/// - `cpu_index < MAX_CPUS` — caller's obligation.
/// - Must be called once per logical CPU, before anything on that CPU
///   reads `gs:[…]`.
pub unsafe fn init(cpu_index: usize) {
    assert!(cpu_index < MAX_CPUS);
    let slot = PERCPU[cpu_index].0.get();
    unsafe {
        (*slot).self_ptr = slot;
        (*slot).cpu_index = cpu_index as u32;
    }
    let base = slot as u64;
    let low = base as u32;
    let high = (base >> 32) as u32;
    unsafe {
        core::arch::asm!(
            "wrmsr",
            in("ecx") IA32_GS_BASE,
            in("eax") low,
            in("edx") high,
            options(nostack, preserves_flags),
        );
    }
}

/// Return a reference to the current CPU's `PerCpu`. Relies on
/// [`init`] having populated `self_ptr` at offset 0 of the slot GS
/// is pointing at.
///
/// # Safety
/// Caller must guarantee [`init`] ran on this CPU. Cross-thread
/// aliasing is the caller's problem — mutable fields should be wrapped
/// in atomics or per-CPU synchronization.
#[inline]
#[allow(dead_code)] // exercised once Silo wires up thread-local state
pub unsafe fn current() -> &'static mut PerCpu {
    let p: *mut PerCpu;
    unsafe {
        core::arch::asm!(
            "mov {out}, gs:[0]",
            out = out(reg) p,
            options(nostack, readonly, preserves_flags),
        );
        &mut *p
    }
}
