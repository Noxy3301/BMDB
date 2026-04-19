//! SMP-d.1: validate AP wake-up with a minimal real-mode trampoline.
//!
//! Structure follows the small-kernel pattern shared by Atmosphere and
//! xv6: one assembly blob + one orchestration file, reusing the LAPIC
//! driver for IPI send. The trampoline here is deliberately *not* a
//! mode-transition stub — it only proves that a SIPI reaches the AP and
//! that the AP executes bytes we copied. Full long-mode transition and
//! Rust AP entry are added in SMP-d.2.
//!
//! AP sequence the trampoline performs:
//!   1. disable interrupts
//!   2. zero DS so `mov [0xA000], al` addresses linear `0xA000`
//!   3. write the marker byte `0xAB` at linear `0xA000`
//!   4. halt forever
//!
//! BSP sequence per enabled AP (Intel SDM Vol 3A §11.4.4 MP-init):
//!   1. zero the marker cell
//!   2. INIT IPI
//!   3. wait >= 10 ms
//!   4. Startup IPI #1 with vector `0x08`
//!   5. wait >= 200 µs
//!   6. Startup IPI #2 (canonical, covers hardware that misses SIPI #1)
//!   7. wait >= 200 µs
//!   8. read marker; `0xAB` means the AP ran our code
//!
//! TSC-based waits are coarse. Without a calibrated TSC frequency the
//! cycle constants are sized against the fastest realistic CPU clock
//! (~5 GHz) so they exceed the protocol minimum on any current
//! hardware. Calibration via CPUID leaf 0x15 or a PIT reference is
//! tech debt for SMP-d.2.

use core::arch::global_asm;
use core::ptr;

use bmdb_serial::serial_println;

use crate::acpi::{ACPI, CpuInfo, MAX_CPUS};
use crate::apic::LAPIC;

/// Physical destination for the AP trampoline. 16-byte aligned and below
/// 1 MiB so it is addressable in real mode. SIPI vector `0x08` selects
/// this page (`vector << 12 == 0x8000`).
///
/// Chosen by convention: BIOS-era low memory from `0x7C00..0x10_0000` is
/// free once the bootloader has handed off, and `0x8000` / `0xA000` sit
/// in the classic "conventional memory" region that every SMP trampoline
/// (Linux, xv6, Redox) historically targets. A proper memory-map check
/// against `BootInfo::memory_map` is deferred to SMP-d.2, when per-AP
/// stack allocation will need the same infrastructure.
const AP_TRAMPOLINE_PHYS: u64 = 0x8000;

/// Physical address the trampoline writes a marker byte to, for the BSP
/// to poll. Clear of the trampoline page itself and well within
/// real-mode reach. See AP_TRAMPOLINE_PHYS for the memory-map caveat.
const AP_MARKER_PHYS: u64 = 0xA000;
const AP_MARKER_VALUE: u8 = 0xAB;

/// Intel SDM Vol 3A §11.4.4 requires at least 10 ms between INIT and
/// Startup IPI. The wait is implemented as a TSC cycle count; without
/// a calibrated TSC frequency the constant has to be set against the
/// fastest modern CPU clock (~5 GHz) with margin. 200 million cycles is
/// 40 ms even at 5 GHz and 200 ms at 1 GHz — comfortably above the
/// protocol minimum on every realistic TSC rate. Calibration via CPUID
/// leaf 0x15 or a PIT reference is deferred to SMP-d.2.
const INIT_TO_SIPI_CYCLES: u64 = 200_000_000;

/// Per-SIPI follow-up. SDM recommends 200 µs before probing the AP or
/// issuing the second SIPI; 10 million cycles is 2 ms at 5 GHz and
/// 10 ms at 1 GHz, plenty of margin over the 200 µs floor.
const POST_SIPI_CYCLES: u64 = 10_000_000;

unsafe extern "C" {
    static ap_trampoline_start: u8;
    static ap_trampoline_end: u8;
}

// The trampoline lives in its own .text.ap_trampoline section so it is
// never executed from its kernel-virtual load address — the BSP copies
// it to phys 0x8000 and APs run it there. `.code16` tells the assembler
// to emit 16-bit opcodes; we restore `.code64` before closing the block
// so nothing downstream inherits the directive.
global_asm!(
    r#"
    .pushsection .text.ap_trampoline, "ax"
    .code16
    .global ap_trampoline_start
    .global ap_trampoline_end

ap_trampoline_start:
    cli
    xor ax, ax
    mov ds, ax
    mov al, 0xAB
    mov [0xA000], al
2:
    hlt
    jmp 2b
ap_trampoline_end:

    .code64
    .popsection
    "#
);

fn rdtsc() -> u64 {
    // Safety: `_rdtsc` is an unconditionally available intrinsic on
    // x86_64; no MSR or privilege gate needed in kernel mode.
    unsafe { core::arch::x86_64::_rdtsc() }
}

fn busy_wait_cycles(cycles: u64) {
    let start = rdtsc();
    while rdtsc().wrapping_sub(start) < cycles {
        core::hint::spin_loop();
    }
}

unsafe fn trampoline_bytes() -> &'static [u8] {
    // Take addresses via `&symbol`; the u8 type is only there to name
    // a well-defined ABI for `extern static`.
    let start = unsafe { &ap_trampoline_start as *const u8 };
    let end = unsafe { &ap_trampoline_end as *const u8 };
    let len = end as usize - start as usize;
    unsafe { core::slice::from_raw_parts(start, len) }
}

unsafe fn copy_trampoline(phys_mem_offset: u64) {
    let bytes = unsafe { trampoline_bytes() };
    let dest = (phys_mem_offset + AP_TRAMPOLINE_PHYS) as *mut u8;
    // Safety: the destination page is within bootloader-mapped physical
    // memory and is unreserved at this point in boot; the trampoline is
    // small (single-digit bytes in this MVP).
    unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), dest, bytes.len()) };
}

unsafe fn write_marker(phys_mem_offset: u64, value: u8) {
    let ptr = (phys_mem_offset + AP_MARKER_PHYS) as *mut u8;
    unsafe { ptr::write_volatile(ptr, value) };
}

unsafe fn read_marker(phys_mem_offset: u64) -> u8 {
    let ptr = (phys_mem_offset + AP_MARKER_PHYS) as *const u8;
    unsafe { ptr::read_volatile(ptr) }
}

/// Copy the trampoline, then probe each enabled non-BSP AP with
/// INIT → wait → SIPI → wait → marker read. Prints a per-AP result
/// line so the QEMU smoke test is grep-friendly.
///
/// # Safety
/// Requires the bootloader's `map_physical_memory` mapping at
/// `phys_mem_offset` and must be called exactly once, after
/// [`apic::init`] and [`acpi::init`] have populated their statics.
pub unsafe fn init(phys_mem_offset: u64) {
    unsafe { copy_trampoline(phys_mem_offset) };
    let tramp_len = unsafe { trampoline_bytes() }.len();
    serial_println!(
        "SMP: trampoline copied to 0x{:x} ({} bytes)",
        AP_TRAMPOLINE_PHYS,
        tramp_len,
    );

    // Snapshot the AP list so we do not hold the ACPI lock across the
    // per-AP busy waits.
    let (cpus, cpu_count) = {
        let guard = ACPI.lock();
        let acpi = match guard.as_ref() {
            Some(a) => a,
            None => {
                serial_println!("SMP: no ACPI info, cannot enumerate APs");
                return;
            }
        };
        let list = acpi.cpus();
        let count = list.len().min(MAX_CPUS);
        let mut snap = [CpuInfo::default(); MAX_CPUS];
        snap[..count].copy_from_slice(&list[..count]);
        (snap, count)
    };

    let bsp_apic_id = match LAPIC.lock().as_ref() {
        Some(l) => l.id() as u8,
        None => {
            serial_println!("SMP: LAPIC not initialized, cannot wake APs");
            return;
        }
    };

    let mut woken = 0usize;
    for cpu in &cpus[..cpu_count] {
        if cpu.apic_id == bsp_apic_id || !cpu.enabled {
            continue;
        }

        // Zero the marker cell so a stale `0xAB` from a prior iteration
        // cannot be mistaken for this AP's own write.
        unsafe { write_marker(phys_mem_offset, 0) };

        {
            let mut g = LAPIC.lock();
            let lapic = g.as_mut().expect("LAPIC present");
            lapic.send_init_ipi(cpu.apic_id);
        }
        busy_wait_cycles(INIT_TO_SIPI_CYCLES);

        // SIPI #1. Vector byte is the high byte of the phys start
        // address (`0x08 << 12 == 0x8000`).
        {
            let mut g = LAPIC.lock();
            let lapic = g.as_mut().expect("LAPIC present");
            lapic.send_startup_ipi(cpu.apic_id, 0x08);
        }
        busy_wait_cycles(POST_SIPI_CYCLES);

        // SIPI #2. Intel's MP-init sequence sends two SIPIs
        // unconditionally; the second covers hardware that silently
        // drops the first. A no-op on an AP that already latched #1.
        {
            let mut g = LAPIC.lock();
            let lapic = g.as_mut().expect("LAPIC present");
            lapic.send_startup_ipi(cpu.apic_id, 0x08);
        }
        busy_wait_cycles(POST_SIPI_CYCLES);

        let marker = unsafe { read_marker(phys_mem_offset) };
        if marker == AP_MARKER_VALUE {
            serial_println!(
                "SMP: AP apic_id={} woke (marker=0x{:02x})",
                cpu.apic_id,
                marker,
            );
            woken += 1;
        } else {
            serial_println!(
                "SMP: AP apic_id={} did not wake (marker=0x{:02x})",
                cpu.apic_id,
                marker,
            );
        }
    }

    serial_println!("SMP: {} AP(s) woken", woken);
}
