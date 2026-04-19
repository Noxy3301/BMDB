//! SMP-d.2: wake APs into long mode and land them in a Rust entry.
//!
//! Structure follows the Redox pattern (chosen over Linux's separate
//! `trampoline_pgd` because it stays inside ~100 lines of orchestration):
//!
//!   * BSP injects an identity mapping for the trampoline's physical
//!     page into its own PML4 before sending INIT/SIPI, so the AP can
//!     keep fetching instructions at `phys 0x8000` after `CR0.PG` is
//!     set. After transition to long mode the AP executes through the
//!     same shared PML4, so the kernel's high-VA mapping is reachable
//!     for `ap_main` and per-AP stacks without switching CR3.
//!
//!   * Trampoline assembly performs the canonical 16 → 32 → 64 bit
//!     mode transition, loads a per-AP stack + entry pointer out of a
//!     shared trampoline header, and tail-jumps into Rust.
//!
//!   * Per-AP 16 KiB stacks are preallocated from a static pool indexed
//!     by AP bring-up order (not APIC ID) so the pool is tightly sized.
//!
//!   * Rust [`ap_main`] increments a shared `ONLINE_APS` atomic and
//!     halts. The BSP polls that counter per AP with a bounded TSC
//!     busy-wait, matching the `SMP-d.1` timing shape.
//!
//! Assumptions inherited from SMP-d.1 (documented scope-defer): phys
//! `0x8000`, `0xB000`, `0xC000` are free. Proper reservation against
//! `BootInfo::memory_map` plus a real frame allocator come in a later
//! commit when per-AP stacks move to dynamic allocation.

use core::arch::global_asm;
use core::ptr;
use core::sync::atomic::{AtomicU32, Ordering};

use bmdb_serial::serial_println;
use x86_64::registers::control::Cr3;

use crate::acpi::{ACPI, CpuInfo, MAX_CPUS};
use crate::apic::LAPIC;

// -- Physical layout ---------------------------------------------------

/// AP trampoline page. SIPI vector `0x08` selects this.
const AP_TRAMPOLINE_PHYS: u64 = 0x8000;

/// Scratch frame used when the existing `PML4[0] → L3 → L2` tree has
/// no `L3[0]` yet — the BSP must install an L2 before adding the
/// identity mapping. The L3 and everything above come from bootloader
/// 0.9's kernel mapping, so no L3 scratch is needed here. See module
/// doc comment for the memory-map scope-defer note.
const IDENTITY_L2_PHYS: u64 = 0xC000;

// -- AP stack pool -----------------------------------------------------

const AP_STACK_SIZE: usize = 16 * 1024;

#[repr(C, align(16))]
struct ApStack([u8; AP_STACK_SIZE]);

// One 16 KiB stack per possible AP. 64 × 16 KiB = 1 MiB static; sized
// against the same ceiling as `MAX_CPUS`. Indexed by bring-up order,
// not APIC ID.
static mut AP_STACKS: [ApStack; MAX_CPUS] = {
    const EMPTY: ApStack = ApStack([0; AP_STACK_SIZE]);
    [EMPTY; MAX_CPUS]
};

unsafe fn ap_stack_top(index: usize) -> u64 {
    // Safety: `index < MAX_CPUS` is enforced by the caller (the AP loop
    // cannot produce more bringups than MADT-enumerated CPUs).
    let stack_ptr = unsafe { ptr::addr_of_mut!(AP_STACKS[index]) };
    stack_ptr as u64 + AP_STACK_SIZE as u64
}

// -- Online counter ----------------------------------------------------

/// Incremented by every AP when it reaches `ap_main`. BSP polls this.
static ONLINE_APS: AtomicU32 = AtomicU32::new(0);

/// SMP-f contention smoke: every AP does a bounded fetch_add spin on
/// this counter before parking. The BSP then reads the total and checks
/// it against `N_AP * CONTENTION_ITERS`. Non-zero mismatch would mean a
/// lost update — which should be impossible given `fetch_add` is
/// atomic — so the test doubles as a quick regression gate for the
/// AP bring-up path touching memory correctly.
pub(crate) static CONTENTION_COUNTER: core::sync::atomic::AtomicU64 =
    core::sync::atomic::AtomicU64::new(0);
pub(crate) const CONTENTION_ITERS: u64 = 10_000;

// -- Timing ------------------------------------------------------------

/// INIT→SIPI delay. Sized against a 5 GHz TSC so the wait exceeds
/// Intel's 10 ms MP-init minimum at any realistic clock rate.
const INIT_TO_SIPI_CYCLES: u64 = 200_000_000;
/// Per-SIPI follow-up. Over the 200 µs minimum at any realistic clock.
const POST_SIPI_CYCLES: u64 = 10_000_000;
/// Upper bound on how long we wait for an AP to report online. A well
/// hardware should reach `ap_main` in single-digit milliseconds after
/// SIPI, so ~200 ms is orders of magnitude above expected.
const AP_ONLINE_TIMEOUT_CYCLES: u64 = 2_000_000_000;

fn rdtsc() -> u64 {
    unsafe { core::arch::x86_64::_rdtsc() }
}

fn busy_wait_cycles(cycles: u64) {
    let start = rdtsc();
    while rdtsc().wrapping_sub(start) < cycles {
        core::hint::spin_loop();
    }
}

// -- AP trampoline assembly -------------------------------------------
//
// The trampoline lives in `.text.ap_trampoline` so it is never executed
// from its kernel-VA load address; the BSP copies it to phys `0x8000`
// and APs run it there. `global_asm!` emits Intel syntax by default
// (the Rust 2021+ default), and `.code16` / `.code32` / `.code64`
// directives switch instruction encodings as the trampoline progresses
// through mode transitions.
//
// Header layout at the end of the trampoline is a 4 × u64 block the
// BSP populates before SIPI: CR3, stack top, entry pointer, AP id.

global_asm!(
    r#"
    .pushsection .text.ap_trampoline, "ax"
    .code16
    .global ap_trampoline_start
    .global ap_trampoline_end
    .global ap_header_cr3
    .global ap_header_stack
    .global ap_header_entry
    .global ap_header_arg

    # LLVM's integrated assembler forbids two symbols inside one memory
    # operand (e.g. `[0x8000 + (a - b)]`), so pre-compute the absolute
    # runtime addresses as named constants here. These resolve to plain
    # integers at assembly time because every symbol below lives in
    # this same section.
    .set AP_HDR_CR3_ADDR,   0x8000 + (ap_header_cr3   - ap_trampoline_start)
    .set AP_HDR_STACK_ADDR, 0x8000 + (ap_header_stack - ap_trampoline_start)
    .set AP_HDR_ENTRY_ADDR, 0x8000 + (ap_header_entry - ap_trampoline_start)
    .set AP_HDR_ARG_ADDR,   0x8000 + (ap_header_arg   - ap_trampoline_start)
    .set AP_32BIT_ADDR,     0x8000 + (ap_32bit_phys   - ap_trampoline_start)
    .set AP_64BIT_ADDR,     0x8000 + (ap_64bit_phys   - ap_trampoline_start)
    .set AP_GDT_ADDR,       0x8000 + (ap_gdt          - ap_trampoline_start)
    .set AP_GDT_DESC_OFF,   ap_gdt_desc - ap_trampoline_start

ap_trampoline_start:
    cli
    cld

    # Copy CS to DS so [offset] inside this segment hits our bytes.
    mov ax, cs
    mov ds, ax

    # Load GDT via a descriptor inside our segment.
    lgdt [AP_GDT_DESC_OFF]

    # Protected mode.
    mov eax, cr0
    or eax, 1
    mov cr0, eax

    # Far jump to 32-bit code, manually encoded because LLVM's Intel
    # syntax assembler rejects both `ljmp sel,off` and `jmp sel:off`
    # when the offset is a symbolic expression. Opcode:
    #   0x66 0xEA <offset32> <selector16>
    # The 0x66 operand-size prefix makes the offset 32-bit inside a
    # `.code16` block, matching flat protected-mode addressing.
    .byte 0x66, 0xEA
    .long AP_32BIT_ADDR
    .word 0x08

    .code32
ap_32bit_phys:
    # Load 32-bit flat data selectors.
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax

    # PAE enables the 4-level paging structure long mode requires.
    mov eax, cr4
    or eax, 1 << 5
    mov cr4, eax

    # CR3 points at the kernel PML4. Bootloader 0.9 places it below
    # 4 GiB so a 32-bit load covers it.
    mov eax, [AP_HDR_CR3_ADDR]
    mov cr3, eax

    # EFER: set LME (bit 8) to enable long mode, and NXE (bit 11) so
    # page-table entries with the NX bit are not treated as reserved
    # — bootloader 0.9 marks kernel data pages with NX, and without
    # NXE the AP would #PF on any such access even though the BSP
    # tolerates it with its own NXE=1.
    mov ecx, 0xC0000080
    rdmsr
    or eax, (1 << 8) | (1 << 11)
    wrmsr

    # Enable paging — long mode becomes active.
    mov eax, cr0
    or eax, 1 << 31
    mov cr0, eax

    # Far jump to 64-bit code. Opcode EA with a 32-bit offset + 16-bit
    # selector; LLVM's Intel-syntax assembler rejects both `ljmp sel,off`
    # and `jmp sel:off` when the offset is a symbolic expression, so the
    # bytes are emitted by hand.
    .byte 0xEA
    .long AP_64BIT_ADDR
    .word 0x18

    .code64
ap_64bit_phys:
    # Long mode mostly ignores data segment registers, but zero them
    # to avoid inheriting garbage from the AP's INIT state.
    xor rax, rax
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax

    # Per-AP stack is a kernel-VA pointer populated by the BSP in the
    # header. Since AP shares BSP's PML4 the mapping is live.
    mov rsp, [AP_HDR_STACK_ADDR]
    mov rbp, rsp

    # First System V ABI argument (APIC ID) into RDI.
    mov rdi, [AP_HDR_ARG_ADDR]

    # CALL (not JMP) so the System V AMD64 ABI's RSP alignment contract
    # holds at the entry of `ap_main`: CALL pushes an 8-byte return
    # slot, making `(RSP + 8) % 16 == 0`. `ap_main` never returns, so
    # the slot is ignored.
    mov rax, [AP_HDR_ENTRY_ADDR]
    call rax
    ud2

    .align 8
ap_gdt:
    .quad 0                       # null
    .quad 0x00CF9A000000FFFF      # 32-bit code: base=0 limit=4G D=1
    .quad 0x00CF92000000FFFF      # 32-bit data
    .quad 0x00AF9A000000FFFF      # 64-bit code: L=1
ap_gdt_end:

ap_gdt_desc:
    .word ap_gdt_end - ap_gdt - 1
    .long AP_GDT_ADDR

    .align 8
ap_header_cr3:
    .quad 0
ap_header_stack:
    .quad 0
ap_header_entry:
    .quad 0
ap_header_arg:
    .quad 0

ap_trampoline_end:

    .code64
    .popsection
    "#
);

unsafe extern "C" {
    static ap_trampoline_start: u8;
    static ap_trampoline_end: u8;
    static ap_header_cr3: u64;
    static ap_header_stack: u64;
    static ap_header_entry: u64;
    static ap_header_arg: u64;
}

unsafe fn trampoline_bytes() -> &'static [u8] {
    let start = unsafe { &ap_trampoline_start as *const u8 };
    let end = unsafe { &ap_trampoline_end as *const u8 };
    let len = end as usize - start as usize;
    unsafe { core::slice::from_raw_parts(start, len) }
}

/// Offset of a header field, relative to `ap_trampoline_start`. Symbols
/// in `global_asm!` resolve to kernel-virtual addresses; subtracting
/// `ap_trampoline_start` gives the offset we copied to `0x8000`.
unsafe fn header_offset(field: *const u64) -> u64 {
    let start = unsafe { &ap_trampoline_start as *const u8 as u64 };
    (field as u64) - start
}

unsafe fn copy_trampoline(phys_mem_offset: u64) {
    let bytes = unsafe { trampoline_bytes() };
    let dest = (phys_mem_offset + AP_TRAMPOLINE_PHYS) as *mut u8;
    unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), dest, bytes.len()) };
}

unsafe fn write_trampoline_header(
    phys_mem_offset: u64,
    cr3: u64,
    stack_top: u64,
    entry: u64,
    arg: u64,
) {
    let base = phys_mem_offset + AP_TRAMPOLINE_PHYS;
    let cr3_off = unsafe { header_offset(&ap_header_cr3) };
    let stack_off = unsafe { header_offset(&ap_header_stack) };
    let entry_off = unsafe { header_offset(&ap_header_entry) };
    let arg_off = unsafe { header_offset(&ap_header_arg) };

    unsafe {
        ptr::write_volatile((base + cr3_off) as *mut u64, cr3);
        ptr::write_volatile((base + stack_off) as *mut u64, stack_top);
        ptr::write_volatile((base + entry_off) as *mut u64, entry);
        ptr::write_volatile((base + arg_off) as *mut u64, arg);
    }
}

// -- Identity mapping injection ---------------------------------------

/// Install `virt 0x8000` → `phys 0x8000` into the kernel's active page
/// tree, creating the minimum number of intermediate entries needed to
/// reach the target. Every existing entry above the inserted 4 KiB PTE
/// is preserved — bootloader 0.9 already uses PML4[0] through L1 to
/// map the kernel image, so clobbering any level would unmap live
/// kernel state and triple-fault.
///
/// Returns `true` when the mapping is definitely in place (either just
/// installed or already present), `false` on any case we refuse to
/// touch. Callers should treat `false` as "APs must not be woken" —
/// the trampoline's paging-on jump would triple-fault.
///
/// # Safety
/// - `phys_mem_offset` must map all physical memory.
/// - The frame at `IDENTITY_L2_PHYS` must not be used by anyone else
///   (only claimed if the tree happens to need a new L1 at L2[0]).
///   See module-level scope-defer note.
/// - Must be called on the BSP exactly once before any AP is woken.
unsafe fn inject_identity_mapping(phys_mem_offset: u64) -> bool {
    const PRESENT: u64 = 1 << 0;
    const WRITABLE: u64 = 1 << 1;
    const HUGE: u64 = 1 << 7;
    const ADDR_MASK: u64 = 0x000F_FFFF_FFFF_F000;

    let target_phys = AP_TRAMPOLINE_PHYS;
    let target_virt = AP_TRAMPOLINE_PHYS;

    let pml4_idx = ((target_virt >> 39) & 0x1FF) as usize;
    let l3_idx = ((target_virt >> 30) & 0x1FF) as usize;
    let l2_idx = ((target_virt >> 21) & 0x1FF) as usize;
    let l1_idx = ((target_virt >> 12) & 0x1FF) as usize;

    let (cr3_frame, _) = Cr3::read();
    let pml4_phys = cr3_frame.start_address().as_u64();
    let pml4_ptr = (phys_mem_offset + pml4_phys) as *mut u64;
    let pml4_entry = unsafe { ptr::read_volatile(pml4_ptr.add(pml4_idx)) };

    if pml4_entry & PRESENT == 0 {
        serial_println!(
            "SMP: PML4[{}] absent — bootloader did not map the kernel through it?",
            pml4_idx
        );
        return false;
    }

    // L3.
    let l3_phys = pml4_entry & ADDR_MASK;
    let l3_ptr = (phys_mem_offset + l3_phys) as *mut u64;
    let l3_entry = unsafe { ptr::read_volatile(l3_ptr.add(l3_idx)) };
    if l3_entry & PRESENT == 0 {
        serial_println!("SMP: L3[{}] absent — need a fresh L2, not implemented", l3_idx);
        return false;
    }
    if l3_entry & HUGE != 0 {
        // 1 GiB huge page. Identity only if its base aligns with the
        // target's 1 GiB region.
        let base = l3_entry & ADDR_MASK;
        if base == target_phys & !0x3FFF_FFFF {
            return true;
        }
        serial_println!(
            "SMP: L3[{}] is a 1 GiB huge at phys 0x{:x}, refusing to overwrite",
            l3_idx, base,
        );
        return false;
    }

    // L2.
    let l2_phys = l3_entry & ADDR_MASK;
    let l2_ptr = (phys_mem_offset + l2_phys) as *mut u64;
    let l2_entry = unsafe { ptr::read_volatile(l2_ptr.add(l2_idx)) };

    if l2_entry & PRESENT != 0 && l2_entry & HUGE != 0 {
        let base = l2_entry & ADDR_MASK;
        if base == target_phys & !0x1F_FFFF {
            return true;
        }
        serial_println!(
            "SMP: L2[{}] is a 2 MiB huge at phys 0x{:x}, refusing to overwrite",
            l2_idx, base,
        );
        return false;
    }

    // L1 — follow existing L2 pointer, or create a new L1 at our
    // scratch frame if L2[l2_idx] is absent.
    let l1_phys = if l2_entry & PRESENT != 0 {
        l2_entry & ADDR_MASK
    } else {
        let l1_frame = IDENTITY_L2_PHYS;
        let l1_virt = phys_mem_offset + l1_frame;
        unsafe { ptr::write_bytes(l1_virt as *mut u8, 0, 4096) };
        unsafe {
            ptr::write_volatile(l2_ptr.add(l2_idx), l1_frame | PRESENT | WRITABLE);
        }
        l1_frame
    };

    let l1_ptr = (phys_mem_offset + l1_phys) as *mut u64;
    let l1_entry = unsafe { ptr::read_volatile(l1_ptr.add(l1_idx)) };
    if l1_entry & PRESENT != 0 {
        let mapped = l1_entry & ADDR_MASK;
        if mapped == target_phys {
            return true;
        }
        serial_println!(
            "SMP: L1[{}] already maps to 0x{:x}, refusing to overwrite",
            l1_idx, mapped,
        );
        return false;
    }

    unsafe {
        ptr::write_volatile(l1_ptr.add(l1_idx), target_phys | PRESENT | WRITABLE);
    }

    // Reload CR3 to drop stale non-global TLB entries. Sufficient for a
    // previously-absent translation.
    unsafe {
        core::arch::asm!(
            "mov {tmp}, cr3",
            "mov cr3, {tmp}",
            tmp = out(reg) _,
            options(nostack, preserves_flags),
        );
    }
    true
}

// -- Rust AP entry -----------------------------------------------------

/// Called by every Application Processor immediately after the
/// trampoline's tail-jump. Runs on the per-AP 16 KiB stack with a
/// System V ABI argument (the AP's APIC ID) in `rdi`.
#[unsafe(no_mangle)]
pub extern "C" fn ap_main(cpu_index: u64) -> ! {
    // Install this AP's per-CPU slot first — every future access to
    // `gs:[…]` on this CPU depends on it.
    unsafe { crate::percpu::init(cpu_index as usize) };

    // Verify GS-base round trip: `current().cpu_index` must match the
    // value we just wrote. If not, GS is wrong and Silo state would be
    // routed to the wrong slot.
    let seen = unsafe { crate::percpu::current().cpu_index };
    assert!(seen as u64 == cpu_index, "percpu GS base mismatch");

    // SMP-f contention loop — ensures atomic RMW on a shared location
    // works under real multi-core contention, not just single-AP
    // round-trip.
    for _ in 0..CONTENTION_ITERS {
        CONTENTION_COUNTER.fetch_add(1, Ordering::Relaxed);
    }

    ONLINE_APS.fetch_add(1, Ordering::Release);
    // Park the AP. Interrupts are masked from the trampoline's `cli`,
    // so `hlt` parks the core indefinitely.
    loop {
        unsafe { core::arch::asm!("hlt", options(nomem, nostack, preserves_flags)) };
    }
}

// -- BSP orchestration -------------------------------------------------

/// Wake every enabled non-BSP AP into [`ap_main`]. Each AP is woken
/// serially: fill the shared trampoline header with this AP's per-CPU
/// state, send INIT+SIPI+SIPI, wait for the atomic counter to tick, and
/// move to the next. Serial bring-up keeps the header race-free.
///
/// # Safety
/// Requires the bootloader's `map_physical_memory` mapping at
/// `phys_mem_offset`, the LAPIC and ACPI modules already initialized,
/// and a single call on the BSP.
pub unsafe fn init(phys_mem_offset: u64) {
    if !unsafe { inject_identity_mapping(phys_mem_offset) } {
        serial_println!("SMP: identity mapping failed, aborting AP bring-up");
        return;
    }
    serial_println!("SMP: identity mapping for trampoline page installed");

    unsafe { copy_trampoline(phys_mem_offset) };
    let tramp_len = unsafe { trampoline_bytes() }.len();
    serial_println!(
        "SMP: trampoline copied to 0x{:x} ({} bytes)",
        AP_TRAMPOLINE_PHYS,
        tramp_len,
    );

    // BSP's active PML4 — all APs share it.
    let (cr3_frame, _) = Cr3::read();
    let cr3_phys = cr3_frame.start_address().as_u64();

    // Snapshot ACPI CPU list to avoid holding the lock across per-AP
    // waits. Snapshot before reading BSP APIC ID so the lock order
    // stays consistent across the function.
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

    let entry_addr = ap_main as *const () as u64;
    let mut ap_index = 0usize;
    for cpu in &cpus[..cpu_count] {
        if cpu.apic_id == bsp_apic_id || !cpu.enabled {
            continue;
        }
        if ap_index >= MAX_CPUS {
            serial_println!("SMP: AP count exceeds MAX_CPUS, skipping rest");
            break;
        }

        let stack_top = unsafe { ap_stack_top(ap_index) };
        // Header's `arg` is the per-CPU index (bring-up order). BSP is
        // index 0 and skipped; APs are 1..=N in the order they come up.
        let cpu_index = (ap_index + 1) as u64;
        unsafe {
            write_trampoline_header(
                phys_mem_offset,
                cr3_phys,
                stack_top,
                entry_addr,
                cpu_index,
            );
        }

        let before = ONLINE_APS.load(Ordering::Acquire);

        // Canonical Intel MP-init: INIT → wait → SIPI → wait → SIPI.
        {
            let mut g = LAPIC.lock();
            let lapic = g.as_mut().expect("LAPIC present");
            lapic.send_init_ipi(cpu.apic_id);
        }
        busy_wait_cycles(INIT_TO_SIPI_CYCLES);

        // SIPI #1. Vector `0x08` points the AP at phys `0x8000`.
        {
            let mut g = LAPIC.lock();
            let lapic = g.as_mut().expect("LAPIC present");
            lapic.send_startup_ipi(cpu.apic_id, 0x08);
        }
        busy_wait_cycles(POST_SIPI_CYCLES);

        // SIPI #2: Intel's MP-init sequence sends two SIPIs
        // unconditionally; a no-op if #1 already latched.
        {
            let mut g = LAPIC.lock();
            let lapic = g.as_mut().expect("LAPIC present");
            lapic.send_startup_ipi(cpu.apic_id, 0x08);
        }
        busy_wait_cycles(POST_SIPI_CYCLES);

        // Wait for this AP to tick the counter, or time out.
        let deadline = rdtsc().wrapping_add(AP_ONLINE_TIMEOUT_CYCLES);
        loop {
            if ONLINE_APS.load(Ordering::Acquire) > before {
                serial_println!("SMP: AP apic_id={} online", cpu.apic_id);
                break;
            }
            if rdtsc() >= deadline {
                serial_println!(
                    "SMP: AP apic_id={} did not come online before timeout",
                    cpu.apic_id,
                );
                break;
            }
            core::hint::spin_loop();
        }

        ap_index += 1;
    }

    let total = ONLINE_APS.load(Ordering::Acquire);
    serial_println!("SMP: {} AP(s) online", total);

    // SMP-f gate: every online AP did `CONTENTION_ITERS` atomic
    // increments on `CONTENTION_COUNTER`. The total is a tight
    // invariant — lost updates would collapse the sum.
    let expected = total as u64 * CONTENTION_ITERS;
    let got = CONTENTION_COUNTER.load(Ordering::Acquire);
    if got == expected {
        serial_println!(
            "SMP: contention counter = {} (expected {}, no lost updates)",
            got, expected,
        );
    } else {
        serial_println!(
            "SMP: contention counter = {} (expected {}, {} LOST)",
            got, expected, expected.wrapping_sub(got),
        );
    }
}
