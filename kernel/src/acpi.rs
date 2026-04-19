//! Minimal ACPI parser: find the MADT, enumerate Local APIC entries.
//!
//! Goal is narrow — SMP-d needs the list of Application Processor APIC
//! IDs. We walk only enough of ACPI to reach there: RSDP → XSDT/RSDT →
//! MADT, skipping every other table and every MADT entry type besides
//! Type 0 (Processor Local APIC).
//!
//! All ACPI tables are firmware-written byte blobs that are not aligned
//! to their field sizes, so every multi-byte field read uses
//! `read_unaligned`.
//!
//! Discovery assumes legacy BIOS boot: the RSDP lives in the last 128
//! KiB of the first megabyte (0xE_0000..0x10_0000). UEFI systems hand
//! the RSDP pointer over through the bootloader; this kernel uses
//! `bootloader` 0.9 which does not, so we fall back to the scan.

use bmdb_core::sync::SpinLock;
use bmdb_serial::serial_println;

/// All ACPI tables share a 36-byte header (signature, length, revision,
/// checksum, OEM ID, OEM Table ID, OEM Revision, Creator ID,
/// Creator Revision). Data follows immediately after.
const ACPI_HEADER_SIZE: usize = 36;

/// Max CPUs we care to record. Bounds the static array. ThinkCentre M920q
/// has at most 6 cores × 2 threads = 12, so 64 is comfortably above any
/// realistic workstation-class target.
pub const MAX_CPUS: usize = 64;

#[derive(Clone, Copy, Default)]
pub struct CpuInfo {
    /// Firmware-assigned logical processor ID; opaque, used only for display.
    pub processor_id: u8,
    /// Physical APIC ID. This is what INIT/SIPI targets reference.
    pub apic_id: u8,
    /// MADT flags bit 0. Only enabled CPUs are valid IPI targets.
    pub enabled: bool,
}

pub struct AcpiInfo {
    cpus: [CpuInfo; MAX_CPUS],
    count: usize,
    /// LAPIC MMIO base reported by MADT header. Should match `IA32_APIC_BASE`
    /// on well-behaved firmware; recorded here for cross-validation.
    pub lapic_address: u32,
}

impl AcpiInfo {
    pub fn cpus(&self) -> &[CpuInfo] {
        &self.cpus[..self.count]
    }

    /// Count only enabled CPUs — the set a caller would actually try to
    /// wake via INIT/SIPI.
    #[allow(dead_code)] // consumed by SMP-d
    pub fn enabled_cpu_count(&self) -> usize {
        self.cpus().iter().filter(|c| c.enabled).count()
    }
}

/// Populated by [`init`]. Remains `None` if RSDP/MADT discovery fails —
/// callers that need ACPI must handle that explicitly.
pub static ACPI: SpinLock<Option<AcpiInfo>> = SpinLock::new(None);

/// Scan the BIOS ROM region for the RSDP signature. Returns the virtual
/// pointer for a valid RSDP, or `None` if nothing passes the checksum.
/// For revision >= 2 RSDPs, both the 20-byte (ACPI 1.0) and the 36-byte
/// (extended) checksums must hold — otherwise the XSDT pointer at
/// offset 24 is not trustworthy (ACPI 6.4 §5.2.5.3).
unsafe fn find_rsdp(phys_mem_offset: u64) -> Option<*const u8> {
    const SIG: &[u8; 8] = b"RSD PTR ";
    const START: u64 = 0xE_0000;
    const END: u64 = 0x10_0000;
    // ACPI spec: RSDP is aligned on a 16-byte boundary within this region.
    const STEP: u64 = 16;

    let mut phys = START;
    // A valid RSDP starts with the 8-byte signature and continues for 20
    // (ACPI 1.0) or 36 (ACPI 2.0+) bytes. The outer bound only needs to
    // guarantee that the signature plus the 1.0 footprint is in range;
    // the revision-specific 36-byte check is gated again below so we do
    // not miss a 1.0 RSDP that sits in the last aligned slot.
    while phys + 20 <= END {
        let ptr = (phys_mem_offset + phys) as *const u8;
        let sig = unsafe { core::slice::from_raw_parts(ptr, 8) };
        if sig == SIG {
            let first20 = unsafe { core::slice::from_raw_parts(ptr, 20) };
            let sum20: u8 = first20.iter().fold(0u8, |a, b| a.wrapping_add(*b));
            if sum20 != 0 {
                phys += STEP;
                continue;
            }
            let revision = unsafe { *ptr.add(15) };
            if revision >= 2 {
                if phys + 36 > END {
                    // The RSDP claims ACPI 2.0+ but its extended tail
                    // would run past the ROM window. Treat as malformed.
                    phys += STEP;
                    continue;
                }
                let full = unsafe { core::slice::from_raw_parts(ptr, 36) };
                let sum36: u8 = full.iter().fold(0u8, |a, b| a.wrapping_add(*b));
                if sum36 != 0 {
                    // ACPI 1.0 checksum matched but extended one didn't —
                    // reject: treating the XSDT pointer as valid would
                    // feed garbage into `walk_xsdt`.
                    phys += STEP;
                    continue;
                }
            }
            return Some(ptr);
        }
        phys += STEP;
    }
    None
}

/// Read a 4-byte signature from the beginning of an ACPI table.
unsafe fn read_signature(table: *const u8) -> [u8; 4] {
    let mut sig = [0u8; 4];
    for (i, slot) in sig.iter_mut().enumerate() {
        *slot = unsafe { *table.add(i) };
    }
    sig
}

/// Read the `length` field from an ACPI table header.
unsafe fn read_table_length(table: *const u8) -> u32 {
    unsafe { core::ptr::read_unaligned(table.add(4) as *const u32) }
}

/// Walk an XSDT (ACPI 2.0+, 64-bit entry pointers) looking for "APIC".
unsafe fn walk_xsdt(phys_mem_offset: u64, xsdt_phys: u64) -> Option<*const u8> {
    let xsdt = (phys_mem_offset + xsdt_phys) as *const u8;
    let length = unsafe { read_table_length(xsdt) } as usize;
    if length < ACPI_HEADER_SIZE {
        return None;
    }
    let entry_bytes = length - ACPI_HEADER_SIZE;
    let entry_count = entry_bytes / core::mem::size_of::<u64>();
    let entries = unsafe { xsdt.add(ACPI_HEADER_SIZE) } as *const u64;
    for i in 0..entry_count {
        let entry_phys = unsafe { core::ptr::read_unaligned(entries.add(i)) };
        let table = (phys_mem_offset + entry_phys) as *const u8;
        if unsafe { read_signature(table) } == *b"APIC" {
            return Some(table);
        }
    }
    None
}

/// Walk an RSDT (ACPI 1.0, 32-bit entry pointers) looking for "APIC".
unsafe fn walk_rsdt(phys_mem_offset: u64, rsdt_phys: u64) -> Option<*const u8> {
    let rsdt = (phys_mem_offset + rsdt_phys) as *const u8;
    let length = unsafe { read_table_length(rsdt) } as usize;
    if length < ACPI_HEADER_SIZE {
        return None;
    }
    let entry_bytes = length - ACPI_HEADER_SIZE;
    let entry_count = entry_bytes / core::mem::size_of::<u32>();
    let entries = unsafe { rsdt.add(ACPI_HEADER_SIZE) } as *const u32;
    for i in 0..entry_count {
        let entry_phys = unsafe { core::ptr::read_unaligned(entries.add(i)) } as u64;
        let table = (phys_mem_offset + entry_phys) as *const u8;
        if unsafe { read_signature(table) } == *b"APIC" {
            return Some(table);
        }
    }
    None
}

/// Resolve RSDP → root table → MADT.
unsafe fn find_madt(phys_mem_offset: u64, rsdp: *const u8) -> Option<*const u8> {
    let revision = unsafe { *rsdp.add(15) };
    if revision >= 2 {
        let xsdt_phys: u64 =
            unsafe { core::ptr::read_unaligned(rsdp.add(24) as *const u64) };
        if xsdt_phys != 0 {
            if let Some(madt) = unsafe { walk_xsdt(phys_mem_offset, xsdt_phys) } {
                return Some(madt);
            }
        }
    }
    let rsdt_phys: u32 = unsafe { core::ptr::read_unaligned(rsdp.add(16) as *const u32) };
    unsafe { walk_rsdt(phys_mem_offset, rsdt_phys as u64) }
}

/// Extract Type-0 (Processor Local APIC) entries from an already-located
/// MADT. A malformed header (length < 44 bytes = ACPI header + the two
/// mandatory MADT-specific u32s) returns an empty result rather than
/// reading past the table.
unsafe fn parse_madt(madt: *const u8) -> AcpiInfo {
    let length = unsafe { read_table_length(madt) } as usize;
    // MADT requires header (36) + local_apic_address (u32) + flags (u32).
    // Without that minimum the fixed-field reads below would run past
    // whatever the table actually contains.
    const MIN_MADT_LEN: usize = ACPI_HEADER_SIZE + 8;
    if length < MIN_MADT_LEN {
        return AcpiInfo {
            cpus: [CpuInfo::default(); MAX_CPUS],
            count: 0,
            lapic_address: 0,
        };
    }

    // Every address derivation from a firmware-controlled length is
    // done in `usize` with `checked_add` before being cast back to a
    // pointer. That keeps a pathological high-address wraparound from
    // silently producing an in-range-looking pointer that actually
    // overflowed.
    let madt_addr = madt as usize;
    let entries_end_addr = match madt_addr.checked_add(length) {
        Some(v) => v,
        None => {
            return AcpiInfo {
                cpus: [CpuInfo::default(); MAX_CPUS],
                count: 0,
                lapic_address: 0,
            };
        }
    };
    let entries_start_addr = madt_addr + MIN_MADT_LEN; // MIN_MADT_LEN <= length, already validated
    let lapic_address_addr = madt_addr + ACPI_HEADER_SIZE;

    // First MADT-specific field: Local APIC Address (u32) at header+0.
    let lapic_address: u32 =
        unsafe { core::ptr::read_unaligned(lapic_address_addr as *const u32) };
    // Second field: Flags (u32) at header+4. Unused.

    let mut cpus = [CpuInfo::default(); MAX_CPUS];
    let mut count = 0usize;

    let mut p_addr = entries_start_addr;
    loop {
        match p_addr.checked_add(2) {
            Some(v) if v <= entries_end_addr => {}
            _ => break,
        }
        let p = p_addr as *const u8;
        let ty = unsafe { *p };
        let len = unsafe { *p.add(1) } as usize;
        if len < 2 {
            // Zero- or one-byte entries would loop forever or truncate
            // the next header read. Stop rather than trust the firmware.
            break;
        }
        let next_addr = match p_addr.checked_add(len) {
            Some(v) if v <= entries_end_addr => v,
            _ => break,
        };
        // Type 0 = Processor Local APIC (8 bytes total).
        if ty == 0 && len == 8 && count < MAX_CPUS {
            let processor_id = unsafe { *p.add(2) };
            let apic_id = unsafe { *p.add(3) };
            let flags: u32 = unsafe { core::ptr::read_unaligned(p.add(4) as *const u32) };
            cpus[count] = CpuInfo {
                processor_id,
                apic_id,
                enabled: (flags & 1) != 0,
            };
            count += 1;
        }
        p_addr = next_addr;
    }

    AcpiInfo {
        cpus,
        count,
        lapic_address,
    }
}

/// Discover and parse ACPI. On success, prints a one-line-per-CPU
/// summary and stores the result in [`ACPI`]. Silent-return on failure
/// so a malformed BIOS does not take the kernel down — SMP-d will see
/// `ACPI.lock().as_ref().is_none()` and refuse to launch APs.
///
/// # Safety
/// Requires the bootloader's `map_physical_memory` mapping to be active
/// at `phys_mem_offset`, and must be called once during kernel init.
pub unsafe fn init(phys_mem_offset: u64) {
    let rsdp = match unsafe { find_rsdp(phys_mem_offset) } {
        Some(p) => p,
        None => {
            serial_println!("ACPI: RSDP not found in BIOS ROM region");
            return;
        }
    };
    let rsdp_phys = rsdp as u64 - phys_mem_offset;
    let revision = unsafe { *rsdp.add(15) };
    serial_println!(
        "ACPI: RSDP at phys 0x{:x} (revision {})",
        rsdp_phys,
        revision,
    );

    let madt = match unsafe { find_madt(phys_mem_offset, rsdp) } {
        Some(p) => p,
        None => {
            serial_println!("ACPI: MADT not found");
            return;
        }
    };

    let info = unsafe { parse_madt(madt) };
    serial_println!(
        "ACPI: MADT local_apic_address=0x{:08x}, {} CPU(s)",
        info.lapic_address,
        info.count,
    );
    for cpu in info.cpus() {
        serial_println!(
            "  CPU processor_id={}, apic_id={}, enabled={}",
            cpu.processor_id,
            cpu.apic_id,
            cpu.enabled,
        );
    }
    *ACPI.lock() = Some(info);
}
