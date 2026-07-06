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
//! Discovery prefers the bootloader-reported RSDP address (`bootloader`
//! 0.11 forwards it on both UEFI and BIOS boots — on UEFI the RSDP
//! lives in an EFI configuration table at an arbitrary high address).
//! When the bootloader reports nothing, fall back to the legacy BIOS
//! scan of the last 128 KiB of the first megabyte
//! (0xE_0000..0x10_0000).

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

/// Validate an RSDP candidate: signature plus checksums. For revision
/// >= 2 RSDPs, both the 20-byte (ACPI 1.0) and the 36-byte (extended)
/// checksums must hold — otherwise the XSDT pointer at offset 24 is not
/// trustworthy (ACPI 6.4 §5.2.5.3). `max_len` bounds how many bytes are
/// readable at `ptr`, so a truncated candidate near a region edge is
/// rejected instead of read past.
unsafe fn validate_rsdp(ptr: *const u8, max_len: usize) -> bool {
    const SIG: &[u8; 8] = b"RSD PTR ";

    if max_len < 20 {
        return false;
    }
    let sig = unsafe { core::slice::from_raw_parts(ptr, 8) };
    if sig != SIG {
        return false;
    }
    let first20 = unsafe { core::slice::from_raw_parts(ptr, 20) };
    let sum20: u8 = first20.iter().fold(0u8, |a, b| a.wrapping_add(*b));
    if sum20 != 0 {
        return false;
    }
    let revision = unsafe { *ptr.add(15) };
    if revision >= 2 {
        // The RSDP claims ACPI 2.0+; its extended tail must be readable
        // and checksum clean, or the XSDT pointer would be garbage.
        if max_len < 36 {
            return false;
        }
        let full = unsafe { core::slice::from_raw_parts(ptr, 36) };
        let sum36: u8 = full.iter().fold(0u8, |a, b| a.wrapping_add(*b));
        if sum36 != 0 {
            return false;
        }
    }
    true
}

/// Scan the BIOS ROM region for the RSDP signature. Returns the virtual
/// pointer for a valid RSDP, or `None` if nothing passes the checksum.
unsafe fn find_rsdp(phys_mem_offset: u64) -> Option<*const u8> {
    const START: u64 = 0xE_0000;
    const END: u64 = 0x10_0000;
    // ACPI spec: RSDP is aligned on a 16-byte boundary within this region.
    const STEP: u64 = 16;

    let mut phys = START;
    while phys + 20 <= END {
        let ptr = (phys_mem_offset + phys) as *const u8;
        if unsafe { validate_rsdp(ptr, (END - phys) as usize) } {
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
/// `rsdp_addr` is the physical RSDP address as reported by the
/// bootloader; the legacy BIOS ROM scan runs only when it is absent
/// or fails validation.
///
/// # Safety
/// Requires the bootloader's whole-physical-memory mapping to be active
/// at `phys_mem_offset`, and must be called once during kernel init.
pub unsafe fn init(phys_mem_offset: u64, rsdp_addr: Option<u64>) {
    let reported = rsdp_addr.and_then(|phys| {
        let ptr = (phys_mem_offset + phys) as *const u8;
        // The full 36-byte footprint is readable through the offset
        // mapping wherever the firmware placed it.
        if unsafe { validate_rsdp(ptr, 36) } {
            Some(ptr)
        } else {
            serial_println!("ACPI: bootloader RSDP at 0x{:x} failed validation", phys);
            None
        }
    });
    let rsdp = match reported.or_else(|| unsafe { find_rsdp(phys_mem_offset) }) {
        Some(p) => p,
        None => {
            serial_println!("ACPI: RSDP not found (no bootloader report, scan empty)");
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
