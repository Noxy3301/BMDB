//! Local APIC (xAPIC mode) driver.
//!
//! The Local APIC is a per-CPU interrupt controller. We need it to send
//! Inter-Processor Interrupts (IPIs) — specifically the INIT+SIPI sequence
//! that wakes Application Processors — and to issue end-of-interrupt
//! acknowledgements.
//!
//! xAPIC was chosen over x2APIC for debuggability: MMIO is easier to
//! inspect under QEMU than MSR-only access. Upgrading to x2APIC is a
//! later optimization; the register layout is bit-compatible for the
//! subset we use here.
//!
//! MMIO safety: the LAPIC MMIO page is covered by firmware MTRR as
//! uncacheable, so the bootloader's default cacheable page mapping is
//! overridden by the CPU and all loads/stores are treated as UC. All
//! register accesses are `read_volatile` / `write_volatile` so the
//! compiler does not coalesce or reorder them.

use bmdb_core::sync::SpinLock;
use bmdb_serial::serial_println;
use x86_64::registers::model_specific::Msr;

/// MSR that exposes the APIC base physical address and control bits.
/// Bits [35:12] = physical base, bit 11 = global enable, bit 8 = BSP flag.
const IA32_APIC_BASE: u32 = 0x1B;

// xAPIC register offsets (bytes from MMIO base).
const REG_ID: usize = 0x020;
const REG_VERSION: usize = 0x030;
const REG_TPR: usize = 0x080;
#[allow(dead_code)] // used when interrupt handlers land
const REG_EOI: usize = 0x0B0;
const REG_SIVR: usize = 0x0F0;
const REG_ICR_LO: usize = 0x300;
const REG_ICR_HI: usize = 0x310;

/// SIVR bit 8: software-enable the local APIC.
const SIVR_APIC_SOFT_ENABLE: u32 = 1 << 8;
/// Arbitrary vector delivered to the CPU when the APIC spuriously asserts.
/// Not wired to a handler yet; if it fires before we install one it falls
/// through to the IDT's default trap, which is a loud enough failure.
const SPURIOUS_VECTOR: u32 = 0xFF;

// ICR low-register field encodings.
const ICR_DELIVERY_INIT: u32 = 0b101 << 8;
const ICR_DELIVERY_STARTUP: u32 = 0b110 << 8;
const ICR_DEST_PHYSICAL: u32 = 0 << 11;
const ICR_LEVEL_ASSERT: u32 = 1 << 14;
const ICR_DELIVERY_STATUS_PENDING: u32 = 1 << 12;

pub struct Lapic {
    mmio_base: *mut u32,
}

// The raw MMIO pointer is `!Send` by default. LAPIC MMIO is per-CPU
// hardware, and all access in this kernel routes through the global
// `SpinLock<Option<Lapic>>`, which serializes both ownership and MMIO
// ordering. The `Send` impl declares that moving this handle between
// threads is sound under that discipline.
unsafe impl Send for Lapic {}

impl Lapic {
    /// Read `IA32_APIC_BASE`, translate the physical base into a virtual
    /// pointer through the bootloader's all-physical-memory mapping, and
    /// software-enable the APIC.
    ///
    /// # Safety
    /// - `phys_mem_offset` must be the offset of a valid identity mapping
    ///   of all physical memory into the kernel's address space (this is
    ///   what the bootloader's `map_physical_memory` feature installs).
    /// - Must not be called more than once for the same logical CPU; two
    ///   simultaneous `&mut Lapic` values would alias the same MMIO region
    ///   and break Rust's aliasing model.
    pub unsafe fn new(phys_mem_offset: u64) -> Self {
        let apic_base_msr = unsafe { Msr::new(IA32_APIC_BASE).read() };
        // Mask to bits [35:12]. Higher address bits are reserved on every
        // CPU we target and must be zero; the APIC enable / BSP flags
        // live below bit 12 so they are masked out too.
        let phys_base = apic_base_msr & 0x0000_000F_FFFF_F000;
        let mmio_base = (phys_mem_offset + phys_base) as *mut u32;

        let mut lapic = Self { mmio_base };
        lapic.enable();
        lapic
    }

    fn reg_ptr(&self, byte_offset: usize) -> *mut u32 {
        debug_assert!(byte_offset % 4 == 0, "LAPIC register offsets are 4-byte aligned");
        // Safety: `byte_offset` is a known xAPIC register offset within
        // the 4 KiB MMIO page; the pointer stays inside the allocation.
        unsafe { (self.mmio_base as *mut u8).add(byte_offset) as *mut u32 }
    }

    fn read_reg(&self, byte_offset: usize) -> u32 {
        // Safety: MMIO page is mapped and the offset is in-range. Volatile
        // keeps the compiler from coalescing repeated reads of the same
        // register (some LAPIC registers — e.g. timer current count — are
        // live hardware counters).
        unsafe { core::ptr::read_volatile(self.reg_ptr(byte_offset)) }
    }

    fn write_reg(&mut self, byte_offset: usize, value: u32) {
        // Safety: same as `read_reg`; volatile ensures each write reaches
        // the hardware register.
        unsafe { core::ptr::write_volatile(self.reg_ptr(byte_offset), value) }
    }

    fn enable(&mut self) {
        // Task priority = 0 so the APIC will deliver any prioritized IRQ.
        self.write_reg(REG_TPR, 0);
        self.write_reg(REG_SIVR, SIVR_APIC_SOFT_ENABLE | SPURIOUS_VECTOR);
    }

    pub fn id(&self) -> u32 {
        self.read_reg(REG_ID) >> 24
    }

    pub fn version(&self) -> u32 {
        self.read_reg(REG_VERSION) & 0xFF
    }

    /// Signal end-of-interrupt for the current in-service IRQ. Every
    /// external-interrupt handler must write this before returning or the
    /// same IRQ will never re-fire.
    #[allow(dead_code)] // wired in once an interrupt handler actually lands
    pub fn eoi(&mut self) {
        self.write_reg(REG_EOI, 0);
    }

    fn wait_for_ipi_delivery(&self) {
        // One-shot spin: the LAPIC's Delivery Status bit clears the
        // instant it finishes the previous IPI. On a healthy APIC this
        // loop runs at most a handful of cycles.
        while self.read_reg(REG_ICR_LO) & ICR_DELIVERY_STATUS_PENDING != 0 {
            core::hint::spin_loop();
        }
    }

    /// Send an INIT IPI (assert, edge-triggered) to the given physical
    /// APIC ID. First leg of the MP init protocol: the target AP resets
    /// and is ready to receive a Startup IPI ~10 ms later. Per Intel SDM
    /// Vol 3A §11.4.4 the canonical encoding is `Level = Assert,
    /// Trigger = Edge`; the trigger bit is ignored for INIT on modern
    /// CPUs but is left edge here to match the spec.
    #[allow(dead_code)]
    pub fn send_init_ipi(&mut self, target_apic_id: u8) {
        self.wait_for_ipi_delivery();
        self.write_reg(REG_ICR_HI, (target_apic_id as u32) << 24);
        self.write_reg(
            REG_ICR_LO,
            ICR_DELIVERY_INIT | ICR_DEST_PHYSICAL | ICR_LEVEL_ASSERT,
        );
        self.wait_for_ipi_delivery();
    }

    /// Send a Startup IPI pointing at `vector << 12` as the AP's 16-bit
    /// real-mode entry point. Typical value is `0x08`, meaning the AP
    /// starts executing at physical `0x8000`. Per Intel SDM the SIPI
    /// encoding is `Level = Assert, Trigger = Edge`.
    #[allow(dead_code)]
    pub fn send_startup_ipi(&mut self, target_apic_id: u8, vector: u8) {
        self.wait_for_ipi_delivery();
        self.write_reg(REG_ICR_HI, (target_apic_id as u32) << 24);
        self.write_reg(
            REG_ICR_LO,
            ICR_DELIVERY_STARTUP | ICR_DEST_PHYSICAL | ICR_LEVEL_ASSERT | vector as u32,
        );
        self.wait_for_ipi_delivery();
    }
}

/// Singleton handle to the BSP's LAPIC. Populated by [`init`]. APs will
/// each get their own `Lapic` instance once SMP-d brings them up; this
/// slot is specifically the bootstrap processor's.
pub static LAPIC: SpinLock<Option<Lapic>> = SpinLock::new(None);

/// Initialize the BSP's local APIC.
///
/// # Safety
/// Carries the same requirements as [`Lapic::new`]: a valid physical
/// memory offset from the bootloader, and this function must be called
/// exactly once per boot on the BSP.
pub unsafe fn init(phys_mem_offset: u64) {
    let lapic = unsafe { Lapic::new(phys_mem_offset) };
    let id = lapic.id();
    let version = lapic.version();
    serial_println!("LAPIC: enabled (id={}, version=0x{:02x})", id, version);
    *LAPIC.lock() = Some(lapic);
}
