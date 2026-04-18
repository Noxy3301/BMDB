//! NVMe controller initialization and Identify Controller command.

use crate::{pci, serial_println};
use core::ptr;
use x86_64::{VirtAddr, structures::paging::Translate};

// Controller register offsets (NVMe spec Section 3.1).
const REG_CAP: usize = 0x00; // Controller Capabilities (64-bit)
const REG_VS: usize = 0x08; // Version (32-bit)
const REG_CC: usize = 0x14; // Controller Configuration (32-bit)
const REG_CSTS: usize = 0x1C; // Controller Status (32-bit)
const REG_AQA: usize = 0x24; // Admin Queue Attributes (32-bit)
const REG_ASQ: usize = 0x28; // Admin SQ base (64-bit)
const REG_ACQ: usize = 0x30; // Admin CQ base (64-bit)

// Admin Submission/Completion Queue depth.
const ADMIN_QD: usize = 64;

// Admin Submission Queue Entry. Exactly 64 bytes per NVMe spec.
#[repr(C)]
#[derive(Clone, Copy)]
struct SqEntry {
    cdw0: u32,
    nsid: u32,
    _rsvd: [u32; 2],
    mptr: u64,
    prp1: u64,
    prp2: u64,
    cdw10: u32,
    cdw11: u32,
    cdw12: u32,
    cdw13: u32,
    cdw14: u32,
    cdw15: u32,
}

const EMPTY_SQE: SqEntry = SqEntry {
    cdw0: 0,
    nsid: 0,
    _rsvd: [0; 2],
    mptr: 0,
    prp1: 0,
    prp2: 0,
    cdw10: 0,
    cdw11: 0,
    cdw12: 0,
    cdw13: 0,
    cdw14: 0,
    cdw15: 0,
};

// Admin Completion Queue Entry. Exactly 16 bytes per NVMe spec.
#[repr(C)]
#[derive(Clone, Copy)]
struct CqEntry {
    dw0: u32,
    dw1: u32,
    sq_head: u16,
    sq_id: u16,
    cid: u16,
    status: u16,
}

const EMPTY_CQE: CqEntry = CqEntry {
    dw0: 0,
    dw1: 0,
    sq_head: 0,
    sq_id: 0,
    cid: 0,
    status: 0,
};

// NVMe requires queues and data buffers to be page-aligned.
#[repr(C, align(4096))]
struct Page4K<T>(T);

static mut ADMIN_SQ: Page4K<[SqEntry; ADMIN_QD]> = Page4K([EMPTY_SQE; ADMIN_QD]);
static mut ADMIN_CQ: Page4K<[CqEntry; ADMIN_QD]> = Page4K([EMPTY_CQE; ADMIN_QD]);
static mut IDENTIFY_BUF: Page4K<[u8; 4096]> = Page4K([0; 4096]);

#[inline]
fn reg_read32(base: *mut u8, off: usize) -> u32 {
    unsafe { ptr::read_volatile(base.add(off) as *const u32) }
}

#[inline]
fn reg_write32(base: *mut u8, off: usize, v: u32) {
    unsafe { ptr::write_volatile(base.add(off) as *mut u32, v) }
}

#[inline]
fn reg_read64(base: *mut u8, off: usize) -> u64 {
    unsafe { ptr::read_volatile(base.add(off) as *const u64) }
}

#[inline]
fn reg_write64(base: *mut u8, off: usize, v: u64) {
    unsafe { ptr::write_volatile(base.add(off) as *mut u64, v) }
}

/// Discover the NVMe controller, initialize its admin queues, and run
/// Identify Controller. Prints key fields on success.
pub fn init(phys_mem_offset: VirtAddr, mapper: &impl Translate) {
    let Some(addr) = pci::find_device(0x01, 0x08) else {
        serial_println!("NVMe: no controller found");
        return;
    };
    serial_println!(
        "NVMe: found at {:02x}:{:02x}.{}",
        addr.bus,
        addr.device,
        addr.function
    );
    pci::enable_device(&addr);

    let bar0 = pci::read_bar(&addr, 0);
    let base = (phys_mem_offset.as_u64() + bar0) as *mut u8;

    let cap = reg_read64(base, REG_CAP);
    let vs = reg_read32(base, REG_VS);
    serial_println!("NVMe: BAR0 = {:#x}, CAP = {:#x}, VS = {:#x}", bar0, cap, vs);

    // Disable the controller before reconfiguring admin queues.
    reg_write32(base, REG_CC, 0);
    while reg_read32(base, REG_CSTS) & 1 != 0 {
        core::hint::spin_loop();
    }

    // Translate kernel-side static buffers to their physical frames so the
    // controller can DMA into them.
    let sq_virt = VirtAddr::new(&raw const ADMIN_SQ as u64);
    let cq_virt = VirtAddr::new(&raw const ADMIN_CQ as u64);
    let id_virt = VirtAddr::new(&raw const IDENTIFY_BUF as u64);
    let sq_phys = mapper
        .translate_addr(sq_virt)
        .expect("ADMIN_SQ has no physical mapping");
    let cq_phys = mapper
        .translate_addr(cq_virt)
        .expect("ADMIN_CQ has no physical mapping");
    let id_phys = mapper
        .translate_addr(id_virt)
        .expect("IDENTIFY_BUF has no physical mapping");

    // AQA: ACQS and ASQS are 0-based, so size N is encoded as N-1.
    let aqa = (((ADMIN_QD - 1) as u32) << 16) | ((ADMIN_QD - 1) as u32);
    reg_write32(base, REG_AQA, aqa);
    reg_write64(base, REG_ASQ, sq_phys.as_u64());
    reg_write64(base, REG_ACQ, cq_phys.as_u64());

    // CC: EN=1, IOSQES=6 (64-byte SQE), IOCQES=4 (16-byte CQE). Other fields 0.
    let cc = (6 << 16) | (4 << 20) | 1;
    reg_write32(base, REG_CC, cc);

    // Wait for CSTS.RDY. Abort if CSTS.CFS (bit 1) lights up.
    loop {
        let csts = reg_read32(base, REG_CSTS);
        if csts & 1 != 0 {
            break;
        }
        if csts & 2 != 0 {
            serial_println!("NVMe: controller fatal status, CSTS = {:#x}", csts);
            return;
        }
        core::hint::spin_loop();
    }
    serial_println!("NVMe: controller enabled");

    // Build Identify Controller command at SQ slot 0.
    // Opcode 0x06, CID=0, NSID=0, PRP1 = buffer physical, CDW10.CNS = 0x01.
    unsafe {
        let sq = &raw mut ADMIN_SQ;
        let slot = &mut (*sq).0[0];
        *slot = EMPTY_SQE;
        slot.cdw0 = 0x06; // opcode only; CID = 0
        slot.prp1 = id_phys.as_u64();
        slot.cdw10 = 0x01; // CNS = Identify Controller
    }

    // Ring Admin SQ tail doorbell. DSTRD = CAP[51:48].
    let dstrd = ((cap >> 32) >> 16) as usize & 0xF;
    let sq_tail_db = 0x1000 + 0 * (4 << dstrd);
    let cq_head_db = 0x1000 + 1 * (4 << dstrd);
    reg_write32(base, sq_tail_db, 1);

    // Poll completion slot 0 until the phase bit flips to 1.
    let completion;
    loop {
        let cqe = unsafe { ptr::read_volatile(&raw const (*(&raw const ADMIN_CQ)).0[0]) };
        if cqe.status & 1 != 0 {
            completion = cqe;
            break;
        }
        core::hint::spin_loop();
    }
    // Acknowledge: advance CQ head to 1.
    reg_write32(base, cq_head_db, 1);

    // NVMe status: bit 0 is phase, bits 15:1 are the status field.
    let status_code = (completion.status >> 1) & 0xFF;
    if status_code != 0 {
        serial_println!("NVMe: Identify failed, status = {:#x}", completion.status);
        return;
    }

    // Parse the Identify Controller data (NVMe spec figure for CNS 01h).
    unsafe {
        let buf = &(*(&raw const IDENTIFY_BUF)).0;
        let vid = u16::from_le_bytes([buf[0], buf[1]]);
        let sn = core::str::from_utf8(&buf[4..24]).unwrap_or("?");
        let mn = core::str::from_utf8(&buf[24..64]).unwrap_or("?");
        let fr = core::str::from_utf8(&buf[64..72]).unwrap_or("?");
        serial_println!("NVMe: VID = {:#06x}", vid);
        serial_println!("NVMe: SN  = {:?}", sn.trim());
        serial_println!("NVMe: MN  = {:?}", mn.trim());
        serial_println!("NVMe: FR  = {:?}", fr.trim());
    }
}
