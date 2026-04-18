//! NVMe controller initialization, admin/I/O queue management, and 512-byte
//! block I/O on namespace 1.

#![no_std]

use bmdb_pci as pci;
use bmdb_serial::serial_println;
use core::ptr;
use x86_64::{VirtAddr, structures::paging::Translate};

// Controller register offsets (NVMe spec Section 3.1).
const REG_CAP: usize = 0x00;
const REG_VS: usize = 0x08;
const REG_CC: usize = 0x14;
const REG_CSTS: usize = 0x1C;
const REG_AQA: usize = 0x24;
const REG_ASQ: usize = 0x28;
const REG_ACQ: usize = 0x30;

const ADMIN_QD: u16 = 64;
const IO_QD: u16 = 64;
const IO_QID: u16 = 1;

pub const BLOCK_SIZE: usize = 512;

// Admin Submission Queue Entry, exactly 64 bytes.
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

// Completion Queue Entry, exactly 16 bytes.
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

#[repr(C, align(4096))]
struct Page4K<T>(T);

static mut ADMIN_SQ: Page4K<[SqEntry; ADMIN_QD as usize]> =
    Page4K([EMPTY_SQE; ADMIN_QD as usize]);
static mut ADMIN_CQ: Page4K<[CqEntry; ADMIN_QD as usize]> =
    Page4K([EMPTY_CQE; ADMIN_QD as usize]);
static mut IO_SQ: Page4K<[SqEntry; IO_QD as usize]> = Page4K([EMPTY_SQE; IO_QD as usize]);
static mut IO_CQ: Page4K<[CqEntry; IO_QD as usize]> = Page4K([EMPTY_CQE; IO_QD as usize]);
static mut IDENTIFY_BUF: Page4K<[u8; 4096]> = Page4K([0; 4096]);
// Bounce buffer for data transfers. A single 4K page fits one PRP entry with
// no chaining, so callers are limited to one block per call for now.
static mut DATA_BUF: Page4K<[u8; 4096]> = Page4K([0; 4096]);

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

/// Runtime state for a submission/completion queue pair.
struct Queue {
    sq: *mut SqEntry,
    cq: *const CqEntry,
    qd: u16,
    sq_doorbell: usize,
    cq_doorbell: usize,
    tail: u16,
    head: u16,
    phase: u16,
}

/// A live NVMe controller with one I/O queue pair. Not Sync: the static bounce
/// buffer and queue memory make this single-consumer by construction.
pub struct Controller {
    base: *mut u8,
    admin: Queue,
    io: Queue,
    data_buf_phys: u64,
}

/// NVMe completion status. Non-zero indicates the controller rejected the
/// command; the lower bits carry the status code per NVMe Section 4.5.
#[derive(Debug, Clone, Copy)]
pub struct IoError(pub u16);

/// Submit a command and spin-poll for its completion. Free function so the
/// caller can split-borrow `base` and the queue out of `Controller`.
fn submit(base: *mut u8, q: &mut Queue, mut cmd: SqEntry) -> CqEntry {
    // Use the slot index as the CID so the completion is unambiguous.
    let cid = q.tail;
    cmd.cdw0 = (cmd.cdw0 & 0x0000_FFFF) | ((cid as u32) << 16);

    unsafe {
        q.sq.add(q.tail as usize).write(cmd);
    }
    q.tail = (q.tail + 1) % q.qd;
    reg_write32(base, q.sq_doorbell, q.tail as u32);

    let cqe = loop {
        let e = unsafe { ptr::read_volatile(q.cq.add(q.head as usize)) };
        if (e.status & 1) == q.phase {
            break e;
        }
        core::hint::spin_loop();
    };

    q.head = (q.head + 1) % q.qd;
    if q.head == 0 {
        q.phase ^= 1;
    }
    reg_write32(base, q.cq_doorbell, q.head as u32);

    cqe
}

fn status_code(cqe: &CqEntry) -> u16 {
    (cqe.status >> 1) & 0x7FF
}

/// Discover the NVMe controller, bring it up, and create one I/O queue pair.
/// Returns a handle usable for block I/O, or `None` if no controller is
/// present or initialization fails.
pub fn init(phys_mem_offset: VirtAddr, mapper: &impl Translate) -> Option<Controller> {
    let addr = pci::find_device(0x01, 0x08)?;
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

    reg_write32(base, REG_CC, 0);
    while reg_read32(base, REG_CSTS) & 1 != 0 {
        core::hint::spin_loop();
    }

    let asq_phys = translate(mapper, &raw const ADMIN_SQ as u64);
    let acq_phys = translate(mapper, &raw const ADMIN_CQ as u64);
    let iosq_phys = translate(mapper, &raw const IO_SQ as u64);
    let iocq_phys = translate(mapper, &raw const IO_CQ as u64);
    let id_phys = translate(mapper, &raw const IDENTIFY_BUF as u64);
    let data_phys = translate(mapper, &raw const DATA_BUF as u64);

    let aqa = (((ADMIN_QD - 1) as u32) << 16) | ((ADMIN_QD - 1) as u32);
    reg_write32(base, REG_AQA, aqa);
    reg_write64(base, REG_ASQ, asq_phys);
    reg_write64(base, REG_ACQ, acq_phys);

    let cc = (6 << 16) | (4 << 20) | 1;
    reg_write32(base, REG_CC, cc);

    loop {
        let csts = reg_read32(base, REG_CSTS);
        if csts & 1 != 0 {
            break;
        }
        if csts & 2 != 0 {
            serial_println!("NVMe: fatal status CSTS = {:#x}", csts);
            return None;
        }
        core::hint::spin_loop();
    }
    serial_println!("NVMe: controller enabled");

    let dstrd = ((cap >> 32) >> 16) as usize & 0xF;
    let stride = 4 << dstrd;

    let mut admin = Queue {
        sq: &raw mut ADMIN_SQ as *mut SqEntry,
        cq: &raw const ADMIN_CQ as *const CqEntry,
        qd: ADMIN_QD,
        sq_doorbell: 0x1000,
        cq_doorbell: 0x1000 + stride,
        tail: 0,
        head: 0,
        phase: 1,
    };

    // Identify Controller (CNS = 0x01).
    let mut cmd = EMPTY_SQE;
    cmd.cdw0 = 0x06;
    cmd.prp1 = id_phys;
    cmd.cdw10 = 0x01;
    let cqe = submit(base, &mut admin, cmd);
    if status_code(&cqe) != 0 {
        serial_println!("NVMe: Identify failed, status = {:#x}", cqe.status);
        return None;
    }
    unsafe {
        let buf = &(*(&raw const IDENTIFY_BUF)).0;
        let sn = core::str::from_utf8(&buf[4..24]).unwrap_or("?");
        let mn = core::str::from_utf8(&buf[24..64]).unwrap_or("?");
        serial_println!("NVMe: SN = {:?}, MN = {:?}", sn.trim(), mn.trim());
    }

    // Create I/O Completion Queue (opcode 0x05).
    let mut cmd = EMPTY_SQE;
    cmd.cdw0 = 0x05;
    cmd.prp1 = iocq_phys;
    cmd.cdw10 = (((IO_QD - 1) as u32) << 16) | (IO_QID as u32);
    cmd.cdw11 = 1; // PC=1 (physically contiguous), IEN=0
    let cqe = submit(base, &mut admin, cmd);
    if status_code(&cqe) != 0 {
        serial_println!("NVMe: Create I/O CQ failed, status = {:#x}", cqe.status);
        return None;
    }

    // Create I/O Submission Queue (opcode 0x01).
    let mut cmd = EMPTY_SQE;
    cmd.cdw0 = 0x01;
    cmd.prp1 = iosq_phys;
    cmd.cdw10 = (((IO_QD - 1) as u32) << 16) | (IO_QID as u32);
    cmd.cdw11 = ((IO_QID as u32) << 16) | 1; // CQID in high half, PC=1
    let cqe = submit(base, &mut admin, cmd);
    if status_code(&cqe) != 0 {
        serial_println!("NVMe: Create I/O SQ failed, status = {:#x}", cqe.status);
        return None;
    }
    serial_println!("NVMe: I/O queue pair {} ready", IO_QID);

    let io = Queue {
        sq: &raw mut IO_SQ as *mut SqEntry,
        cq: &raw const IO_CQ as *const CqEntry,
        qd: IO_QD,
        sq_doorbell: 0x1000 + 2 * stride,
        cq_doorbell: 0x1000 + 3 * stride,
        tail: 0,
        head: 0,
        phase: 1,
    };

    Some(Controller {
        base,
        admin,
        io,
        data_buf_phys: data_phys,
    })
}

impl Controller {
    /// Read one 512-byte block from namespace 1 into `out`.
    pub fn read_block(&mut self, lba: u64, out: &mut [u8; BLOCK_SIZE]) -> Result<(), IoError> {
        let mut cmd = EMPTY_SQE;
        cmd.cdw0 = 0x02; // I/O Read
        cmd.nsid = 1;
        cmd.prp1 = self.data_buf_phys;
        cmd.cdw10 = lba as u32;
        cmd.cdw11 = (lba >> 32) as u32;
        // cdw12 NLB field is zero-based; 0 transfers 1 block.
        let cqe = submit(self.base, &mut self.io, cmd);
        if status_code(&cqe) != 0 {
            return Err(IoError(cqe.status));
        }
        unsafe {
            let buf = &(*(&raw const DATA_BUF)).0;
            out.copy_from_slice(&buf[..BLOCK_SIZE]);
        }
        Ok(())
    }

    /// Write one 512-byte block from `data` to namespace 1 at `lba`.
    pub fn write_block(&mut self, lba: u64, data: &[u8; BLOCK_SIZE]) -> Result<(), IoError> {
        unsafe {
            let buf = &mut (*(&raw mut DATA_BUF)).0;
            buf[..BLOCK_SIZE].copy_from_slice(data);
        }
        let mut cmd = EMPTY_SQE;
        cmd.cdw0 = 0x01; // I/O Write
        cmd.nsid = 1;
        cmd.prp1 = self.data_buf_phys;
        cmd.cdw10 = lba as u32;
        cmd.cdw11 = (lba >> 32) as u32;
        let cqe = submit(self.base, &mut self.io, cmd);
        if status_code(&cqe) != 0 {
            return Err(IoError(cqe.status));
        }
        Ok(())
    }

    // Silence dead-code warnings for the admin queue once it's no longer used
    // after init. Keeping it on the handle lets us add admin commands later
    // (Create Namespace, Get Log Page, etc.) without re-threading state.
    #[allow(dead_code)]
    fn admin(&mut self) -> &mut Queue {
        &mut self.admin
    }
}

fn translate(mapper: &impl Translate, vaddr: u64) -> u64 {
    mapper
        .translate_addr(VirtAddr::new(vaddr))
        .expect("kernel buffer is not mapped")
        .as_u64()
}
