//! NVMe controller initialization, admin/I/O queue management, and a
//! write-read round trip to verify end-to-end DMA.

use crate::{pci, serial_println};
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

struct Controller {
    base: *mut u8,
    admin: Queue,
    io: Option<Queue>,
}

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

/// Discover the NVMe controller, bring it up, create one I/O queue pair, and
/// round-trip a block to confirm DMA works end-to-end.
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
            return;
        }
        core::hint::spin_loop();
    }
    serial_println!("NVMe: controller enabled");

    let dstrd = ((cap >> 32) >> 16) as usize & 0xF;
    let stride = 4 << dstrd;

    let mut ctrl = Controller {
        base,
        admin: Queue {
            sq: &raw mut ADMIN_SQ as *mut SqEntry,
            cq: &raw const ADMIN_CQ as *const CqEntry,
            qd: ADMIN_QD,
            sq_doorbell: 0x1000,
            cq_doorbell: 0x1000 + stride,
            tail: 0,
            head: 0,
            phase: 1,
        },
        io: None,
    };

    // Identify Controller (CNS = 0x01).
    let mut cmd = EMPTY_SQE;
    cmd.cdw0 = 0x06;
    cmd.prp1 = id_phys;
    cmd.cdw10 = 0x01;
    let cqe = submit(ctrl.base, &mut ctrl.admin, cmd);
    if status_code(&cqe) != 0 {
        serial_println!("NVMe: Identify failed, status = {:#x}", cqe.status);
        return;
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
    let cqe = submit(ctrl.base, &mut ctrl.admin, cmd);
    if status_code(&cqe) != 0 {
        serial_println!("NVMe: Create I/O CQ failed, status = {:#x}", cqe.status);
        return;
    }

    // Create I/O Submission Queue (opcode 0x01).
    let mut cmd = EMPTY_SQE;
    cmd.cdw0 = 0x01;
    cmd.prp1 = iosq_phys;
    cmd.cdw10 = (((IO_QD - 1) as u32) << 16) | (IO_QID as u32);
    cmd.cdw11 = ((IO_QID as u32) << 16) | 1; // CQID in high half, PC=1
    let cqe = submit(ctrl.base, &mut ctrl.admin, cmd);
    if status_code(&cqe) != 0 {
        serial_println!("NVMe: Create I/O SQ failed, status = {:#x}", cqe.status);
        return;
    }
    serial_println!("NVMe: I/O queue pair {} ready", IO_QID);

    ctrl.io = Some(Queue {
        sq: &raw mut IO_SQ as *mut SqEntry,
        cq: &raw const IO_CQ as *const CqEntry,
        qd: IO_QD,
        sq_doorbell: 0x1000 + 2 * stride,
        cq_doorbell: 0x1000 + 3 * stride,
        tail: 0,
        head: 0,
        phase: 1,
    });

    // Fill DATA_BUF with a recognizable pattern, write it to LBA 0.
    unsafe {
        let buf = &mut (*(&raw mut DATA_BUF)).0;
        for (i, b) in buf.iter_mut().take(512).enumerate() {
            *b = ((i * 7 + 11) & 0xFF) as u8;
        }
    }

    let mut cmd = EMPTY_SQE;
    cmd.cdw0 = 0x01; // I/O Write
    cmd.nsid = 1;
    cmd.prp1 = data_phys;
    // CDW10/11 = starting LBA (0), CDW12 = (NLB - 1) = 0 means 1 block.
    let cqe = submit(ctrl.base, ctrl.io.as_mut().unwrap(), cmd);
    if status_code(&cqe) != 0 {
        serial_println!("NVMe: Write failed, status = {:#x}", cqe.status);
        return;
    }

    // Zero the buffer to make sure the read actually repopulates it.
    unsafe {
        let buf = &mut (*(&raw mut DATA_BUF)).0;
        for b in buf.iter_mut().take(512) {
            *b = 0;
        }
    }

    let mut cmd = EMPTY_SQE;
    cmd.cdw0 = 0x02; // I/O Read
    cmd.nsid = 1;
    cmd.prp1 = data_phys;
    let cqe = submit(ctrl.base, ctrl.io.as_mut().unwrap(), cmd);
    if status_code(&cqe) != 0 {
        serial_println!("NVMe: Read failed, status = {:#x}", cqe.status);
        return;
    }

    let ok = unsafe {
        let buf = &(*(&raw const DATA_BUF)).0;
        (0..512).all(|i| buf[i] == ((i * 7 + 11) & 0xFF) as u8)
    };
    if ok {
        serial_println!("NVMe: LBA 0 write/read round trip OK (512 bytes verified)");
    } else {
        serial_println!("NVMe: LBA 0 round trip MISMATCH");
    }
}

fn translate(mapper: &impl Translate, vaddr: u64) -> u64 {
    mapper
        .translate_addr(VirtAddr::new(vaddr))
        .expect("kernel buffer is not mapped")
        .as_u64()
}

