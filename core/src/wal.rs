//! Write-Ahead Log.
//!
//! Phase 3 layout: one record per 512-byte block, appended sequentially from
//! `lba_alloc::WAL_START`. Each record carries a magic, monotonic LSN, Silo
//! epoch, operation, fixed-size key and value, and a checksum covering the
//! rest of the record.
//!
//! Recovery scans from `WAL_START` and stops at the first block with a bad
//! magic or a checksum mismatch — those indicate an unwritten block or a
//! torn write. Since one record is exactly one NVMe block, a torn write
//! corrupts only its own record, never a previous one.
//!
//! The checksum is FNV-1a (32-bit). It is a correctness placeholder; a real
//! WAL wants CRC32C both for stronger detection and for hardware acceleration.

use crate::lba_alloc::{BLOCK_SIZE, Lba, WAL_START, wal_end};
use crate::storage::BlockStorage;

pub const WAL_MAGIC: u64 = 0x424D_4442_5741_4C30; // "BMDBWAL0"

pub type Key = [u8; 8];
pub type Value = [u8; 8];

/// Operation captured in a WAL record.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Put = 1,
    Delete = 2,
}

impl Op {
    pub fn from_u32(v: u32) -> Option<Op> {
        match v {
            1 => Some(Op::Put),
            2 => Some(Op::Delete),
            _ => None,
        }
    }
}

/// One WAL record. Layout is stable (`repr(C)`) so on-disk bytes can be
/// reinterpreted as this struct via `core::mem::transmute`-equivalent copies.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Record {
    pub magic: u64,
    pub lsn: u64,
    pub epoch: u64,
    pub op: u32,
    _pad0: u32,
    pub key: Key,
    pub value: Value,
    pub checksum: u32,
    _pad1: u32,
}

impl Record {
    const SIZE: usize = core::mem::size_of::<Self>();
    const CHECKSUM_OFFSET: usize = 48;

    pub fn new(op: Op, lsn: u64, epoch: u64, key: Key, value: Value) -> Self {
        let mut rec = Self {
            magic: WAL_MAGIC,
            lsn,
            epoch,
            op: op as u32,
            _pad0: 0,
            key,
            value,
            checksum: 0,
            _pad1: 0,
        };
        rec.checksum = rec.compute_checksum();
        rec
    }

    pub fn op(&self) -> Option<Op> {
        Op::from_u32(self.op)
    }

    pub fn is_valid(&self) -> bool {
        self.magic == WAL_MAGIC && self.checksum == self.compute_checksum()
    }

    fn compute_checksum(&self) -> u32 {
        let bytes =
            unsafe { core::slice::from_raw_parts(self as *const Self as *const u8, Self::SIZE) };
        // FNV-1a over every byte except the 4-byte checksum field itself.
        let mut h: u32 = 0x811c_9dc5;
        for (i, &b) in bytes.iter().enumerate() {
            if i >= Self::CHECKSUM_OFFSET && i < Self::CHECKSUM_OFFSET + 4 {
                continue;
            }
            h ^= b as u32;
            h = h.wrapping_mul(0x0100_0193);
        }
        h
    }
}

// Layout assumptions the checksum code relies on.
const _: () = assert!(Record::SIZE == 56);
const _: () = assert!(Record::SIZE <= BLOCK_SIZE);
const _: () = assert!(core::mem::offset_of!(Record, checksum) == Record::CHECKSUM_OFFSET);

fn encode(rec: &Record, block: &mut [u8; BLOCK_SIZE]) {
    block.fill(0);
    let bytes =
        unsafe { core::slice::from_raw_parts(rec as *const Record as *const u8, Record::SIZE) };
    block[..bytes.len()].copy_from_slice(bytes);
}

fn decode(block: &[u8; BLOCK_SIZE]) -> Record {
    let mut rec: Record = unsafe { core::mem::zeroed() };
    let bytes = unsafe {
        core::slice::from_raw_parts_mut(&mut rec as *mut Record as *mut u8, Record::SIZE)
    };
    bytes.copy_from_slice(&block[..bytes.len()]);
    rec
}

/// In-memory WAL cursor. Holds the LBA that the next append will land on and
/// the next LSN to assign. Not persisted — recovery rebuilds this state from
/// the records on disk.
pub struct Wal {
    next_lba: Lba,
    next_lsn: u64,
}

impl Wal {
    pub const fn new() -> Self {
        Self {
            next_lba: WAL_START,
            next_lsn: 1,
        }
    }

    pub fn next_lba(&self) -> Lba {
        self.next_lba
    }

    pub fn next_lsn(&self) -> u64 {
        self.next_lsn
    }

    /// Append one record. The call flushes before returning, so the record is
    /// durable by the time the caller sees the LSN. Group-commit (one flush
    /// per epoch) is a Silo-era optimization.
    ///
    /// Panics if the WAL region is full — a circular log / checkpoint +
    /// truncate story is a later phase problem.
    pub fn append<S: BlockStorage>(
        &mut self,
        storage: &mut S,
        op: Op,
        epoch: u64,
        key: Key,
        value: Value,
    ) -> Result<u64, S::Error> {
        assert!(self.next_lba <= wal_end(), "WAL region is full");
        let rec = Record::new(op, self.next_lsn, epoch, key, value);
        let mut block = [0u8; BLOCK_SIZE];
        encode(&rec, &mut block);
        storage.write_block(self.next_lba, &block)?;
        storage.flush()?;
        let lsn = self.next_lsn;
        self.next_lba += 1;
        self.next_lsn += 1;
        Ok(lsn)
    }

    /// Rebuild the WAL cursor by scanning from the start of the region. Stops
    /// at the first block that fails magic or checksum — that's end-of-log.
    /// Because each record occupies its own block, a torn write corrupts
    /// exactly one record and leaves all preceding records intact.
    pub fn recover<S: BlockStorage>(storage: &mut S) -> Result<Self, S::Error> {
        let mut next_lba = WAL_START;
        let mut next_lsn = 1u64;
        while next_lba <= wal_end() {
            match Self::read_at(storage, next_lba)? {
                Some(rec) => {
                    next_lsn = rec.lsn + 1;
                    next_lba += 1;
                }
                None => break,
            }
        }
        Ok(Self { next_lba, next_lsn })
    }

    /// Read the record at `lba`. Returns `None` when the block is not a valid
    /// record — either the block was never written, or a torn write corrupted
    /// it. Recovery treats either case as end-of-log.
    pub fn read_at<S: BlockStorage>(
        storage: &mut S,
        lba: Lba,
    ) -> Result<Option<Record>, S::Error> {
        let mut block = [0u8; BLOCK_SIZE];
        storage.read_block(lba, &mut block)?;
        let rec = decode(&block);
        Ok(if rec.is_valid() { Some(rec) } else { None })
    }
}

impl Default for Wal {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lba_alloc::wal_end;
    use crate::mem_storage::MemStorage;

    fn sample_record() -> Record {
        Record::new(Op::Put, 7, 3, *b"key00001", *b"val00001")
    }

    #[test]
    fn record_roundtrip_through_block() {
        let rec = sample_record();
        let mut block = [0u8; BLOCK_SIZE];
        encode(&rec, &mut block);
        let back = decode(&block);
        assert!(back.is_valid());
        assert_eq!(back.lsn, rec.lsn);
        assert_eq!(back.epoch, rec.epoch);
        assert_eq!(back.op(), Some(Op::Put));
        assert_eq!(back.key, rec.key);
        assert_eq!(back.value, rec.value);
    }

    #[test]
    fn fresh_record_is_valid() {
        assert!(sample_record().is_valid());
    }

    #[test]
    fn bit_flip_in_payload_detected() {
        let rec = sample_record();
        let mut block = [0u8; BLOCK_SIZE];
        encode(&rec, &mut block);
        // Corrupt a byte inside the key field.
        block[40] ^= 0x01;
        let back = decode(&block);
        assert!(!back.is_valid(), "corruption must invalidate the checksum");
    }

    #[test]
    fn zeroed_block_is_invalid() {
        let back = decode(&[0u8; BLOCK_SIZE]);
        assert!(!back.is_valid(), "unwritten block must not look like a record");
    }

    #[test]
    fn append_assigns_sequential_lsns() {
        let mut storage = MemStorage::new();
        let mut wal = Wal::new();

        let lsn0 = wal
            .append(&mut storage, Op::Put, 0, *b"a\0\0\0\0\0\0\0", *b"A\0\0\0\0\0\0\0")
            .unwrap();
        let lsn1 = wal
            .append(&mut storage, Op::Put, 0, *b"b\0\0\0\0\0\0\0", *b"B\0\0\0\0\0\0\0")
            .unwrap();

        assert_eq!(lsn0, 1);
        assert_eq!(lsn1, 2);
        assert_eq!(wal.next_lsn(), 3);
        assert_eq!(wal.next_lba(), WAL_START + 2);
        // One flush per append is the Phase 3 policy (durable-before-return).
        assert_eq!(storage.flush_count(), 2);
    }

    #[test]
    fn read_at_returns_appended_record() {
        let mut storage = MemStorage::new();
        let mut wal = Wal::new();

        let _ = wal
            .append(&mut storage, Op::Delete, 9, *b"key_____", [0; 8])
            .unwrap();

        let rec = Wal::read_at(&mut storage, WAL_START).unwrap().unwrap();
        assert_eq!(rec.lsn, 1);
        assert_eq!(rec.epoch, 9);
        assert_eq!(rec.op(), Some(Op::Delete));
        assert_eq!(rec.key, *b"key_____");
    }

    #[test]
    fn recover_on_empty_storage_is_empty_wal() {
        let mut storage = MemStorage::new();
        let wal = Wal::recover(&mut storage).unwrap();
        assert_eq!(wal.next_lba(), WAL_START);
        assert_eq!(wal.next_lsn(), 1);
    }

    #[test]
    fn recover_restores_cursor_after_appends() {
        let mut storage = MemStorage::new();
        let mut wal = Wal::new();
        for i in 1..=5 {
            wal.append(&mut storage, Op::Put, 0, (i as u64).to_be_bytes(), [0; 8])
                .unwrap();
        }
        let recovered = Wal::recover(&mut storage).unwrap();
        assert_eq!(recovered.next_lsn(), 6);
        assert_eq!(recovered.next_lba(), WAL_START + 5);
    }

    #[test]
    fn recover_stops_at_first_invalid_block() {
        let mut storage = MemStorage::new();
        let mut wal = Wal::new();
        for i in 1..=3 {
            wal.append(&mut storage, Op::Put, 0, (i as u64).to_be_bytes(), [0; 8])
                .unwrap();
        }
        // Clobber the fourth slot with garbage.
        let mut garbage = [0u8; BLOCK_SIZE];
        garbage[0] = 0x55;
        storage.force_write(WAL_START + 3, garbage);

        // Place a well-formed record *after* the corrupt slot. If recovery
        // naively scanned past the first invalid block, it would see this
        // record and advance `next_lba` past it. Stopping at the corrupt
        // block means this record stays unreached.
        let past_gap = Record::new(Op::Put, 99, 0, [0xBB; 8], [0; 8]);
        let mut past_block = [0u8; BLOCK_SIZE];
        encode(&past_gap, &mut past_block);
        storage.force_write(WAL_START + 4, past_block);

        let recovered = Wal::recover(&mut storage).unwrap();
        assert_eq!(recovered.next_lba(), WAL_START + 3);
        assert_eq!(recovered.next_lsn(), 4);

        // Double-check: the record past the gap really is valid on its own,
        // so the test is probing recovery behavior, not record construction.
        let probe = Wal::read_at(&mut storage, WAL_START + 4).unwrap();
        assert!(probe.is_some(), "record after gap must be intact");
    }

    #[test]
    fn record_size_matches_checksum_offset() {
        // Guard against accidental layout drift that would invalidate the
        // checksum loop's `CHECKSUM_OFFSET` constant.
        assert_eq!(core::mem::size_of::<Record>(), 56);
        assert_eq!(
            core::mem::offset_of!(Record, checksum),
            Record::CHECKSUM_OFFSET
        );
    }

    #[test]
    fn region_end_constants_are_self_consistent() {
        assert_eq!(wal_end(), WAL_START + crate::lba_alloc::WAL_LEN - 1);
    }
}
