//! RAM-backed `BlockStorage` for host-side unit tests. Unwritten blocks read
//! back as zeros; decoders that check a magic word therefore see them as
//! invalid, which is enough for WAL end-of-log detection. This is not a
//! faithful model of arbitrary fresh NVMe media — real devices may return
//! any byte pattern until first write — but it matches both the `qemu-img
//! create -f raw` images we use for the smoke test and the common post-TRIM
//! zero-return behavior.

use crate::lba_alloc::{BLOCK_SIZE, Lba};
use crate::storage::BlockStorage;
use std::collections::HashMap;
use std::convert::Infallible;

pub struct MemStorage {
    blocks: HashMap<Lba, [u8; BLOCK_SIZE]>,
    flush_count: usize,
}

impl MemStorage {
    pub fn new() -> Self {
        Self {
            blocks: HashMap::new(),
            flush_count: 0,
        }
    }

    pub fn flush_count(&self) -> usize {
        self.flush_count
    }

    /// Overwrite a block outside of the normal API. Useful for injecting
    /// corruption or bogus records to test recovery stopping conditions.
    pub fn force_write(&mut self, lba: Lba, data: [u8; BLOCK_SIZE]) {
        self.blocks.insert(lba, data);
    }
}

impl Default for MemStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockStorage for MemStorage {
    type Error = Infallible;

    fn read_block(&mut self, lba: Lba, out: &mut [u8; BLOCK_SIZE]) -> Result<(), Self::Error> {
        match self.blocks.get(&lba) {
            Some(data) => out.copy_from_slice(data),
            None => out.fill(0),
        }
        Ok(())
    }

    fn write_block(&mut self, lba: Lba, data: &[u8; BLOCK_SIZE]) -> Result<(), Self::Error> {
        self.blocks.insert(lba, *data);
        Ok(())
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        self.flush_count += 1;
        Ok(())
    }
}
