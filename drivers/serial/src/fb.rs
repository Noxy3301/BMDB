//! Framebuffer text console.
//!
//! Renders the boot log to the linear framebuffer the bootloader hands
//! us. On machines whose only usable output is the video port — no
//! serial header, and AMT Serial-over-LAN dies the moment a driverless
//! OS takes the shared NIC — this is the one console that survives, read
//! back off the video output with a capture card.
//!
//! An 8x8 bitmap font (`font8x8`) keeps the glyph table tiny and needs no
//! allocation. All writes are bounds-checked against the reported byte
//! length, so a wrong stride/format from the firmware garbles the image
//! but never faults.

use font8x8::legacy::BASIC_LEGACY;

const GLYPH: usize = 8;

/// Byte order of one pixel, mirrored from the bootloader's `PixelFormat`
/// so this driver keeps no dependency on `bootloader_api`.
#[derive(Clone, Copy)]
pub enum PixelKind {
    Rgb,
    Bgr,
    Gray,
}

/// A linear-framebuffer character console with a fixed 8x8 font, each
/// glyph pixel drawn as a `scale`x`scale` block so text stays legible on
/// high-resolution panels captured and downscaled off the video port.
pub struct FbConsole {
    buf: *mut u8,
    len: usize,
    width: usize,
    height: usize,
    stride: usize, // pixels per scanline (may exceed width)
    bpp: usize,    // bytes per pixel
    kind: PixelKind,
    scale: usize,
    cell: usize, // GLYPH * scale: pixel size of one character cell
    col: usize,
    row: usize,
    cols: usize,
    rows: usize,
}

impl FbConsole {
    /// # Safety
    /// `buf`/`len` must describe a valid, writable framebuffer mapping
    /// that outlives the run; `stride`/`bpp`/`kind` must match its layout.
    pub unsafe fn new(
        buf: *mut u8,
        len: usize,
        width: usize,
        height: usize,
        stride: usize,
        bpp: usize,
        kind: PixelKind,
        scale: usize,
    ) -> Self {
        let scale = scale.max(1);
        let cell = GLYPH * scale;
        let mut c = FbConsole {
            buf,
            len,
            width,
            height,
            stride,
            bpp,
            kind,
            scale,
            cell,
            col: 0,
            row: 0,
            cols: width / cell,
            rows: height / cell,
        };
        c.clear();
        c
    }

    fn clear(&mut self) {
        // Safety: `len` is the framebuffer's own reported byte length.
        unsafe { core::ptr::write_bytes(self.buf, 0, self.len) };
    }

    fn put_pixel(&mut self, x: usize, y: usize, on: bool) {
        if x >= self.width || y >= self.height {
            return;
        }
        // Bytes this pixel actually touches: one for grayscale, otherwise
        // three colour bytes plus an optional padding byte. The bound check
        // uses this span, not `bpp`, so a format with `bpp < 3` cannot let a
        // colour write run past the buffer.
        let span = match self.kind {
            PixelKind::Gray => 1,
            _ if self.bpp >= 4 => 4,
            _ => 3,
        };
        // Checked so a malformed (huge) stride/bpp cannot wrap the offset
        // back into a value that passes the length test.
        let offset = match y
            .checked_mul(self.stride)
            .and_then(|v| v.checked_add(x))
            .and_then(|v| v.checked_mul(self.bpp))
        {
            Some(o) => o,
            None => return,
        };
        if offset.checked_add(span).map_or(true, |end| end > self.len) {
            return;
        }
        // Safety: `offset + span` was just proven `<= len`; the framebuffer
        // is device memory, so writes go straight through with no aliasing.
        unsafe {
            let p = self.buf.add(offset);
            let v = if on { 0xFF } else { 0x00 };
            match self.kind {
                PixelKind::Gray => p.write_volatile(v),
                _ => {
                    p.write_volatile(v);
                    p.add(1).write_volatile(v);
                    p.add(2).write_volatile(v);
                    if span == 4 {
                        p.add(3).write_volatile(0);
                    }
                }
            }
        }
    }

    fn draw_glyph(&mut self, ch: char) {
        let idx = ch as usize;
        let glyph = if idx < 128 { BASIC_LEGACY[idx] } else { [0; 8] };
        let ox = self.col * self.cell;
        let oy = self.row * self.cell;
        for (dy, bits) in glyph.iter().enumerate() {
            for dx in 0..GLYPH {
                // font8x8 packs each row LSB-first: bit 0 is the leftmost
                // column. Expand each font pixel to a scale x scale block.
                let on = (bits >> dx) & 1 != 0;
                for sy in 0..self.scale {
                    for sx in 0..self.scale {
                        self.put_pixel(
                            ox + dx * self.scale + sx,
                            oy + dy * self.scale + sy,
                            on,
                        );
                    }
                }
            }
        }
    }

    fn newline(&mut self) {
        self.col = 0;
        if self.row + 1 >= self.rows {
            self.scroll();
        } else {
            self.row += 1;
        }
    }

    /// Shift the image up one text row and clear the freed band.
    fn scroll(&mut self) {
        // Saturating so malformed geometry overflows to a value the guard
        // below rejects rather than wrapping into a bad length.
        let line = self.stride.saturating_mul(self.bpp);
        let shift = self.cell.saturating_mul(line);
        let used = self
            .rows
            .saturating_mul(self.cell)
            .saturating_mul(line)
            .min(self.len);
        if shift == 0 || shift >= used {
            self.clear();
            return;
        }
        // Safety: shift < used <= len, so the moved region and the cleared
        // tail both lie within the buffer; regions overlap forward, so
        // `copy` (memmove) is used.
        unsafe {
            core::ptr::copy(self.buf.add(shift), self.buf, used - shift);
            core::ptr::write_bytes(self.buf.add(used - shift), 0, shift);
        }
    }

    pub fn put_str(&mut self, s: &str) {
        for ch in s.chars() {
            match ch {
                '\n' => self.newline(),
                '\r' => self.col = 0,
                _ => {
                    if self.col >= self.cols {
                        self.newline();
                    }
                    self.draw_glyph(ch);
                    self.col += 1;
                }
            }
        }
    }
}
