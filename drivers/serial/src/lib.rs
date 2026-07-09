//! Serial console output.
//!
//! Two possible sinks, mirrored: the legacy COM1 port (QEMU `-serial
//! stdio`, dev boards with a real UART) and the Intel AMT
//! Serial-over-LAN "KT" UART that vPro machines expose as a PCI
//! function — the only console path on headless boxes like the
//! ThinkCentre Tiny, which have no legacy COM port at all.
//!
//! The driver is transmit-only and deliberately self-contained (no
//! external UART crate): the KT device is 16550-*compatible*, not a
//! verbatim 16550, so we avoid init sequences we cannot control
//! (loopback self-tests, modem-status flow-control gates) that would
//! misclassify it or drop output depending on SoL session state.
//!
//! Liveness rule: a sink must never stall the kernel. Every TX waits
//! on THR-empty with a bounded spin; a sink that times out repeatedly
//! is dropped for good. Absence is detected up front with a scratch-
//! register round trip, so probing a machine without the device is
//! harmless.

#![no_std]

mod fb;
pub use fb::PixelKind;

use core::fmt;
use fb::FbConsole;
use spin::Mutex;
use x86_64::instructions::port::Port;

// 16550 register indices. PIO adds them to the base port; MMIO scales
// them by the BAR's register stride.
const REG_THR: u16 = 0; // TX holding (write); DLL when DLAB=1
const REG_IER: u16 = 1; // interrupt enable; DLM when DLAB=1
const REG_FCR: u16 = 2; // FIFO control (write)
const REG_LCR: u16 = 3; // line control
const REG_MCR: u16 = 4; // modem control
const REG_LSR: u16 = 5; // line status
const REG_SPR: u16 = 7; // scratch — used for the presence probe

const LSR_THR_EMPTY: u8 = 1 << 5;

/// Spins allowed per byte while waiting for THR-empty. At any realistic
/// clock this is well past the drain time of a 16-byte FIFO, so hitting
/// the bound means nobody is draining (e.g. KT with no SoL session).
const TX_SPIN_LIMIT: u32 = 100_000;

/// Consecutive timed-out bytes before a sink is declared dead. Keeps a
/// drain-less UART from turning every log line into N × TX_SPIN_LIMIT
/// spins for the whole run.
const TX_STRIKE_LIMIT: u32 = 16;

/// Register access for the two bus flavors the KT BAR can take.
enum Regs {
    Pio { base: u16 },
    Mmio { base: *mut u8, stride: usize },
}

impl Regs {
    /// # Safety
    /// Caller guarantees the base (port or mapped MMIO) addresses a
    /// UART-compatible register file for the lifetime of the value.
    unsafe fn write(&mut self, reg: u16, value: u8) {
        match self {
            Regs::Pio { base } => unsafe { Port::new(*base + reg).write(value) },
            Regs::Mmio { base, stride } => unsafe {
                core::ptr::write_volatile(base.add(reg as usize * *stride), value)
            },
        }
    }

    /// # Safety
    /// Same contract as [`Regs::write`].
    unsafe fn read(&mut self, reg: u16) -> u8 {
        match self {
            Regs::Pio { base } => unsafe { Port::new(*base + reg).read() },
            Regs::Mmio { base, stride } => unsafe {
                core::ptr::read_volatile(base.add(reg as usize * *stride))
            },
        }
    }
}

struct Uart {
    regs: Regs,
    /// Consecutive TX timeouts; resets on any successful byte.
    strikes: u32,
    alive: bool,
}

impl Uart {
    /// Probe for a UART at `regs` and initialize it for 115200 8N1 TX.
    ///
    /// The scratch-register round trip rejects empty bus positions:
    /// reads from a missing PIO device float to 0xFF, and a missing
    /// MMIO device never echoes both patterns.
    ///
    /// # Safety
    /// `regs` must be safe to poke per [`Regs::write`]; a false-positive
    /// probe on a non-UART device would issue harmless register-sized
    /// writes to its BAR.
    unsafe fn probe(mut regs: Regs) -> Option<Uart> {
        unsafe {
            regs.write(REG_SPR, 0x55);
            if regs.read(REG_SPR) != 0x55 {
                return None;
            }
            regs.write(REG_SPR, 0xAA);
            if regs.read(REG_SPR) != 0xAA {
                return None;
            }

            // TX-only bring-up. Interrupts off — the kernel never takes
            // UART IRQs. Baud is nominal for SoL (the KT link is a
            // network stream; the divisor is ignored) and standard for
            // any physical terminal.
            regs.write(REG_IER, 0x00);
            regs.write(REG_LCR, 0x80); // DLAB=1
            regs.write(REG_THR, 0x01); // DLL: divisor 1 = 115200
            regs.write(REG_IER, 0x00); // DLM
            regs.write(REG_LCR, 0x03); // DLAB=0, 8N1
            regs.write(REG_FCR, 0x07); // FIFO enable + clear both
            regs.write(REG_MCR, 0x0B); // DTR | RTS | OUT2
        }
        Some(Uart {
            regs,
            strikes: 0,
            alive: true,
        })
    }

    fn put(&mut self, byte: u8) {
        if !self.alive {
            return;
        }
        let mut spins = 0u32;
        // Wait for THR-empty, but never unbounded: with no SoL session
        // the KT FIFO may simply not drain, and console output must not
        // become a boot hang.
        while unsafe { self.regs.read(REG_LSR) } & LSR_THR_EMPTY == 0 {
            spins += 1;
            if spins >= TX_SPIN_LIMIT {
                self.strikes += 1;
                if self.strikes >= TX_STRIKE_LIMIT {
                    self.alive = false;
                }
                return;
            }
            core::hint::spin_loop();
        }
        self.strikes = 0;
        unsafe { self.regs.write(REG_THR, byte) };
    }

    fn put_str(&mut self, s: &str) {
        for &b in s.as_bytes() {
            if b == b'\n' {
                self.put(b'\r');
            }
            self.put(b);
        }
    }
}

enum Com1 {
    /// Probed lazily on the first print so output works before any
    /// explicit init call.
    Unprobed,
    Present(Uart),
    Absent,
}

struct Console {
    com1: Com1,
    kt: Option<Uart>,
    fb: Option<FbConsole>,
}

// The MMIO variant holds a raw pointer, which is !Send by default. All
// register access happens under the CONSOLE mutex, which serializes
// ownership the same way the previous `Mutex<Uart16550Tty>` did.
unsafe impl Send for Console {}

static CONSOLE: Mutex<Console> = Mutex::new(Console {
    com1: Com1::Unprobed,
    kt: None,
    fb: None,
});

/// Legacy COM1 I/O port. Standard on every x86 PC that has the device;
/// machines without it (headless vPro boxes) fail the scratch probe.
const COM1_BASE: u16 = 0x3F8;

impl Console {
    fn write_str_all(&mut self, s: &str) {
        if let Com1::Unprobed = self.com1 {
            // Safety: COM1's port range is either a UART or unclaimed;
            // the probe handles both.
            self.com1 = match unsafe { Uart::probe(Regs::Pio { base: COM1_BASE }) } {
                Some(u) => Com1::Present(u),
                None => Com1::Absent,
            };
        }
        if let Com1::Present(u) = &mut self.com1 {
            u.put_str(s);
        }
        if let Some(u) = &mut self.kt {
            u.put_str(s);
        }
        if let Some(fb) = &mut self.fb {
            fb.put_str(s);
        }
    }
}

struct ConsoleWriter<'a>(&'a mut Console);

impl fmt::Write for ConsoleWriter<'_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.0.write_str_all(s);
        Ok(())
    }
}

/// Attach an AMT SoL KT UART behind an I/O BAR. Returns whether the
/// probe found a live UART.
///
/// # Safety
/// `port` must be the I/O BAR base of a 16550-compatible function whose
/// I/O decoding is enabled, and must stay valid for the rest of the run.
/// Attach a linear framebuffer as a console sink. Mirrors every line to
/// the video output, the only console that survives on boxes with no
/// serial port and unusable AMT SoL.
///
/// # Safety
/// `buf`/`len` must describe a valid, writable framebuffer mapping that
/// stays valid for the rest of the run, with the given `stride` (pixels
/// per scanline), `bpp` (bytes per pixel) and pixel `kind`.
pub unsafe fn install_fb(
    buf: *mut u8,
    len: usize,
    width: usize,
    height: usize,
    stride: usize,
    bpp: usize,
    kind: PixelKind,
    scale: usize,
) {
    let console = unsafe { FbConsole::new(buf, len, width, height, stride, bpp, kind, scale) };
    CONSOLE.lock().fb = Some(console);
}

pub unsafe fn install_kt_pio(port: u16) -> bool {
    let uart = unsafe { Uart::probe(Regs::Pio { base: port }) };
    let found = uart.is_some();
    CONSOLE.lock().kt = uart;
    found
}

/// Attach an AMT SoL KT UART behind a memory BAR.
///
/// # Safety
/// `base` must be a mapped, uncached-safe virtual pointer to the BAR of
/// a 16550-compatible function with memory decoding enabled, `stride`
/// its register spacing, both valid for the rest of the run.
pub unsafe fn install_kt_mmio(base: *mut u8, stride: usize) -> bool {
    let uart = unsafe { Uart::probe(Regs::Mmio { base, stride }) };
    let found = uart.is_some();
    CONSOLE.lock().kt = uart;
    found
}

#[doc(hidden)]
pub fn _print(args: core::fmt::Arguments) {
    use core::fmt::Write;
    // Writes cannot fail: dead sinks drop bytes instead of erroring, so
    // logging never becomes a second panic source inside the panic
    // handler.
    let mut console = CONSOLE.lock();
    let _ = ConsoleWriter(&mut console).write_fmt(args);
}

#[macro_export]
macro_rules! serial_print {
    ($($arg:tt)*) => {
        $crate::_print(format_args!($($arg)*));
    };
}

#[macro_export]
macro_rules! serial_println {
    () => ($crate::serial_print!("\n"));
    ($fmt:expr) => ($crate::serial_print!(concat!($fmt, "\n")));
    ($fmt:expr, $($arg:tt)*) => ($crate::serial_print!(
        concat!($fmt, "\n"), $($arg)*));
}
