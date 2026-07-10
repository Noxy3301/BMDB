//! Build/boot orchestrator for BMDB.
//!
//! Wraps the kernel ELF into bootable artifacts and runs them:
//!
//!   hw-runner build   [--release] [--features <list>]
//!   hw-runner qemu    [--uefi] [--release] [--features <list>] [--fresh-disk]
//!   hw-runner deploy  [--tftp-root <dir>] [--release] [--features <list>]
//!   hw-runner hw-test [--tftp-root <dir>] [--timeout <secs>] [--release]
//!                     [--features <list>]
//!
//! `build` emits target/boot/{bmdb-bios.img, bmdb-uefi.img, tftp/}.
//! `qemu` boots the BIOS image by default (fast iteration) or the UEFI
//! image with OVMF (`--uefi`, the real-hardware-equivalent path).
//! `deploy` copies the PXE/TFTP folder into the dnsmasq TFTP root.
//! `hw-test` deploys, then PXE-boots the AMT-managed test box and
//! captures Serial-over-LAN until a success/failure marker or timeout.
//!
//! AMT credentials come from ~/.bmdb-amt.env (AMT_HOST=…,
//! AMT_PASSWORD=…); the file stays outside the repo.

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use bootloader::{BiosBoot, UefiBoot};

/// Serial line that every kernel mode prints on clean completion.
const SUCCESS_MARKER: &str = "It did not crash!";
/// Prefix of the panic handler's output.
const FAILURE_MARKER: &str = "panic: ";

const OVMF_CODE: &str = "/usr/share/OVMF/OVMF_CODE_4M.fd";
const OVMF_VARS: &str = "/usr/share/OVMF/OVMF_VARS_4M.fd";

struct Opts {
    command: String,
    release: bool,
    uefi: bool,
    pxe: bool,
    ci: bool,
    fresh_disk: bool,
    features: Option<String>,
    tftp_root: PathBuf,
    timeout: Duration,
}

fn parse_args() -> Result<Opts> {
    let mut args = std::env::args().skip(1);
    let command = args.next().context("missing subcommand")?;
    let mut opts = Opts {
        command,
        release: false,
        uefi: false,
        pxe: false,
        ci: false,
        fresh_disk: false,
        features: None,
        tftp_root: PathBuf::from("/srv/tftp"),
        timeout: Duration::from_secs(300),
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--release" => opts.release = true,
            "--uefi" => opts.uefi = true,
            "--pxe" => opts.pxe = true,
            "--ci" => opts.ci = true,
            "--fresh-disk" => opts.fresh_disk = true,
            "--features" => {
                opts.features = Some(args.next().context("--features needs a value")?)
            }
            "--tftp-root" => {
                opts.tftp_root = PathBuf::from(args.next().context("--tftp-root needs a value")?)
            }
            "--timeout" => {
                let secs: u64 = args
                    .next()
                    .context("--timeout needs a value")?
                    .parse()
                    .context("--timeout wants seconds")?;
                opts.timeout = Duration::from_secs(secs);
            }
            other => bail!("unknown flag: {other}"),
        }
    }
    Ok(opts)
}

/// Workspace root, derived from this crate's location (tools/hw-runner).
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("tools/hw-runner sits two levels below the workspace root")
        .to_path_buf()
}

/// Compile the kernel ELF and return its path.
fn build_kernel(opts: &Opts) -> Result<PathBuf> {
    let root = workspace_root();
    let mut cmd = Command::new("cargo");
    cmd.current_dir(&root)
        .args(["build", "--target", "x86_64-unknown-none", "-p", "bmdb-kernel"]);
    if opts.release {
        cmd.arg("--release");
    }
    if let Some(f) = &opts.features {
        cmd.args(["--features", f]);
    }
    let status = cmd.status().context("running cargo build")?;
    if !status.success() {
        bail!("kernel build failed");
    }
    let profile = if opts.release { "release" } else { "debug" };
    let elf = root
        .join("target/x86_64-unknown-none")
        .join(profile)
        .join("bmdb");
    if !elf.exists() {
        bail!("kernel ELF not found at {}", elf.display());
    }
    Ok(elf)
}

struct Artifacts {
    bios_img: PathBuf,
    uefi_img: PathBuf,
    tftp_dir: PathBuf,
}

/// Wrap the kernel ELF into BIOS + UEFI disk images and the PXE/TFTP
/// folder (bootloader as "bootloader", kernel as "kernel-x86_64").
fn build_images(kernel: &Path) -> Result<Artifacts> {
    let out = workspace_root().join("target/boot");
    fs::create_dir_all(&out)?;

    let artifacts = Artifacts {
        bios_img: out.join("bmdb-bios.img"),
        uefi_img: out.join("bmdb-uefi.img"),
        tftp_dir: out.join("tftp"),
    };

    BiosBoot::new(kernel)
        .create_disk_image(&artifacts.bios_img)
        .map_err(|e| anyhow!("bios image: {e:#}"))?;
    let uefi = UefiBoot::new(kernel);
    uefi.create_disk_image(&artifacts.uefi_img)
        .map_err(|e| anyhow!("uefi image: {e:#}"))?;
    uefi.create_pxe_tftp_folder(&artifacts.tftp_dir)
        .map_err(|e| anyhow!("pxe tftp folder: {e:#}"))?;

    println!("built: {}", artifacts.bios_img.display());
    println!("built: {}", artifacts.uefi_img.display());
    println!("built: {}/ (PXE: bootloader + kernel-x86_64)", artifacts.tftp_dir.display());
    Ok(artifacts)
}

/// The NVMe backing file QEMU attaches. 128 MiB of zeroes, recreated
/// on demand — its contents are scratch state, never a build input.
fn ensure_nvme_disk(fresh: bool) -> Result<PathBuf> {
    let disk = workspace_root().join("nvme.img");
    if fresh && disk.exists() {
        fs::remove_file(&disk)?;
    }
    if !disk.exists() {
        let f = fs::File::create(&disk)?;
        f.set_len(128 * 1024 * 1024)?;
        println!("created fresh {}", disk.display());
    }
    Ok(disk)
}

fn run_qemu(opts: &Opts, artifacts: &Artifacts) -> Result<()> {
    let disk = ensure_nvme_disk(opts.fresh_disk)?;
    let mut cmd = Command::new("qemu-system-x86_64");
    cmd.current_dir(workspace_root());

    // Both the UEFI-disk and PXE-rehearsal paths boot through OVMF
    // firmware; the BIOS-disk path uses QEMU's built-in SeaBIOS.
    if opts.uefi || opts.pxe {
        // OVMF wants two pflash chips: read-only code and writable
        // NVRAM vars. Vars are copied per-run so UEFI boot-entry writes
        // never leak between runs.
        let vars_copy = workspace_root().join("target/boot/ovmf-vars.fd");
        fs::copy(OVMF_VARS, &vars_copy)
            .with_context(|| format!("copying {OVMF_VARS} (is ovmf installed?)"))?;
        cmd.arg("-drive")
            .arg(format!("if=pflash,format=raw,readonly=on,file={OVMF_CODE}"));
        cmd.arg("-drive")
            .arg(format!("if=pflash,format=raw,file={}", vars_copy.display()));
    }

    if opts.pxe {
        // Netboot rehearsal: no boot disk. QEMU's user-net stack serves
        // the real tftp/ folder, so OVMF's PXE ROM fetches the exact
        // "bootloader" + "kernel-x86_64" set we would deploy to dnsmasq.
        cmd.arg("-netdev").arg(format!(
            "user,id=net0,tftp={},bootfile=bootloader",
            artifacts.tftp_dir.display()
        ));
        cmd.args(["-device", "virtio-net-pci,netdev=net0"]);
    } else if opts.uefi {
        cmd.arg("-drive")
            .arg(format!("format=raw,file={}", artifacts.uefi_img.display()));
    } else {
        cmd.arg("-drive")
            .arg(format!("format=raw,file={}", artifacts.bios_img.display()));
    }

    // KVM + host CPU passthrough. The silo-bench numbers without this
    // flag carry a ~17x TCG emulation penalty that drowns OCC hot
    // paths — keep it on by default. Requires `/dev/kvm` access; if
    // `Could not access KVM kernel module: Permission denied` fires,
    // `sudo usermod -aG kvm $USER` then log out and back in.
    // -smp from BMDB_SMP (default 4) so a scaling sweep can vary the vCPU
    // count without a rebuild; +invtsc exposes an invariant TSC (and the KVM
    // TSC-frequency leaf) so the guest can convert cycles to seconds under KVM.
    let smp = std::env::var("BMDB_SMP").unwrap_or_else(|_| "4".to_string());
    cmd.args(["-serial", "stdio", "-display", "none", "-smp", &smp]);
    cmd.args(["-enable-kvm", "-cpu", "host,+invtsc"]);
    cmd.arg("-drive")
        .arg(format!("file={},format=raw,if=none,id=nvme0", disk.display()));
    cmd.args(["-device", "nvme,serial=BMDB0001,drive=nvme0"]);

    if opts.ci {
        // The kernel hlt-loops after its final line, so QEMU never
        // exits on its own — stop as soon as a marker appears.
        if run_qemu_ci(cmd, opts.timeout)? {
            Ok(())
        } else {
            bail!("QEMU CI run did not reach the success marker");
        }
    } else {
        let status = cmd.status().context("running qemu-system-x86_64")?;
        if !status.success() {
            bail!("qemu exited with {status}");
        }
        Ok(())
    }
}

/// Run QEMU headless, tee its serial output to our stdout, and return
/// as soon as the success or failure marker lands (or the timeout
/// fires). QEMU is killed on every path since the kernel never halts
/// the machine.
fn run_qemu_ci(mut cmd: Command, timeout: Duration) -> Result<bool> {
    use std::sync::mpsc;

    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::inherit());
    let mut child = cmd.spawn().context("running qemu-system-x86_64")?;
    let stdout = child.stdout.take().unwrap();

    // A reader thread scans serial lines so the main thread can enforce
    // a wall-clock deadline even if the guest goes completely silent
    // (the signature of an early boot failure).
    let (tx, rx) = mpsc::channel::<Option<bool>>();
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };
            println!("QEMU| {line}");
            if line.contains(SUCCESS_MARKER) {
                let _ = tx.send(Some(true));
                return;
            }
            if line.contains(FAILURE_MARKER) {
                let _ = tx.send(Some(false));
                return;
            }
        }
        // Stream closed before any marker: QEMU exited or died.
        let _ = tx.send(None);
    });

    let outcome = rx.recv_timeout(timeout);
    let _ = child.kill();
    let _ = child.wait();
    let _ = reader.join();

    match outcome {
        Ok(Some(true)) => {
            println!("CI PASS (marker: {SUCCESS_MARKER:?})");
            Ok(true)
        }
        Ok(Some(false)) => {
            println!("CI FAIL (kernel panicked)");
            Ok(false)
        }
        Ok(None) => {
            println!("CI FAIL (QEMU exited before any marker)");
            Ok(false)
        }
        Err(_) => {
            println!("CI TIMEOUT after {timeout:?} (no marker — likely an early boot failure)");
            Ok(false)
        }
    }
}

/// Copy the PXE/TFTP folder contents into the dnsmasq TFTP root.
fn deploy(opts: &Opts, artifacts: &Artifacts) -> Result<()> {
    if !opts.tftp_root.is_dir() {
        bail!(
            "TFTP root {} does not exist — set up dnsmasq first (see tools/pxe/README.md)",
            opts.tftp_root.display()
        );
    }
    for entry in fs::read_dir(&artifacts.tftp_dir)? {
        let entry = entry?;
        let dest = opts.tftp_root.join(entry.file_name());
        fs::copy(entry.path(), &dest).with_context(|| {
            format!(
                "copying {} to {} (does your user have write access?)",
                entry.path().display(),
                dest.display()
            )
        })?;
        println!("deployed: {}", dest.display());
    }
    Ok(())
}

struct AmtEnv {
    host: String,
    password: String,
}

/// AMT credentials live in ~/.bmdb-amt.env (mode 600), never in the repo.
fn read_amt_env() -> Result<AmtEnv> {
    let home = std::env::var("HOME").context("HOME not set")?;
    let path = Path::new(&home).join(".bmdb-amt.env");
    let content = fs::read_to_string(&path)
        .with_context(|| format!("reading {} (create it: AMT_HOST=…, AMT_PASSWORD=…)", path.display()))?;
    let mut host = None;
    let mut password = None;
    for line in content.lines() {
        if let Some(v) = line.strip_prefix("AMT_HOST=") {
            host = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("AMT_PASSWORD=") {
            password = Some(v.trim().to_string());
        }
    }
    Ok(AmtEnv {
        host: host.context("AMT_HOST missing from ~/.bmdb-amt.env")?,
        password: password.context("AMT_PASSWORD missing from ~/.bmdb-amt.env")?,
    })
}

/// The WS-Management helper that drives AMT power + the SoL listener.
/// (`amttool` cannot drive AMT 12; this wraps the python `amt` client.)
fn bmdb_amt_path() -> PathBuf {
    workspace_root().join("tools/amt/bmdb-amt.py")
}

/// Run one `bmdb-amt.py <cmd>` and return its trimmed stdout. Credentials
/// come from ~/.bmdb-amt.env, which the script reads itself. Wrapped in
/// `timeout` as a hard backstop so a wedged management-engine call can
/// never hang the gate even if the script's own network timeout fails.
fn bmdb_amt(cmd: &str) -> Result<String> {
    let out = Command::new("timeout")
        // -k: SIGKILL 5s after the SIGTERM if the child ignores it.
        .args(["-k", "5", "30"])
        .arg(bmdb_amt_path())
        .arg(cmd)
        .output()
        .with_context(|| format!("running bmdb-amt.py {cmd}"))?;
    if !out.status.success() {
        // `timeout` exits 124 when it has to kill the child.
        let code = out.status.code().unwrap_or(-1);
        bail!(
            "bmdb-amt.py {cmd} failed (exit {code}{}): {}",
            if code == 124 { ", timed out" } else { "" },
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Path to amtterm 1.7 (`$AMT_TERM`, else ~/.local/bin/amtterm17). The
/// distro amtterm 1.4 cannot authenticate to AMT 12, so there is no
/// fallback to it.
fn amtterm_path() -> Result<String> {
    if let Ok(p) = std::env::var("AMT_TERM") {
        return Ok(p);
    }
    let home = std::env::var("HOME").unwrap_or_default();
    let p = format!("{home}/.local/bin/amtterm17");
    if Path::new(&p).exists() {
        return Ok(p);
    }
    bail!(
        "amtterm 1.7 not found at {p}; set $AMT_TERM or build it (see \
         tools/pxe/README.md). The distro amtterm 1.4 cannot auth to AMT 12."
    )
}

/// Open a Serial-over-LAN session and watch it until a success/failure
/// marker or the deadline. A reader thread feeds a channel so the
/// deadline holds even when the link stays completely silent — which is
/// the normal state until BMDB's own kernel starts driving the KT UART
/// (firmware POST and the UEFI PXE ROM do not print on SoL).
fn capture_sol(amt: &AmtEnv, amtterm: &str, timeout: Duration, log_path: &Path) -> Result<bool> {
    use std::io::Write;
    use std::sync::mpsc;

    let mut child = Command::new(amtterm)
        .arg(&amt.host)
        .env("AMT_PASSWORD", &amt.password)
        // Hold stdin open: amtterm exits on stdin EOF, so a closed or null
        // stdin would drop the session immediately.
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| format!("spawning {amtterm}"))?;

    let stdin = child.stdin.take(); // keep the handle alive for the child's lifetime
    let stdout = child.stdout.take().unwrap();
    let log_path_buf = log_path.to_path_buf();

    // The reader thread mirrors every line to stdout and the log, and
    // signals the outcome on the first marker. A silent link never sends,
    // so the main thread's recv_timeout enforces the deadline.
    let (tx, rx) = mpsc::channel::<bool>();
    let reader = std::thread::spawn(move || {
        let mut log = fs::File::create(&log_path_buf).ok();
        for line in BufReader::new(stdout).lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };
            println!("SOL| {line}");
            if let Some(f) = log.as_mut() {
                let _ = writeln!(f, "{line}");
            }
            if line.contains(SUCCESS_MARKER) {
                let _ = tx.send(true);
                return;
            }
            if line.contains(FAILURE_MARKER) {
                let _ = tx.send(false);
                return;
            }
        }
    });

    let outcome = rx.recv_timeout(timeout);
    let _ = child.kill();
    let _ = child.wait();
    drop(stdin);
    let _ = reader.join();

    match outcome {
        Ok(true) => {
            println!("HW-TEST PASS (marker: {SUCCESS_MARKER:?})");
            Ok(true)
        }
        Ok(false) => {
            println!("HW-TEST FAIL (kernel panicked — see {})", log_path.display());
            Ok(false)
        }
        Err(_) => {
            println!(
                "HW-TEST TIMEOUT/silent after {timeout:?} — no marker on SoL. \
                 Check the dnsmasq log: did the M920q DHCP and TFTP-fetch? See {}",
                log_path.display()
            );
            Ok(false)
        }
    }
}

fn hw_test(opts: &Opts, artifacts: &Artifacts) -> Result<()> {
    let amt = read_amt_env()?;
    let amtterm = amtterm_path()?;
    deploy(opts, artifacts)?;

    // Make sure the SoL listener is on (idempotent WS-Man Put).
    println!("AMT: {}", bmdb_amt("sol-enable")?);
    let state = bmdb_amt("status")?;
    println!("AMT: power state = {state}");

    // Force PXE for the next boot, then power the box: on from off, reset
    // if already running (a power cycle is rejected while SoL is open).
    let power = std::thread::spawn(move || -> Result<()> {
        // Let the SoL session attach first so no early output is missed.
        std::thread::sleep(Duration::from_secs(2));
        bmdb_amt("pxe-next")?;
        let cmd = if state == "off" { "on" } else { "reset" };
        println!("AMT: {}", bmdb_amt(cmd)?);
        Ok(())
    });

    let log_path = workspace_root().join("target/boot/hw-test.log");
    println!("AMT: opening SoL to {} (via {amtterm}) …", amt.host);
    let passed = capture_sol(&amt, &amtterm, opts.timeout, &log_path)?;

    power.join().map_err(|_| anyhow!("power thread panicked"))??;
    if !passed {
        bail!("hardware test did not pass");
    }
    Ok(())
}

fn main() -> Result<()> {
    let opts = parse_args()?;
    match opts.command.as_str() {
        "build" => {
            let kernel = build_kernel(&opts)?;
            build_images(&kernel)?;
        }
        "qemu" => {
            let kernel = build_kernel(&opts)?;
            let artifacts = build_images(&kernel)?;
            run_qemu(&opts, &artifacts)?;
        }
        "deploy" => {
            let kernel = build_kernel(&opts)?;
            let artifacts = build_images(&kernel)?;
            deploy(&opts, &artifacts)?;
        }
        "hw-test" => {
            let kernel = build_kernel(&opts)?;
            let artifacts = build_images(&kernel)?;
            hw_test(&opts, &artifacts)?;
        }
        other => bail!("unknown subcommand: {other} (build|qemu|deploy|hw-test)"),
    }
    Ok(())
}
