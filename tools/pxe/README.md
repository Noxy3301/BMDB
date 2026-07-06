# BMDB real-hardware PXE + AMT test loop

Runbook for booting BMDB on the ThinkCentre M920q over the network and
capturing its output, with no human at the box. The dev host serves the
boot image; Intel AMT (vPro) drives power and the serial console.

```
 dev host (192.168.0.222, eno2)                 M920q (AMT 192.168.0.2)
 ┌─────────────────────────────┐                ┌──────────────────────┐
 │ dnsmasq  proxyDHCP + TFTP    │──DHCP/PXE────▶ │ UEFI PXE ROM         │
 │   serves /srv/tftp/bootloader│──TFTP────────▶ │  → bootloader        │
 │   + kernel-x86_64            │                │  → BMDB kernel        │
 │ bmdb-amt.py  (WS-Man 16992)  │──force PXE────▶│                       │
 │   power on / reset           │                │ KT UART (SoL)         │
 │ amtterm17  (SoL 16994)       │◀──serial──────│  ← console output     │
 └─────────────────────────────┘                └──────────────────────┘
```

## One-time setup

### 0. BIOS (physical, at the box — once)

- **Disable Secure Boot** (our bootloader is unsigned).
- Enable the **UEFI IPv4 PXE** network stack / Network Boot.
- Confirm the **NVMe is sacrificial**: BMDB writes its superblock and WAL
  from LBA 0 and will destroy any existing contents.

### 1. TFTP root + firewall (dev host, needs root — Claude cannot sudo)

```bash
sudo mkdir -p /srv/tftp && sudo chown "$USER":"$USER" /srv/tftp
sudo ufw allow proto udp from 192.168.0.0/24 to any port 67   comment 'BMDB proxyDHCP'
sudo ufw allow proto udp from 192.168.0.0/24 to any port 69   comment 'BMDB TFTP'
sudo ufw allow proto udp from 192.168.0.0/24 to any port 4011 comment 'BMDB PXE'
```

### 2. amtterm 1.7 (dev host)

The distro `amtterm` (1.4) cannot authenticate to AMT 12. Build 1.7:

```bash
cd /tmp && curl -fsSLO https://github.com/kraxel/amtterm/archive/refs/tags/amtterm-1.7-1.tar.gz
tar xf amtterm-1.7-1.tar.gz && cd amtterm-amtterm-1.7-1
# The generated Make.config can carry an `echo -e` artefact; if `make`
# fails on its first line, rewrite it to the four plain assignments.
make amtterm && install -D amtterm ~/.local/bin/amtterm17
```

`hw-runner` looks for `$AMT_TERM`, then `~/.local/bin/amtterm17`; it never
falls back to the useless distro 1.4.

## Per-session

Terminal 1 — start the boot server (stays in the foreground; Ctrl-C stops it):

```bash
sudo dnsmasq --no-daemon --conf-file=tools/pxe/dnsmasq-bmdb.conf
```

Terminal 2 — enable the AMT SoL listener once (the M920q ships with it off,
which is why a raw `amtterm` just hangs at CONNECT), then run a test:

```bash
tools/amt/bmdb-amt.py sol-enable          # one WS-Man Put; persists
cargo run -p hw-runner -- hw-test                     # engine durability gate
cargo run -p hw-runner -- hw-test --features silo-bench --timeout 600
```

`hw-test` deploys the freshly built boot files to `/srv/tftp`, opens the SoL
session, forces a one-shot PXE boot, powers the box on (or resets it if it is
already up), and captures serial until `It did not crash!` / `panic:` /
timeout.

## Debugging a failed boot

Server-side is the only signal until BMDB's own kernel starts talking, so
watch the dnsmasq terminal:

1. No `DHCPDISCOVER` from the M920q MAC → cabling / PXE not enabled in BIOS.
2. `DHCPDISCOVER` but no TFTP `sent .../bootloader` → proxyDHCP/arch mismatch
   (check the `client-arch` in `--log-dhcp`; add the arch to
   `dnsmasq-bmdb.conf` if it is outside 6/7/9).
3. `bootloader` sent but no `kernel-x86_64` fetch → bootloader/kernel pairing.
4. Kernel fetched but SoL stays silent → the KT UART console path
   (firmware POST is silent on SoL by design; only BMDB's kernel drives it).
   Confirm SoL was opened *before* power-on, then compare `lspci -v` for
   `00:16.3` against a Linux boot on the same box.
