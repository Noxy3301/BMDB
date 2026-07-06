#!/home/noxy/.amt-venv/bin/python3
"""AMT control for the BMDB hardware test loop.

Wraps the `amt` (Sean Dague) WS-Management client with the two things it
lacks for our loop:

  * `reset`  — CIM power state 10 (Master Bus Reset). The stock package
    only knows on(2)/off(8)/reboot(5); reboot is Power-Cycle-Off-Soft,
    which AMT REJECTS while a Serial-over-LAN session is open. From S0
    with SoL open we must use reset(10); from S5 we use on(2).
  * `sol-enable` — turn the redirection listener on. The M920q ships
    with SOL/IDER *enabled* (EnabledState 32771) but the network
    *listener* off, so port 16994 is filtered and `amtterm` hangs at
    CONNECT. One WS-Man Put flips it; it persists across reboots.

Credentials come from ~/.bmdb-amt.env (AMT_HOST, AMT_PASSWORD).

Usage: bmdb-amt.py {status|sol-enable|pxe-next|on|reset|off}
"""

import os
import re
import socket
import sys
from xml.sax.saxutils import escape

import requests
from requests.auth import HTTPDigestAuth

import amt.client
import amt.wsman

# Bound every AMT network call so an unresponsive management engine can
# never hang the automated hardware test. `socket.setdefaulttimeout` is
# not enough on its own: requests/urllib3 turn an omitted timeout into an
# explicit `None`, overriding the process default — so the amt.client
# calls (status/pxe-next/on/reset) would stay unbounded. Wrap
# `requests.post` (which every amt.client network call and our own
# wsman_post go through) to inject a default timeout when none is given.
AMT_TIMEOUT = 15
socket.setdefaulttimeout(AMT_TIMEOUT)

_orig_requests_post = requests.post


def _post_with_timeout(*args, **kwargs):
    kwargs.setdefault("timeout", AMT_TIMEOUT)
    return _orig_requests_post(*args, **kwargs)


requests.post = _post_with_timeout

REDIR_URI = "http://intel.com/wbem/wscim/1/amt-schema/1/AMT_RedirectionService"

# Master Bus Reset. Absent from amt.wsman.POWER_STATES; injected so
# power_state_request("reset") builds a valid RequestPowerStateChange.
amt.wsman.POWER_STATES.setdefault("reset", 10)


def load_env():
    path = os.path.expanduser("~/.bmdb-amt.env")
    host = password = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("AMT_HOST="):
                host = line.split("=", 1)[1].strip()
            elif line.startswith("AMT_PASSWORD="):
                password = line.split("=", 1)[1].strip()
    if not host or not password:
        sys.exit(f"AMT_HOST / AMT_PASSWORD missing from {path}")
    return host, password


def wsman_post(uri, password, payload):
    resp = requests.post(
        uri,
        headers={"content-type": "application/soap+xml;charset=UTF-8"},
        auth=HTTPDigestAuth("admin", password),
        data=payload,
        timeout=AMT_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.content.decode("utf-8", "replace")


def cmd_status(client):
    state = int(client.power_status())
    print({2: "on", 3: "sleep", 4: "off", 8: "off"}.get(state, f"state({state})"))
    return state


def cmd_sol_enable(uri, password):
    """GET the redirection service, then PUT it back with ListenerEnabled
    true. The PUT re-emits the instance's own property values (so AMT sees
    a complete, valid tuple) with the flag flipped, declares the schema
    namespace on the resource element, and carries the four key selectors
    in the header — AMT rejects a Put missing any of these with 400.
    """
    from xml.etree import ElementTree as ET

    ns = "{" + REDIR_URI + "}"
    doc = ET.fromstring(wsman_post(uri, password, amt.wsman.get_request(uri, REDIR_URI)))
    svc = doc.find(f".//{ns}AMT_RedirectionService")
    if svc is None:
        sys.exit("sol-enable: AMT_RedirectionService not found in GET response")

    # Preserve the instance's property order; force ListenerEnabled true.
    props = []
    for child in svc:
        name = child.tag[len(ns):] if child.tag.startswith(ns) else child.tag
        value = "true" if name == "ListenerEnabled" else (child.text or "")
        props.append((name, value))

    wsman_post(uri, password, _put_envelope(uri, props))

    # Confirm.
    body = wsman_post(uri, password, amt.wsman.get_request(uri, REDIR_URI))
    ok = bool(re.search(r"ListenerEnabled\s*>\s*(?:true|1)\s*<", body, re.I))
    print("ListenerEnabled=true (SoL listener open on 16994)" if ok else "ListenerEnabled still false")
    if not ok:
        sys.exit(1)


def _put_envelope(uri, props):
    keys = ("CreationClassName", "Name", "SystemCreationClassName", "SystemName")
    prop_by_name = dict(props)
    missing = [k for k in keys if not prop_by_name.get(k)]
    if missing:
        sys.exit(f"sol-enable: GET response missing key selector(s): {', '.join(missing)}")
    body = f'<r:AMT_RedirectionService xmlns:r="{REDIR_URI}">' + "".join(
        f"<r:{name}>{escape(value)}</r:{name}>" for name, value in props
    ) + "</r:AMT_RedirectionService>"
    selectors = "".join(
        f'<wsman:Selector Name="{k}">{escape(prop_by_name[k])}</wsman:Selector>' for k in keys
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"'
        ' xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing"'
        ' xmlns:wsman="http://schemas.dmtf.org/wbem/wsman/1/wsman.xsd">'
        "<s:Header>"
        '<wsa:Action s:mustUnderstand="true">'
        "http://schemas.xmlsoap.org/ws/2004/09/transfer/Put</wsa:Action>"
        f'<wsa:To s:mustUnderstand="true">{uri}</wsa:To>'
        f'<wsman:ResourceURI s:mustUnderstand="true">{REDIR_URI}</wsman:ResourceURI>'
        '<wsa:MessageID s:mustUnderstand="true">uuid:bmdb-sol-enable</wsa:MessageID>'
        "<wsa:ReplyTo><wsa:Address>"
        "http://schemas.xmlsoap.org/ws/2004/08/addressing/role/anonymous"
        "</wsa:Address></wsa:ReplyTo>"
        f"<wsman:SelectorSet>{selectors}</wsman:SelectorSet>"
        "</s:Header>"
        f"<s:Body>{body}</s:Body>"
        "</s:Envelope>"
    )


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    action = sys.argv[1]
    host, password = load_env()
    client = amt.client.Client(host, password)

    if action == "status":
        cmd_status(client)
    elif action == "sol-enable":
        cmd_sol_enable(client.uri, password)
    elif action == "pxe-next":
        client.pxe_next_boot()
        print("next boot: PXE (one-shot)")
    elif action == "on":
        client.power_on()
        print("power: on")
    elif action == "reset":
        client.post(amt.wsman.power_state_request(client.uri, "reset"),
                    amt.client.CIM_PowerManagementService)
        print("power: reset")
    elif action == "off":
        client.power_off()
        print("power: off")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
