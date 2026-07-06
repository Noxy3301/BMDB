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
import sys

import requests
from requests.auth import HTTPDigestAuth

import amt.client
import amt.wsman

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
    )
    resp.raise_for_status()
    return resp.content.decode("utf-8", "replace")


def cmd_status(client):
    state = int(client.power_status())
    print({2: "on", 3: "sleep", 4: "off", 8: "off"}.get(state, f"state({state})"))
    return state


def cmd_sol_enable(uri, password):
    """GET the redirection service, flip ListenerEnabled true, PUT it back.

    Round-tripping the live instance avoids guessing the exact property
    set/order AMT expects in a Put body.
    """
    body = wsman_post(uri, password, amt.wsman.get_request(uri, REDIR_URI))

    if not re.search(r"ListenerEnabled\s*>\s*(?:true|1)\s*<", body, re.I):
        # Extract the resource element verbatim and flip the flag.
        m = re.search(r"(<(\w+):AMT_RedirectionService\b.*?</\2:AMT_RedirectionService>)",
                      body, re.S)
        if not m:
            sys.exit("sol-enable: AMT_RedirectionService not found in GET response")
        resource = re.sub(r"(<\w+:ListenerEnabled>)[^<]*(</\w+:ListenerEnabled>)",
                          r"\g<1>true\g<2>", m.group(1))
        put = _put_envelope(uri, resource)
        wsman_post(uri, password, put)

    # Confirm.
    body = wsman_post(uri, password, amt.wsman.get_request(uri, REDIR_URI))
    ok = bool(re.search(r"ListenerEnabled\s*>\s*(?:true|1)\s*<", body, re.I))
    print("ListenerEnabled=true" if ok else "ListenerEnabled still false")
    if not ok:
        sys.exit(1)


def _put_envelope(uri, resource):
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"'
        ' xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing"'
        ' xmlns:wsman="http://schemas.dmtf.org/wbem/wsman/1/wsman.xsd">'
        "<s:Header>"
        "<wsa:Action s:mustUnderstand=\"true\">"
        "http://schemas.xmlsoap.org/ws/2004/09/transfer/Put</wsa:Action>"
        f"<wsa:To s:mustUnderstand=\"true\">{uri}</wsa:To>"
        f"<wsman:ResourceURI s:mustUnderstand=\"true\">{REDIR_URI}</wsman:ResourceURI>"
        "<wsa:MessageID s:mustUnderstand=\"true\">uuid:bmdb-sol-enable</wsa:MessageID>"
        "<wsa:ReplyTo><wsa:Address>"
        "http://schemas.xmlsoap.org/ws/2004/08/addressing/role/anonymous"
        "</wsa:Address></wsa:ReplyTo>"
        "</s:Header>"
        f"<s:Body>{resource}</s:Body>"
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
