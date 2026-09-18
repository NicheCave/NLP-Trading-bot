#!/usr/bin/env python
"""
Check the system clock against Bybit's, and say what to do about it.

    python -m tools.clockcheck

Bybit accepts a signed request only when

    server_time − recv_window  ≤  req_timestamp  ≤  server_time + 1000

The window is ASYMMETRIC. A slow clock is forgiven up to recv_window; a
clock more than ~1s FAST is rejected outright, and enlarging recv_window
cannot help — which is what makes retCode 10002 so confusing to debug.
"""
import sys

import config
from data import get_clock_offset_ms

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass


def main() -> int:
    samples = [get_clock_offset_ms() for _ in range(3)]
    samples = [s for s in samples if s is not None]
    if not samples:
        print("Could not reach Bybit to read server time.")
        return 2

    offset = sorted(samples)[len(samples) // 2]     # median of 3
    print(f"Clock offset : {offset:+.0f} ms "
          f"({'ahead of' if offset > 0 else 'behind'} Bybit)")
    print(f"recv_window  : {config.RECV_WINDOW} ms")
    print(f"Accepted     : {-config.RECV_WINDOW} ms .. +1000 ms")
    print()

    if offset > 1000:
        print("✗ TOO FAR AHEAD — signed requests will be rejected (10002).")
        print("  recv_window cannot fix this direction; the clock must be resynced.")
    elif offset > 500:
        print("⚠ Running fast. Under the 1000 ms limit but with little margin.")
    elif offset < -config.RECV_WINDOW:
        print("✗ TOO FAR BEHIND — signed requests will be rejected (10002).")
    elif offset < -config.RECV_WINDOW * 0.5:
        print("⚠ Running slow, over halfway to the recv_window limit.")
    else:
        print("✓ Clock is fine.")
        return 0

    print()
    print("Fix, in an ADMINISTRATOR PowerShell:")
    print("    net start w32time")
    print('    w32tm /config /manualpeerlist:"time.windows.com,0x9" '
          '/syncfromflags:manual /update')
    print("    w32tm /resync /force")
    print()
    print("If it still drifts, Windows Settings → Time & Language → Date & time")
    print("→ turn 'Set time automatically' off and on, then 'Sync now'.")
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
