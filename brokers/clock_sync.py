# brokers/clock_sync.py — make signed requests immune to a bad system clock.
# ─────────────────────────────────────────────────────────────
# Bybit validates every signed request against ITS clock:
#
#     server_time − recv_window  ≤  req_timestamp  ≤  server_time + 1000
#
# The window is asymmetric. A SLOW clock is forgiven up to recv_window; a
# clock more than ~1s FAST is rejected outright (retCode 10002), and
# enlarging recv_window cannot help in that direction — pybit's response to
# a 10002 is to widen recv_window and retry, which in the fast case simply
# fails again until it gives up.
#
# Relying on the OS to keep time is a weak dependency. On the machine this
# was written for, `w32tm /query /status` reported:
#
#     Leap Indicator: 3 (not synchronized)
#     Source:         Local CMOS Clock
#     Last successful sync time: unspecified
#
# — i.e. Windows had never once synced, and the motherboard clock was ~6s
# fast. Fixing that needs Administrator rights the bot may not have.
#
# The timestamp in a signed request does not actually have to come from the
# system clock. It only has to be close to the SERVER's. So we measure the
# offset once at startup and correct for it. The bot then trades correctly
# on a machine whose clock is wrong — while the clock itself should still
# be fixed, because everything else on the box is affected too.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import threading
import time
from typing import Optional

from loguru import logger

# Re-measure periodically: a clock with no NTP source doesn't just sit at a
# fixed offset, it keeps sliding.
REFRESH_SECONDS = 900

_offset_ms: float = 0.0
_last_measured: float = 0.0
_lock = threading.Lock()
_installed = False


def measure_offset(samples: int = 3) -> Optional[float]:
    """
    Local clock minus Bybit's, in milliseconds. Positive means we run fast.

    Takes the median of several samples and brackets each one by the
    request's own round trip, so network latency doesn't masquerade as skew.
    """
    from data import get_clock_offset_ms

    readings = []
    for _ in range(samples):
        value = get_clock_offset_ms()
        if value is not None:
            readings.append(value)
    if not readings:
        return None
    return sorted(readings)[len(readings) // 2]


def _corrected_timestamp() -> int:
    """Drop-in for pybit's generate_timestamp, shifted onto server time."""
    global _offset_ms, _last_measured

    now = time.time()
    if now - _last_measured > REFRESH_SECONDS:
        # Refresh opportunistically. Never let a failed measurement block a
        # request — a stale correction beats no request at all.
        with _lock:
            if now - _last_measured > REFRESH_SECONDS:
                _last_measured = now
                fresh = measure_offset(samples=1)
                if fresh is not None:
                    if abs(fresh - _offset_ms) > 1000:
                        logger.info(f"Clock offset re-measured: "
                                    f"{_offset_ms:+.0f} → {fresh:+.0f} ms")
                    _offset_ms = fresh

    return int(now * 1000 - _offset_ms)


def install(offset_ms: Optional[float] = None) -> Optional[float]:
    """
    Patch pybit to timestamp requests against Bybit's clock.

    Returns the measured offset, or None if it couldn't be determined (in
    which case nothing is patched and the system clock is used as before).
    """
    global _offset_ms, _last_measured, _installed

    if offset_ms is None:
        offset_ms = measure_offset()
    if offset_ms is None:
        logger.warning("Could not measure the clock offset — using the system "
                       "clock unchanged")
        return None

    _offset_ms = offset_ms
    _last_measured = time.time()

    if not _installed:
        from pybit import _helpers
        _helpers.generate_timestamp = _corrected_timestamp

        # _http_manager imported the symbol directly, so rebinding the
        # module attribute alone would miss it.
        try:
            from pybit import _http_manager
            if hasattr(_http_manager, '_helpers'):
                _http_manager._helpers.generate_timestamp = _corrected_timestamp
        except Exception as e:  # pragma: no cover
            logger.debug(f"Could not patch _http_manager helpers: {e}")

        _installed = True

    if abs(offset_ms) > 1000:
        logger.warning(
            f"System clock is {offset_ms / 1000:+.1f}s off Bybit — correcting "
            f"request timestamps in software. Fix the clock as well: every "
            f"other program on this machine has the same wrong time."
        )
    else:
        logger.info(f"Clock offset {offset_ms:+.0f} ms — request timestamps "
                    f"corrected")

    return offset_ms


def current_offset_ms() -> float:
    return _offset_ms
