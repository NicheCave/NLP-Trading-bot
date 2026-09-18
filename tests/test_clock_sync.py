"""
Tests for software clock correction.

Bybit's signature window is asymmetric:

    server_time − recv_window  ≤  req_timestamp  ≤  server_time + 1000

so a FAST clock is fatal past ~1s no matter how large recv_window is. The
machine this was built on had never synced (`Source: Local CMOS Clock`,
`Leap Indicator: 3 (not synchronized)`) and ran ~6s fast, which made every
signed request fail. Correcting the timestamp in software removes the
dependency on the OS keeping time.
"""
import time

import pytest

from brokers import clock_sync


@pytest.fixture(autouse=True)
def reset_module_state(monkeypatch):
    monkeypatch.setattr(clock_sync, '_offset_ms', 0.0)
    monkeypatch.setattr(clock_sync, '_last_measured', 0.0)
    yield


def test_fast_clock_is_corrected_backwards(monkeypatch):
    """A clock running 6s fast must produce timestamps 6s lower."""
    monkeypatch.setattr(clock_sync, '_offset_ms', 6000.0)
    monkeypatch.setattr(clock_sync, '_last_measured', time.time())

    corrected = clock_sync._corrected_timestamp()
    uncorrected = int(time.time() * 1000)

    assert uncorrected - corrected == pytest.approx(6000, abs=50)


def test_slow_clock_is_corrected_forwards(monkeypatch):
    monkeypatch.setattr(clock_sync, '_offset_ms', -4000.0)
    monkeypatch.setattr(clock_sync, '_last_measured', time.time())

    corrected = clock_sync._corrected_timestamp()
    uncorrected = int(time.time() * 1000)

    assert corrected - uncorrected == pytest.approx(4000, abs=50)


def test_corrected_timestamp_lands_inside_bybit_window(monkeypatch):
    """
    The whole point: after correction, the timestamp must satisfy
    server_time − recv_window ≤ ts ≤ server_time + 1000.
    """
    for skew_ms in (+6000.0, -6000.0, +900.0, 0.0, -25_000.0):
        monkeypatch.setattr(clock_sync, '_offset_ms', skew_ms)
        monkeypatch.setattr(clock_sync, '_last_measured', time.time())

        ts = clock_sync._corrected_timestamp()

        # Careful with frames of reference: time.time() IS the skewed clock,
        # so true server time is that reading minus the skew.
        true_server_ms = time.time() * 1000 - skew_ms

        assert ts - true_server_ms < 1000, f"still too fast at skew {skew_ms}"
        assert ts - true_server_ms > -20_000, f"overcorrected at skew {skew_ms}"


def test_offset_is_refreshed_when_stale(monkeypatch):
    """A clock with no NTP source keeps sliding, so re-measure periodically."""
    monkeypatch.setattr(clock_sync, '_offset_ms', 1000.0)
    monkeypatch.setattr(clock_sync, '_last_measured',
                        time.time() - clock_sync.REFRESH_SECONDS - 10)
    monkeypatch.setattr(clock_sync, 'measure_offset', lambda samples=1: 7000.0)

    clock_sync._corrected_timestamp()
    assert clock_sync.current_offset_ms() == pytest.approx(7000.0)


def test_fresh_offset_is_not_re_measured(monkeypatch):
    calls = []
    monkeypatch.setattr(clock_sync, '_offset_ms', 1234.0)
    monkeypatch.setattr(clock_sync, '_last_measured', time.time())
    monkeypatch.setattr(clock_sync, 'measure_offset',
                        lambda samples=1: calls.append(1) or 9999.0)

    clock_sync._corrected_timestamp()
    assert calls == []
    assert clock_sync.current_offset_ms() == pytest.approx(1234.0)


def test_failed_refresh_keeps_the_last_known_offset(monkeypatch):
    """A stale correction beats blocking the request."""
    monkeypatch.setattr(clock_sync, '_offset_ms', 5000.0)
    monkeypatch.setattr(clock_sync, '_last_measured',
                        time.time() - clock_sync.REFRESH_SECONDS - 10)
    monkeypatch.setattr(clock_sync, 'measure_offset', lambda samples=1: None)

    ts = clock_sync._corrected_timestamp()
    assert clock_sync.current_offset_ms() == pytest.approx(5000.0)
    assert ts == pytest.approx(int(time.time() * 1000) - 5000, abs=100)


def test_install_without_a_reachable_server_changes_nothing(monkeypatch):
    monkeypatch.setattr(clock_sync, 'measure_offset', lambda samples=3: None)
    assert clock_sync.install() is None


def test_install_patches_pybit(monkeypatch):
    monkeypatch.setattr(clock_sync, 'measure_offset', lambda samples=3: 3000.0)
    monkeypatch.setattr(clock_sync, '_installed', False)

    assert clock_sync.install() == pytest.approx(3000.0)

    from pybit import _helpers
    assert _helpers.generate_timestamp is clock_sync._corrected_timestamp

    # Every signed request pybit builds now carries the corrected time.
    assert int(time.time() * 1000) - _helpers.generate_timestamp() == \
        pytest.approx(3000, abs=50)
