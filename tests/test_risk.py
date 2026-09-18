"""
Tests for the risk model.

The central claim these defend: a trade sized at RISK_PER_TRADE loses
exactly that fraction of equity when its stop is hit — regardless of price,
stop width, or leverage. The old sizing rule failed this by ~50×, and the
failure was invisible because nothing ever asserted it.
"""
from decimal import Decimal

import pytest

import config
import risk
from brokers.base import BrokerPosition
from data import Instrument


def make_instrument(qty_step='0.001', min_qty='0.001', tick='0.1',
                    min_notional='5', max_qty='1500') -> Instrument:
    return Instrument(
        symbol='TESTUSDT',
        qty_step=Decimal(qty_step),
        min_qty=Decimal(min_qty),
        max_qty=Decimal(max_qty),
        tick_size=Decimal(tick),
        min_notional=Decimal(min_notional),
        max_leverage=100.0,
    )


# ══════════════════════════════════════════════════════════════
# SIZING — the core invariant
# ══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("entry,stop", [
    (100_000.0, 98_000.0),    # BTC-ish, 2% stop
    (100_000.0, 95_000.0),    # wide stop
    (100_000.0, 99_500.0),    # tight stop
    (150.0,     144.0),       # SOL-ish, 4% stop
    (3_500.0,   3_430.0),     # ETH-ish, 2% stop
])
def test_loss_at_stop_equals_risk_budget(entry, stop):
    """Whatever the price or stop width, hitting the stop costs 1% of equity."""
    equity = 10_000.0
    result = risk.size_position(
        equity=equity, entry=entry, stop=stop,
        instrument=make_instrument(qty_step='0.000001', min_qty='0.000001'),
        risk_pct=0.01, leverage=10.0,
    )
    assert result.ok, result.reason

    loss_at_stop = abs(entry - stop) * result.qty
    assert loss_at_stop == pytest.approx(equity * 0.01, rel=0.01)


def test_wider_stop_buys_smaller_position():
    """Risk is held constant by shrinking size as the stop widens."""
    instrument = make_instrument(qty_step='0.000001', min_qty='0.000001')
    tight = risk.size_position(10_000, 100_000, 99_000, instrument, leverage=10)
    wide = risk.size_position(10_000, 100_000, 95_000, instrument, leverage=10)

    assert tight.qty > wide.qty
    assert tight.risk_amount == pytest.approx(wide.risk_amount, rel=0.02)


def test_old_notional_sizing_understated_risk_by_20x():
    """
    Regression guard against the pre-migration bug.

    The old rule risked risk_pct × stop_pct of equity. With 1% and a 5%
    stop that is 0.05%, i.e. 1/stop_pct = 20× too small — the number this
    test pins the fix against.
    """
    equity, entry, stop_pct = 10_000.0, 100_000.0, 0.05
    stop = entry * (1 - stop_pct)

    old_qty = (equity * 0.01) / entry            # notional allocation
    old_loss = abs(entry - stop) * old_qty
    assert old_loss == pytest.approx(equity * 0.0005, rel=0.01)

    new = risk.size_position(equity, entry, stop,
                             make_instrument(qty_step='0.000001', min_qty='0.000001'),
                             risk_pct=0.01, leverage=10)
    new_loss = abs(entry - stop) * new.qty
    assert new_loss / old_loss == pytest.approx(20.0, rel=0.05)


def test_leverage_caps_notional_not_risk():
    """A very tight stop is clamped by leverage, and risk falls accordingly."""
    equity = 10_000.0
    result = risk.size_position(
        equity=equity, entry=100_000.0, stop=99_990.0,     # 0.01% stop
        instrument=make_instrument(qty_step='0.000001', min_qty='0.000001'),
        risk_pct=0.01, leverage=3.0,
    )
    assert result.ok
    assert result.notional <= equity * 3.0 * 1.001
    # Risk ends up BELOW budget once clamped — never above it.
    assert result.risk_amount <= equity * 0.01 * 1.001


def test_rejects_sub_minimum_notional():
    result = risk.size_position(
        equity=100.0, entry=100_000.0, stop=90_000.0,
        instrument=make_instrument(), risk_pct=0.01, leverage=3.0,
    )
    assert not result.ok


def test_quantity_is_rounded_down_to_lot_step():
    """Rounding up would push every position slightly over budget."""
    result = risk.size_position(
        equity=10_000.0, entry=1_000.0, stop=900.0,
        instrument=make_instrument(qty_step='0.01', min_qty='0.01'),
        risk_pct=0.01, leverage=10.0,
    )
    assert result.ok
    # exact size is 1.0 / ... -> check it is a clean multiple of the step
    assert abs(result.qty / 0.01 - round(result.qty / 0.01)) < 1e-9
    assert abs(1_000.0 - 900.0) * result.qty <= 10_000.0 * 0.01 + 1e-6


def test_zero_stop_distance_is_rejected():
    result = risk.size_position(10_000, 100.0, 100.0, make_instrument())
    assert not result.ok


# ══════════════════════════════════════════════════════════════
# STOPS AND TARGETS
# ══════════════════════════════════════════════════════════════

def test_atr_stop_scales_with_volatility():
    quiet_stop, _ = risk.atr_stop_target(100_000, 'long', atr=500)
    wild_stop, _ = risk.atr_stop_target(100_000, 'long', atr=3_000)
    assert (100_000 - wild_stop) > (100_000 - quiet_stop)


def test_stop_and_target_mirror_for_shorts():
    long_stop, long_take = risk.atr_stop_target(1_000, 'long', atr=10)
    short_stop, short_take = risk.atr_stop_target(1_000, 'short', atr=10)
    assert long_stop < 1_000 < long_take
    assert short_take < 1_000 < short_stop
    assert (short_stop - 1_000) == pytest.approx(1_000 - long_stop)


def test_target_preserves_r_multiple_after_clamping():
    """Clamping the stop must not silently change the payoff ratio."""
    entry = 100.0
    stop, take = risk.atr_stop_target(entry, 'long', atr=100.0)   # forces the clamp
    reward = take - entry
    risk_distance = entry - stop
    expected_r = config.ATR_TARGET_MULTIPLIER / config.ATR_STOP_MULTIPLIER
    assert reward / risk_distance == pytest.approx(expected_r, rel=1e-6)


def test_stop_distance_is_clamped_both_ways():
    tiny_stop, _ = risk.atr_stop_target(1_000, 'long', atr=0.0001)
    huge_stop, _ = risk.atr_stop_target(1_000, 'long', atr=10_000)
    assert (1_000 - tiny_stop) == pytest.approx(1_000 * config.MIN_STOP_PCT)
    assert (1_000 - huge_stop) == pytest.approx(1_000 * config.MAX_STOP_PCT)


# ══════════════════════════════════════════════════════════════
# PORTFOLIO RISK
# ══════════════════════════════════════════════════════════════

def make_position(symbol='BTCUSDT', side='long', qty=0.1,
                  entry=100_000.0, stop=98_000.0) -> BrokerPosition:
    return BrokerPosition(
        symbol=symbol, side=side, qty=qty, entry_price=entry,
        mark_price=entry, unrealised_pnl=0.0, stop_loss=stop,
    )


def test_portfolio_heat_sums_open_risk():
    positions = {
        'BTCUSDT': make_position('BTCUSDT', qty=0.05, entry=100_000, stop=98_000),
        'ETHUSDT': make_position('ETHUSDT', qty=1.0, entry=3_500, stop=3_430),
    }
    # 0.05 × 2000 = 100, plus 1.0 × 70 = 70 → 170 of 10_000 = 1.7%
    assert risk.portfolio_heat(positions, 10_000) == pytest.approx(0.017)


def test_unprotected_position_counts_full_notional_as_risk():
    """A position with no stop really can lose everything — say so."""
    position = make_position(stop=None)
    position.stop_loss = None
    assert risk.position_risk(position) == pytest.approx(position.qty * position.mark_price)


def test_heat_cap_blocks_an_over_exposed_entry(monkeypatch):
    monkeypatch.setattr(config, 'MAX_PORTFOLIO_HEAT', 0.02)
    monkeypatch.setattr(config, 'RISK_PER_TRADE', 0.01)
    monkeypatch.setattr(risk, 'correlated_exposure_ok', lambda *a, **k: (True, 'stubbed'))

    positions = {
        'ETHUSDT': make_position('ETHUSDT', qty=1.0, entry=3_500, stop=3_325),  # 175 = 1.75%
    }
    decision = risk.evaluate_entry(
        'BTCUSDT', 'long', 100_000, 98_000, 10_000,
        positions, make_instrument(),
    )
    assert not decision.allowed
    assert 'heat' in decision.reason


def test_max_positions_blocks_entry(monkeypatch):
    monkeypatch.setattr(config, 'MAX_OPEN_POSITIONS', 2)
    positions = {
        'ETHUSDT': make_position('ETHUSDT'),
        'SOLUSDT': make_position('SOLUSDT'),
    }
    decision = risk.evaluate_entry('BTCUSDT', 'long', 100_000, 98_000, 100_000,
                                   positions, make_instrument())
    assert not decision.allowed
    assert 'max positions' in decision.reason


def test_duplicate_symbol_blocked():
    positions = {'BTCUSDT': make_position('BTCUSDT')}
    decision = risk.evaluate_entry('BTCUSDT', 'long', 100_000, 98_000, 100_000,
                                   positions, make_instrument())
    assert not decision.allowed


# ══════════════════════════════════════════════════════════════
# TRAILING STOP
# ══════════════════════════════════════════════════════════════

def test_trail_is_not_armed_before_activation_threshold():
    position = make_position(entry=100_000, stop=98_000)   # 1R = 2000
    # Only +1000 (0.5R) in favour — too early to trail.
    assert risk.trailing_stop_price(position, 101_000, atr=500) is None


def test_trail_arms_after_one_r_and_only_improves():
    position = make_position(entry=100_000, stop=98_000)
    level = risk.trailing_stop_price(position, 103_000, atr=500, multiplier=2.0)
    assert level == pytest.approx(102_000)      # 103_000 − 2×500
    assert level > position.stop_loss

    # A level worse than the existing stop is refused, never widened.
    position.stop_loss = 102_500
    assert risk.trailing_stop_price(position, 103_000, atr=500, multiplier=2.0) is None


def test_trail_mirrors_for_shorts():
    position = make_position(side='short', entry=100_000, stop=102_000)
    level = risk.trailing_stop_price(position, 97_000, atr=500, multiplier=2.0)
    assert level == pytest.approx(98_000)
    assert level < position.stop_loss


# ══════════════════════════════════════════════════════════════
# FUNDING
# ══════════════════════════════════════════════════════════════

def test_funding_exit_is_direction_aware():
    high = config.FUNDING_EXIT_THRESHOLD * 2
    assert risk.funding_argues_for_exit('long', high)
    assert not risk.funding_argues_for_exit('short', high)
    assert risk.funding_argues_for_exit('short', -high)
    assert not risk.funding_argues_for_exit('long', 0.0)
