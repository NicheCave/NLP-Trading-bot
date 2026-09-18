"""
Tests for the local paper broker, against hand-computed fills.

Prices are injected rather than fetched, so every expected number below is
arithmetic that can be checked by hand — which is the point: if the
simulator and the real venue ever disagree, the backtest is measuring
fiction.
"""
import pytest

import config
from brokers.base import BrokerPosition
from brokers.sim import SimBroker


@pytest.fixture
def prices():
    """Mutable price book the tests drive directly."""
    return {'BTCUSDT': 100_000.0}


@pytest.fixture
def broker(prices):
    return SimBroker(
        initial_equity=10_000.0,
        taker_fee=0.001,        # round numbers make the arithmetic checkable
        slippage_bps=0.0,
        price_fn=lambda symbol: prices.get(symbol),
        persist=False,
    )


# ══════════════════════════════════════════════════════════════
# FILLS
# ══════════════════════════════════════════════════════════════

def test_open_charges_taker_fee(broker):
    assert broker.open_position('BTCUSDT', 'long', 0.01)
    # 0.01 × 100_000 × 0.001 = 1.0
    assert broker.cash == pytest.approx(10_000.0 - 1.0)


def test_long_round_trip_pnl(broker, prices):
    broker.open_position('BTCUSDT', 'long', 0.01)      # fee 1.0
    prices['BTCUSDT'] = 110_000.0
    broker.close_position('BTCUSDT')                   # fee 0.01×110_000×0.001 = 1.1

    # gross = 0.01 × 10_000 = 100 ; net = 100 − 1.0 − 1.1 = 97.9
    assert broker.get_equity() == pytest.approx(10_000.0 + 97.9)
    assert broker.closed_trades[-1]['gross_pnl'] == pytest.approx(100.0)


def test_short_profits_when_price_falls(broker, prices):
    broker.open_position('BTCUSDT', 'short', 0.01)
    prices['BTCUSDT'] = 90_000.0
    broker.close_position('BTCUSDT')

    trade = broker.closed_trades[-1]
    assert trade['gross_pnl'] == pytest.approx(100.0)
    assert broker.get_equity() > 10_000.0


def test_short_loses_when_price_rises(broker, prices):
    broker.open_position('BTCUSDT', 'short', 0.01)
    prices['BTCUSDT'] = 110_000.0
    broker.close_position('BTCUSDT')
    assert broker.closed_trades[-1]['gross_pnl'] == pytest.approx(-100.0)


def test_slippage_moves_fills_against_us(prices):
    broker = SimBroker(initial_equity=10_000.0, taker_fee=0.0, slippage_bps=10.0,
                       price_fn=lambda s: prices.get(s), persist=False)
    broker.open_position('BTCUSDT', 'long', 0.01)
    # Buying pays UP by 10 bps.
    assert broker.get_position('BTCUSDT').entry_price == pytest.approx(100_100.0)

    broker.close_position('BTCUSDT')
    # Selling receives DOWN 10 bps — a round trip at an unchanged price loses.
    assert broker.closed_trades[-1]['exit_price'] == pytest.approx(99_900.0)
    assert broker.closed_trades[-1]['gross_pnl'] < 0


# ══════════════════════════════════════════════════════════════
# STOPS
# ══════════════════════════════════════════════════════════════

def test_stop_loss_fires_for_long(broker, prices):
    broker.open_position('BTCUSDT', 'long', 0.01, stop_loss=98_000.0, take_profit=104_000.0)
    prices['BTCUSDT'] = 97_500.0

    assert broker.mark_to_market() == ['BTCUSDT']
    trade = broker.closed_trades[-1]
    assert trade['reason'] == 'stop_loss'
    # Filled AT the stop level, not at the worse traded price.
    assert trade['exit_price'] == pytest.approx(98_000.0)


def test_take_profit_fires_for_long(broker, prices):
    broker.open_position('BTCUSDT', 'long', 0.01, stop_loss=98_000.0, take_profit=104_000.0)
    prices['BTCUSDT'] = 104_500.0

    assert broker.mark_to_market() == ['BTCUSDT']
    assert broker.closed_trades[-1]['reason'] == 'take_profit'


def test_short_stops_are_inverted(broker, prices):
    broker.open_position('BTCUSDT', 'short', 0.01, stop_loss=102_000.0, take_profit=96_000.0)

    prices['BTCUSDT'] = 101_000.0
    assert broker.mark_to_market() == []          # not breached yet

    prices['BTCUSDT'] = 102_500.0
    assert broker.mark_to_market() == ['BTCUSDT']
    assert broker.closed_trades[-1]['reason'] == 'stop_loss'


def test_stops_do_not_fire_inside_the_band(broker, prices):
    broker.open_position('BTCUSDT', 'long', 0.01, stop_loss=98_000.0, take_profit=104_000.0)
    for price in (99_000.0, 100_000.0, 103_000.0):
        prices['BTCUSDT'] = price
        assert broker.mark_to_market() == []
    assert broker.has_position('BTCUSDT')


def test_set_stops_updates_an_open_position(broker):
    broker.open_position('BTCUSDT', 'long', 0.01, stop_loss=98_000.0)
    assert broker.set_stops('BTCUSDT', stop_loss=99_000.0)
    assert broker.get_position('BTCUSDT').stop_loss == 99_000.0


# ══════════════════════════════════════════════════════════════
# GUARDS
# ══════════════════════════════════════════════════════════════

def test_cannot_double_open_the_same_symbol(broker):
    assert broker.open_position('BTCUSDT', 'long', 0.01)
    result = broker.open_position('BTCUSDT', 'long', 0.01)
    assert not result.ok and 'already in position' in result.error


def test_closing_a_flat_symbol_fails_cleanly(broker):
    result = broker.close_position('BTCUSDT')
    assert not result.ok and 'no open position' in result.error


def test_sub_minimum_quantity_is_rejected(broker):
    result = broker.open_position('BTCUSDT', 'long', 0.0000001)
    assert not result.ok


def test_missing_price_is_handled(broker):
    result = broker.open_position('DOGEUSDT', 'long', 100)
    assert not result.ok


# ══════════════════════════════════════════════════════════════
# REPORTED STATE
# ══════════════════════════════════════════════════════════════

def test_unrealised_pnl_is_quote_currency_not_margin_percent(broker, prices):
    """
    The leverage trap: Bybit's unrealised % is a percentage of MARGIN.
    BrokerPosition.pnl_pct_of_price must report the move in the UNDERLYING,
    or an ATR target computed in price terms fires leverage-times too early.
    """
    broker.open_position('BTCUSDT', 'long', 0.01)
    prices['BTCUSDT'] = 102_000.0

    position = broker.get_position('BTCUSDT')
    assert position.unrealised_pnl == pytest.approx(20.0)     # 0.01 × 2000
    assert position.pnl_pct_of_price() == pytest.approx(0.02)  # 2%, not 6%


def test_short_pnl_pct_is_signed_correctly(broker, prices):
    broker.open_position('BTCUSDT', 'short', 0.01)
    prices['BTCUSDT'] = 98_000.0
    assert broker.get_position('BTCUSDT').pnl_pct_of_price() == pytest.approx(0.02)


def test_equity_tracks_open_position(broker, prices):
    broker.open_position('BTCUSDT', 'long', 0.01)
    baseline = broker.get_equity()
    prices['BTCUSDT'] = 105_000.0
    assert broker.get_equity() == pytest.approx(baseline + 50.0)
