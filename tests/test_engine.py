"""
Tests for the backtest engine.

Two things these must pin down, because getting either wrong makes every
downstream number meaningless:

  1. Trade accounting reconciles with the equity curve. If sum(net_pnl)
     drifts from the change in equity, the reported expectancy is fiction.
  2. No lookahead. A decision at bar i may only fill at bar i+1's open.
"""
import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestConfig, BacktestEngine, Intent


def make_bars(closes, start='2025-01-01', high_pct=0.0, low_pct=0.0) -> pd.DataFrame:
    index = pd.date_range(start, periods=len(closes), freq='1h', tz='UTC')
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        'open': closes,
        'high': closes * (1 + high_pct),
        'low': closes * (1 - low_pct),
        'close': closes,
        'volume': np.full(len(closes), 100.0),
        'turnover': closes * 100.0,
    }, index=index)


def make_intents(index, entries: dict) -> pd.DataFrame:
    """entries: {position_in_index: Intent}"""
    rows = []
    for i in range(len(index)):
        intent = entries.get(i)
        if intent is None:
            rows.append({'direction': 'flat', 'confidence': 0.0, 'stop_price': None,
                         'take_price': None, 'atr': None, 'reason': '', 'tags': []})
        else:
            rows.append({'direction': intent.direction, 'confidence': intent.confidence,
                         'stop_price': intent.stop_price, 'take_price': intent.take_price,
                         'atr': intent.atr, 'reason': intent.reason, 'tags': intent.tags})
    return pd.DataFrame(rows, index=index)


@pytest.fixture
def zero_cost_config():
    """Costs off, so arithmetic is exact and failures are unambiguous."""
    return BacktestConfig(
        initial_equity=10_000.0, taker_fee=0.0, slippage_bps=0.0,
        risk_per_trade=0.01, max_positions=4, max_leverage=10.0,
        funding_enabled=False, time_stop_bars=0, use_trailing=False,
    )


# ══════════════════════════════════════════════════════════════
# ACCOUNTING
# ══════════════════════════════════════════════════════════════

def test_net_pnl_includes_both_fees():
    """
    Regression guard: net_pnl once subtracted only the exit fee, which made
    sum(net_pnl) disagree with the equity curve by the entire entry-fee bill
    and quietly inflated the win rate.
    """
    bars = make_bars([100.0] * 10)
    index = bars.index
    intents = make_intents(index, {3: Intent('long', 1.0, stop_price=90.0, take_price=110.0)})

    cfg = BacktestConfig(initial_equity=10_000.0, taker_fee=0.001, slippage_bps=0.0,
                         risk_per_trade=0.01, funding_enabled=False,
                         time_stop_bars=0, use_trailing=False, max_leverage=10.0)
    engine = BacktestEngine(cfg)
    result = engine.run({'X': {'1h': bars, '1d': pd.DataFrame(), 'funding': pd.DataFrame()}},
                        intents={'X': intents}, warmup=2)

    assert len(result.trades) == 1
    trade = result.trades[0]
    # Price never moved, so the whole loss is fees on both legs.
    assert trade.gross_pnl == pytest.approx(0.0, abs=1e-9)
    assert trade.net_pnl == pytest.approx(-trade.fees)
    assert trade.fees > 0


def test_trade_pnl_reconciles_with_equity_curve():
    """sum(net_pnl) must equal the change in equity, to the cent."""
    rng = np.random.default_rng(7)
    closes = 100 * np.cumprod(1 + rng.normal(0, 0.01, 400))
    bars = make_bars(closes)
    index = bars.index

    entries = {i: Intent('long' if i % 2 else 'short',
                         0.8,
                         stop_price=float(closes[i] * (0.95 if i % 2 else 1.05)),
                         take_price=float(closes[i] * (1.05 if i % 2 else 0.95)))
               for i in range(50, 350, 30)}
    intents = make_intents(index, entries)

    cfg = BacktestConfig(initial_equity=10_000.0, taker_fee=0.00055, slippage_bps=2.0,
                         risk_per_trade=0.01, funding_enabled=False,
                         time_stop_bars=0, use_trailing=False, max_leverage=10.0)
    engine = BacktestEngine(cfg)
    result = engine.run({'X': {'1h': bars, '1d': pd.DataFrame(), 'funding': pd.DataFrame()}},
                        intents={'X': intents}, warmup=20)

    assert result.trades
    realised = sum(t.net_pnl for t in result.trades)
    equity_change = float(result.equity_curve.iloc[-1]) - cfg.initial_equity
    assert realised == pytest.approx(equity_change, abs=0.01)


def test_stop_loss_costs_one_r(zero_cost_config):
    """With costs off, a stopped-out trade must lose exactly 1R."""
    # Drift down so the stop at 95 is hit.
    closes = [100.0] * 5 + [94.0] * 5
    bars = make_bars(closes)
    intents = make_intents(bars.index,
                           {3: Intent('long', 1.0, stop_price=95.0, take_price=120.0)})

    engine = BacktestEngine(zero_cost_config)
    result = engine.run({'X': {'1h': bars, '1d': pd.DataFrame(), 'funding': pd.DataFrame()}},
                        intents={'X': intents}, warmup=2)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == 'stop'
    assert trade.r_multiple == pytest.approx(-1.0, abs=0.01)
    assert trade.net_pnl == pytest.approx(-100.0, rel=0.01)   # 1% of 10_000


def test_short_wins_when_price_falls(zero_cost_config):
    closes = [100.0] * 5 + [90.0] * 5
    bars = make_bars(closes)
    intents = make_intents(bars.index,
                           {3: Intent('short', 1.0, stop_price=105.0, take_price=90.0)})

    engine = BacktestEngine(zero_cost_config)
    result = engine.run({'X': {'1h': bars, '1d': pd.DataFrame(), 'funding': pd.DataFrame()}},
                        intents={'X': intents}, warmup=2)

    assert len(result.trades) == 1
    assert result.trades[0].side == 'short'
    assert result.trades[0].net_pnl > 0
    assert result.trades[0].exit_reason == 'target'


# ══════════════════════════════════════════════════════════════
# NO LOOKAHEAD
# ══════════════════════════════════════════════════════════════

def test_entry_fills_at_next_bar_open(zero_cost_config):
    """A signal on bar i must fill at bar i+1's open, never bar i's close."""
    closes = [100.0, 100.0, 100.0, 100.0, 150.0, 150.0, 150.0]
    bars = make_bars(closes)
    # Decide on bar 3 (close 100). Bar 4 opens at 150.
    intents = make_intents(bars.index,
                           {3: Intent('long', 1.0, stop_price=80.0, take_price=500.0)})

    engine = BacktestEngine(zero_cost_config)
    result = engine.run({'X': {'1h': bars, '1d': pd.DataFrame(), 'funding': pd.DataFrame()}},
                        intents={'X': intents}, warmup=2)

    assert len(result.trades) == 1
    # Filled at 150 (the next bar's open), NOT 100. Getting 100 would mean
    # the engine acted on a price it could not have known.
    assert result.trades[0].entry_price == pytest.approx(150.0)
    assert result.trades[0].entry_ts == bars.index[4]


def test_signal_on_final_bar_never_fills(zero_cost_config):
    """There is no next bar to fill against, so it must be dropped."""
    bars = make_bars([100.0] * 6)
    intents = make_intents(bars.index,
                           {5: Intent('long', 1.0, stop_price=90.0, take_price=110.0)})

    engine = BacktestEngine(zero_cost_config)
    result = engine.run({'X': {'1h': bars, '1d': pd.DataFrame(), 'funding': pd.DataFrame()}},
                        intents={'X': intents}, warmup=2)
    assert result.trades == []


def test_gap_through_stop_is_refused(zero_cost_config):
    """If the next open is already past the stop, the setup is void."""
    closes = [100.0, 100.0, 100.0, 100.0, 80.0, 80.0]
    bars = make_bars(closes)
    intents = make_intents(bars.index,
                           {3: Intent('long', 1.0, stop_price=95.0, take_price=110.0)})

    engine = BacktestEngine(zero_cost_config)
    result = engine.run({'X': {'1h': bars, '1d': pd.DataFrame(), 'funding': pd.DataFrame()}},
                        intents={'X': intents}, warmup=2)

    assert result.trades == []
    assert result.skipped.get('gapped_through_stop') == 1


# ══════════════════════════════════════════════════════════════
# EXIT PRECEDENCE
# ══════════════════════════════════════════════════════════════

def test_stop_wins_when_stop_and_target_share_a_bar(zero_cost_config):
    """
    OHLC cannot say which level was touched first, so the engine must take
    the pessimistic reading. Assuming the target would inflate every result.
    """
    bars = make_bars([100.0] * 6, high_pct=0.20, low_pct=0.20)
    intents = make_intents(bars.index,
                           {3: Intent('long', 1.0, stop_price=90.0, take_price=110.0)})

    engine = BacktestEngine(zero_cost_config)
    result = engine.run({'X': {'1h': bars, '1d': pd.DataFrame(), 'funding': pd.DataFrame()}},
                        intents={'X': intents}, warmup=2)

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == 'stop'


def test_time_stop_closes_a_stale_position():
    cfg = BacktestConfig(initial_equity=10_000.0, taker_fee=0.0, slippage_bps=0.0,
                         funding_enabled=False, time_stop_bars=3, use_trailing=False,
                         max_leverage=10.0)
    bars = make_bars([100.0] * 12)
    intents = make_intents(bars.index,
                           {2: Intent('long', 1.0, stop_price=90.0, take_price=110.0)})

    engine = BacktestEngine(cfg)
    result = engine.run({'X': {'1h': bars, '1d': pd.DataFrame(), 'funding': pd.DataFrame()}},
                        intents={'X': intents}, warmup=1)

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == 'time_stop'
    assert result.trades[0].bars_held == 3


def test_trailing_stop_does_not_arm_before_one_r():
    """
    A trail armed immediately turns noise into stop-outs. It must only
    engage after the position has actually run trailing_activate_r.
    """
    cfg = BacktestConfig(initial_equity=10_000.0, taker_fee=0.0, slippage_bps=0.0,
                         funding_enabled=False, time_stop_bars=0, use_trailing=True,
                         trailing_atr_mult=1.0, trailing_activate_r=1.0, max_leverage=10.0)
    # Entry 100, stop 90 → 1R = 10. Price wobbles to 104 (0.4R) then back.
    closes = [100.0, 100.0, 100.0, 104.0, 101.0, 100.0, 99.0, 99.0]
    bars = make_bars(closes)
    intents = make_intents(bars.index,
                           {1: Intent('long', 1.0, stop_price=90.0, take_price=200.0, atr=2.0)})

    engine = BacktestEngine(cfg)
    result = engine.run({'X': {'1h': bars, '1d': pd.DataFrame(), 'funding': pd.DataFrame()}},
                        intents={'X': intents}, warmup=0)

    # Never reached +1R, so the trail never armed and nothing stopped out.
    assert all(t.exit_reason != 'trailing_stop' for t in result.trades)


def test_funding_is_charged_to_longs():
    """Longs pay a positive funding rate; it must reduce equity."""
    bars = make_bars([100.0] * 12)
    funding = pd.DataFrame(
        {'funding_rate': [0.001]},
        index=[bars.index[4]],      # one settlement, inside the hold
    )
    intents = make_intents(bars.index,
                           {1: Intent('long', 1.0, stop_price=90.0, take_price=110.0)})

    cfg = BacktestConfig(initial_equity=10_000.0, taker_fee=0.0, slippage_bps=0.0,
                         funding_enabled=True, time_stop_bars=6, use_trailing=False,
                         max_leverage=10.0)
    engine = BacktestEngine(cfg)
    result = engine.run({'X': {'1h': bars, '1d': pd.DataFrame(), 'funding': funding}},
                        intents={'X': intents}, warmup=0)

    assert len(result.trades) == 1
    assert result.trades[0].funding > 0
    assert result.trades[0].net_pnl < 0        # nothing but carry cost


def test_max_positions_is_enforced():
    cfg = BacktestConfig(initial_equity=100_000.0, taker_fee=0.0, slippage_bps=0.0,
                         funding_enabled=False, time_stop_bars=0, use_trailing=False,
                         max_positions=2, max_leverage=10.0)
    bars = make_bars([100.0] * 10)
    data, intents = {}, {}
    for symbol in ('A', 'B', 'C'):
        data[symbol] = {'1h': bars, '1d': pd.DataFrame(), 'funding': pd.DataFrame()}
        intents[symbol] = make_intents(
            bars.index, {3: Intent('long', 1.0, stop_price=90.0, take_price=110.0)})

    engine = BacktestEngine(cfg)
    result = engine.run(data, intents=intents, warmup=2)

    # Only two may be open at once.
    assert len({t.symbol for t in result.trades}) <= 2
    assert result.skipped.get('max_positions', 0) >= 1


def test_limited_slots_go_to_the_strongest_signals():
    """
    With more symbols than slots, conviction decides — not list order.

    Once the universe grew past max_positions this became load-bearing:
    filling in watchlist order would hand every contested slot to whichever
    symbol happened to be listed first, regardless of signal strength.
    """
    cfg = BacktestConfig(initial_equity=100_000.0, taker_fee=0.0, slippage_bps=0.0,
                         funding_enabled=False, time_stop_bars=0, use_trailing=False,
                         max_positions=2, max_leverage=10.0)
    bars = make_bars([100.0] * 10)

    # 'A' is first in the dict but weakest; 'C' and 'D' are strongest.
    confidences = {'A': 0.30, 'B': 0.40, 'C': 0.95, 'D': 0.90}
    data, intents = {}, {}
    for symbol, confidence in confidences.items():
        data[symbol] = {'1h': bars, '1d': pd.DataFrame(), 'funding': pd.DataFrame()}
        intents[symbol] = make_intents(
            bars.index,
            {3: Intent('long', confidence, stop_price=90.0, take_price=110.0)})

    result = BacktestEngine(cfg).run(data, intents=intents, warmup=2)

    taken = {t.symbol for t in result.trades}
    assert taken == {'C', 'D'}, f"slots went to {taken}, expected the two best"
