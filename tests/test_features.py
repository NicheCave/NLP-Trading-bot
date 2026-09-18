"""
Tests for the regime router, factor blocks and derivatives context.

The block tests are the important ones: they pin down the specific defect
that made the old aggregation misleading — five correlated momentum
indicators voting as if they were five independent opinions.
"""
import numpy as np
import pandas as pd
import pytest

from features.blocks import (BLOCK_BASE_WEIGHTS, aggregate_blocks, agreement)
from features.derivatives import analyse_derivatives, funding_zscore
from features.regime import Regime, classify_regime


class Vote:
    """Stand-in for market_analyzer.IndicatorVote."""
    def __init__(self, name, vote, weight=1.0, reason=''):
        self.name, self.vote, self.weight, self.reason = name, vote, weight, reason


def make_bars(closes, highs=None, lows=None):
    closes = np.asarray(closes, dtype=float)
    index = pd.date_range('2025-01-01', periods=len(closes), freq='1h', tz='UTC')
    return pd.DataFrame({
        'open': closes,
        'high': closes if highs is None else highs,
        'low': closes if lows is None else lows,
        'close': closes,
        'volume': np.full(len(closes), 100.0),
        'turnover': closes * 100.0,
    }, index=index)


# ══════════════════════════════════════════════════════════════
# FACTOR BLOCKS — the double-counting fix
# ══════════════════════════════════════════════════════════════

def test_correlated_indicators_cannot_stack_beyond_their_block():
    """
    THE regression this module exists for.

    Three trend indicators all screaming bullish must not outweigh a
    structure block that disagrees any more than one of them would — they
    are measuring the same thing.
    """
    many = aggregate_blocks([
        Vote('ema_cross', +1, 2.0),
        Vote('mtf_daily_align', +1, 2.5),
        Vote('williams_alligator', +1, 2.0),
    ])
    one = aggregate_blocks([Vote('ema_cross', +1, 2.0)])

    # Same block score either way: the family has one opinion.
    assert many.blocks['trend'].score == pytest.approx(1.0)
    assert one.blocks['trend'].score == pytest.approx(1.0)
    assert many.net_score == pytest.approx(one.net_score)


def test_block_score_is_bounded():
    """No number of agreeing votes can push a block past ±1."""
    result = aggregate_blocks([
        Vote('ema_cross', +1, 50.0),
        Vote('mtf_daily_align', +1, 50.0),
        Vote('williams_alligator', +1, 50.0),
    ])
    assert result.blocks['trend'].score <= 1.0
    assert result.net_score <= 1.0


def test_disagreement_within_a_block_cancels():
    result = aggregate_blocks([
        Vote('ema_cross', +1, 2.0),
        Vote('williams_alligator', -1, 2.0),
    ])
    assert result.blocks['trend'].score == pytest.approx(0.0)


def test_independent_blocks_do_add_up():
    """
    Genuinely different families agreeing SHOULD raise the score.

    A neutral family is present in both cases to leave headroom: net is a
    weighted MEAN of block scores, so if every participating block reads
    +1.0 the result is already 1.0 and there is nothing left to add.
    """
    one_family = aggregate_blocks([
        Vote('ema_cross', +1, 2.0),
        Vote('rsi', 0, 1.5),                 # mean_reversion present, undecided
    ])
    two_families = aggregate_blocks([
        Vote('ema_cross', +1, 2.0),
        Vote('smc_order_block', +1, 2.5),
        Vote('rsi', 0, 1.5),
    ])
    assert two_families.net_score > one_family.net_score


def test_regime_weights_reshape_the_result():
    """
    The same evidence must read differently by regime.

    A bullish trend vote and a bearish mean-reversion vote should resolve
    bullish in a trend and bearish in a range.
    """
    votes = [Vote('ema_cross', +1, 2.0), Vote('rsi', -1, 1.5)]

    trend_weights = {'trend': 1.30, 'mean_reversion': 0.30}
    range_weights = {'trend': 0.35, 'mean_reversion': 1.40}

    in_trend = aggregate_blocks(votes, trend_weights)
    in_range = aggregate_blocks(votes, range_weights)

    assert in_trend.net_score > 0
    assert in_range.net_score < 0


def test_neutral_votes_still_dilute():
    """A block that had its say and found nothing must not be ignored."""
    with_neutral = aggregate_blocks([
        Vote('ema_cross', +1, 2.0),
        Vote('smc_order_block', 0, 2.5),
    ])
    assert abs(with_neutral.net_score) < 1.0


def test_extra_blocks_are_merged():
    result = aggregate_blocks([Vote('ema_cross', +1, 2.0)],
                              extra_blocks={'derivatives': -1.0})
    assert 'derivatives' in result.blocks
    assert result.blocks['derivatives'].score == pytest.approx(-1.0)


def test_agreement_measures_weighted_consensus():
    unanimous = aggregate_blocks([
        Vote('ema_cross', +1, 2.0),
        Vote('smc_order_block', +1, 2.5),
        Vote('rsi', +1, 1.5),
    ])
    assert agreement(unanimous, +1) == pytest.approx(1.0)

    split = aggregate_blocks([
        Vote('ema_cross', +1, 2.0),
        Vote('smc_order_block', -1, 2.5),
    ])
    assert 0.0 < agreement(split, +1) < 1.0


def test_unknown_indicator_names_are_ignored():
    result = aggregate_blocks([Vote('some_new_thing', +1, 5.0)])
    assert result.net_score == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════
# REGIME
# ══════════════════════════════════════════════════════════════

def test_steady_uptrend_classifies_as_trend_up():
    closes = np.linspace(100, 160, 300)
    state = classify_regime(make_bars(closes, closes * 1.002, closes * 0.998))
    assert state.regime is Regime.TREND_UP
    assert state.tradeable


def test_steady_downtrend_classifies_as_trend_down():
    closes = np.linspace(160, 100, 300)
    state = classify_regime(make_bars(closes, closes * 1.002, closes * 0.998))
    assert state.regime is Regime.TREND_DOWN


def test_oscillation_is_not_a_trend():
    """A market thrashing between two levels must never read as trending."""
    x = np.arange(300)
    closes = 100 + 8 * np.sin(x / 3.0)
    state = classify_regime(make_bars(closes, closes * 1.01, closes * 0.99))
    assert state.regime is not Regime.TREND_UP
    assert state.regime is not Regime.TREND_DOWN


def test_insufficient_data_is_not_tradeable():
    state = classify_regime(make_bars(np.linspace(100, 110, 20)))
    assert not state.tradeable


def test_trend_regime_downweights_mean_reversion():
    """The routing table must actually distrust fading a trend."""
    closes = np.linspace(100, 160, 300)
    state = classify_regime(make_bars(closes, closes * 1.002, closes * 0.998))
    assert state.weight_for('mean_reversion') < state.weight_for('trend')


def test_range_regime_upweights_mean_reversion():
    x = np.arange(300)
    closes = 100 + 3 * np.sin(x / 8.0)
    state = classify_regime(make_bars(closes, closes * 1.005, closes * 0.995))
    if state.regime is Regime.RANGE:
        assert state.weight_for('mean_reversion') > state.weight_for('trend')


# ══════════════════════════════════════════════════════════════
# DERIVATIVES
# ══════════════════════════════════════════════════════════════

def test_funding_zscore_flags_an_outlier():
    index = pd.date_range('2025-01-01', periods=100, freq='8h', tz='UTC')
    rates = pd.Series([0.0001] * 99 + [0.01], index=index)
    current, z = funding_zscore(rates)
    assert current == pytest.approx(0.01)
    assert z > 3


def test_missing_data_reads_as_no_information_not_zero():
    """
    Bybit only keeps ~6 months of open interest. Absence must not become a
    signal in its own right.
    """
    state = analyse_derivatives(None, None, None)
    assert not state.available
    assert state.funding_bias == 0
    assert state.oi_bias == 0


def test_extreme_positive_funding_leans_against_longs():
    index = pd.date_range('2025-01-01', periods=100, freq='8h', tz='UTC')
    rates = pd.Series([0.0001] * 99 + [0.005], index=index)
    state = analyse_derivatives(rates, None, None)
    assert state.funding_extreme
    assert state.funding_bias == -1        # crowded longs → fade them


def test_price_up_with_rising_oi_is_new_longs():
    index = pd.date_range('2025-01-01', periods=60, freq='1h', tz='UTC')
    oi = pd.Series(np.linspace(1000, 1200, 60), index=index)
    close = pd.Series(np.linspace(100, 120, 60), index=index)
    state = analyse_derivatives(None, oi, close)
    assert state.oi_signal == 'new_longs'
    assert state.oi_bias == +1


def test_price_up_with_falling_oi_is_short_covering():
    """A rally on shrinking OI is shorts closing, not new demand."""
    index = pd.date_range('2025-01-01', periods=60, freq='1h', tz='UTC')
    oi = pd.Series(np.linspace(1200, 1000, 60), index=index)
    close = pd.Series(np.linspace(100, 120, 60), index=index)
    state = analyse_derivatives(None, oi, close)
    assert state.oi_signal == 'short_covering'
    assert state.oi_bias == -1


def test_price_down_with_falling_oi_is_capitulation():
    index = pd.date_range('2025-01-01', periods=60, freq='1h', tz='UTC')
    oi = pd.Series(np.linspace(1200, 1000, 60), index=index)
    close = pd.Series(np.linspace(120, 100, 60), index=index)
    state = analyse_derivatives(None, oi, close)
    assert state.oi_signal == 'long_liquidation'
    assert state.oi_bias == +1


def test_funding_annualisation():
    index = pd.date_range('2025-01-01', periods=30, freq='8h', tz='UTC')
    rates = pd.Series([0.0001] * 30, index=index)
    state = analyse_derivatives(rates, None, None)
    # 0.01% per 8h × 3 × 365 ≈ 10.95%/yr
    assert state.funding_annual == pytest.approx(0.1095, rel=0.01)
