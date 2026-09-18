"""
The two v7 code paths must agree.

Live trading goes through strategy_v7.decide() bar by bar. Walk-forward
tuning goes through feature_stream.intents_from_features(), a vectorised
re-implementation that scores a whole series at once — that shortcut is the
only reason out-of-sample tuning is feasible at all.

If the two ever drift apart, the tuner optimises parameters for a strategy
that is not the one that trades, and nothing else in the suite would catch
it. These tests pin them together on real market data.
"""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import config
from backtest.feature_stream import build_feature_stream, intents_from_features
from data import get_funding_history, get_klines
from features.derivatives import align_funding_to_bars
from strategy_v7 import DEFAULT_PARAMS, decide

SYMBOL = 'BTCUSDT'
DAYS = 45


@pytest.fixture(scope='module')
def market():
    bars = get_klines(SYMBOL, '60', days=DAYS)
    daily = get_klines(SYMBOL, 'D', days=400)
    funding = get_funding_history(SYMBOL, days=120)
    if bars.empty:
        pytest.skip("no market data available")
    return bars, daily, funding


@pytest.fixture(scope='module')
def streams(market):
    bars, daily, funding = market
    data = {SYMBOL: {'1h': bars, '1d': daily, 'funding': funding,
                     'oi': pd.DataFrame()}}
    warmup = max(0, len(bars) - 120)      # keep the fixture fast
    features = build_feature_stream([SYMBOL], data, warmup=warmup,
                                    use_cache=False, workers=2)
    return features[SYMBOL], bars, daily, funding


PARAM_VARIANTS = {
    'defaults':        DEFAULT_PARAMS,
    # Every optional gate has to be checked in BOTH implementations, or a
    # parameter the tuner is free to select could behave differently live.
    'kill_zones_on':   replace(DEFAULT_PARAMS, use_kill_zones=True),
    'no_funding_gate': replace(DEFAULT_PARAMS, use_funding_gate=False),
    'no_btc_anchor':   replace(DEFAULT_PARAMS, use_btc_anchor=False),
    'squeeze_on':      replace(DEFAULT_PARAMS, trade_squeeze=True),
    'strict':          replace(DEFAULT_PARAMS, min_net=0.35, min_agreement=0.80,
                               min_active_blocks=5, atr_stop_mult=3.5,
                               atr_target_mult=7.0),
}


@pytest.mark.parametrize('variant', list(PARAM_VARIANTS))
def test_feature_stream_matches_the_live_decision_path(streams, variant):
    """
    Same bars, same parameters → same direction.

    Checked across a real 120-bar window rather than a synthetic case,
    because the disagreements that matter come from regime boundaries and
    missing-data handling, not from clean inputs.
    """
    params = PARAM_VARIANTS[variant]
    features, bars, daily, funding = streams
    funding_series = align_funding_to_bars(funding, bars.index)

    vector = intents_from_features(features, params, SYMBOL)

    mismatches = []
    for ts in features.index:
        i = bars.index.get_loc(ts)
        hist_1h = bars.iloc[max(0, i - config.ANALYSIS_WINDOW_BARS + 1): i + 1]
        hist_1d = daily[daily.index <= ts - pd.Timedelta(days=1)]

        live = decide(SYMBOL, hist_1h, hist_1d, ts, None,
                      funding=funding_series.loc[:ts], params=params)
        vectorised = str(vector.at[ts, 'direction'])

        if live.direction != vectorised:
            mismatches.append((ts, live.direction, vectorised, live.rejected))

    if mismatches:
        detail = '\n'.join(f"  {ts}: live={a!r} vectorised={b!r} ({why})"
                           for ts, a, b, why in mismatches[:10])
        pytest.fail(f"[{variant}] {len(mismatches)}/{len(features)} bars disagree:\n{detail}")


def test_stop_and_target_agree_where_both_act(streams):
    """A shared direction must also mean a shared stop and target."""
    features, bars, daily, funding = streams
    funding_series = align_funding_to_bars(funding, bars.index)
    vector = intents_from_features(features, DEFAULT_PARAMS, SYMBOL)

    from strategy_v7 import v7_intent

    compared = 0
    for ts in features.index:
        if str(vector.at[ts, 'direction']) == 'flat':
            continue

        i = bars.index.get_loc(ts)
        hist_1h = bars.iloc[max(0, i - config.ANALYSIS_WINDOW_BARS + 1): i + 1]
        hist_1d = daily[daily.index <= ts - pd.Timedelta(days=1)]

        live = v7_intent(SYMBOL, hist_1h, hist_1d, ts, None,
                         funding=funding_series.loc[:ts], params=DEFAULT_PARAMS)
        assert live is not None, f"live path returned nothing at {ts}"

        assert live.stop_price == pytest.approx(float(vector.at[ts, 'stop_price']), rel=1e-6)
        assert live.take_price == pytest.approx(float(vector.at[ts, 'take_price']), rel=1e-6)
        assert live.confidence == pytest.approx(float(vector.at[ts, 'confidence']), rel=1e-4)
        compared += 1

    if compared == 0:
        pytest.skip("no actionable bars in this window to compare")
