# features/derivatives.py — signals that only exist on futures.
# ─────────────────────────────────────────────────────────────
# This is the payoff for migrating off spot. Funding rate and open interest
# are direct measurements of leveraged POSITIONING, and there is no spot
# equivalent — the old bot was structurally blind to them.
#
# Funding rate
#   The periodic payment between longs and shorts that tethers the perp to
#   spot. Persistently positive means longs are crowded and paying to stay
#   long. Crowded trades unwind violently: extreme funding marks the point
#   of maximum leverage, which is where liquidation cascades start. It is
#   also a real cost — an 8h rate of 0.05% is ~55% annualised.
#
# Open interest
#   Contracts outstanding. Read WITH price it separates real conviction
#   from noise, which price alone cannot do:
#
#     price ↑ + OI ↑   new longs        genuine demand, trend likely continues
#     price ↑ + OI ↓   short covering   fuel exhausting, rally likely fades
#     price ↓ + OI ↑   new shorts       genuine supply, downtrend continues
#     price ↓ + OI ↓   long liquidation capitulation, often near a low
#
# Both are used as CONTEXT, never as standalone entry triggers. Extreme
# funding can persist for weeks in a strong trend; shorting purely because
# funding is high is a well-known way to lose money slowly.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class DerivativesState:
    funding_rate:    float = 0.0     # latest 8h rate
    funding_z:       float = 0.0     # standard deviations from its own mean
    funding_annual:  float = 0.0     # annualised carry cost
    funding_extreme: bool = False    # crowded positioning
    funding_bias:    int = 0         # +1 fade-longs, -1 fade-shorts, 0 neutral

    oi_change_pct:   float = 0.0     # over the lookback window
    price_change_pct: float = 0.0
    oi_signal:       str = 'unknown'  # see the table above
    oi_bias:         int = 0         # +1 bullish, -1 bearish, 0 neutral

    available:       bool = False
    reason:          str = ''


def funding_zscore(funding: pd.Series, lookback: int = 90) -> tuple[float, float]:
    """
    Current funding rate and its z-score against recent history.

    A z-score rather than an absolute threshold because a "high" rate is
    asset- and era-specific: 0.01% per 8h is unremarkable for SOL in a bull
    market and extreme for BTC in a quiet one.
    """
    series = funding.dropna().tail(lookback)
    if len(series) < 20:
        return (float(series.iloc[-1]) if len(series) else 0.0), 0.0

    current = float(series.iloc[-1])
    mean, std = float(series.mean()), float(series.std())
    z = (current - mean) / std if std > 0 else 0.0
    return current, z


def analyse_derivatives(funding: Optional[pd.Series],
                        open_interest: Optional[pd.Series],
                        close: Optional[pd.Series],
                        oi_lookback: int = 24,
                        funding_lookback: int = 90,
                        extreme_z: float = 1.8) -> DerivativesState:
    """
    Build the derivatives context for one symbol.

    Every input is optional — Bybit only retains ~6 months of open interest,
    so a two-year backtest has none for most of its span. Missing data must
    read as "no information", never as zero, or the absence itself becomes a
    fake signal.
    """
    state = DerivativesState()

    # ── funding ───────────────────────────────────────────────
    if funding is not None and len(funding.dropna()) >= 3:
        rate, z = funding_zscore(funding, funding_lookback)
        state.funding_rate = rate
        state.funding_z = z
        state.funding_annual = rate * 3 * 365      # 3 settlements per day
        state.funding_extreme = abs(z) >= extreme_z
        state.available = True

        if state.funding_extreme:
            # Crowded longs paying up → lean against them, and vice versa.
            state.funding_bias = -1 if z > 0 else +1
    else:
        state.reason = 'no funding history'

    # ── open interest ─────────────────────────────────────────
    if (open_interest is not None and close is not None
            and len(open_interest.dropna()) > oi_lookback
            and len(close) > oi_lookback):

        oi = open_interest.dropna()
        oi_now, oi_then = float(oi.iloc[-1]), float(oi.iloc[-oi_lookback - 1])
        px_now, px_then = float(close.iloc[-1]), float(close.iloc[-oi_lookback - 1])

        if oi_then > 0 and px_then > 0:
            state.oi_change_pct = (oi_now - oi_then) / oi_then
            state.price_change_pct = (px_now - px_then) / px_then
            state.available = True

            # 0.5% either way is noise, not a regime change.
            oi_up = state.oi_change_pct > 0.005
            oi_down = state.oi_change_pct < -0.005
            px_up = state.price_change_pct > 0.005
            px_down = state.price_change_pct < -0.005

            if px_up and oi_up:
                state.oi_signal, state.oi_bias = 'new_longs', +1
            elif px_up and oi_down:
                state.oi_signal, state.oi_bias = 'short_covering', -1
            elif px_down and oi_up:
                state.oi_signal, state.oi_bias = 'new_shorts', -1
            elif px_down and oi_down:
                state.oi_signal, state.oi_bias = 'long_liquidation', +1
            else:
                state.oi_signal, state.oi_bias = 'flat', 0
    elif not state.reason:
        state.reason = 'no open-interest history'

    return state


def align_funding_to_bars(funding: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """
    Forward-fill 8-hourly funding onto an hourly bar index.

    Uses merge_asof semantics: each bar sees the most recent rate that had
    ALREADY SETTLED at that point. A plain reindex would let a bar see the
    rate from a settlement still in its future.
    """
    if funding is None or funding.empty:
        return pd.Series(np.nan, index=index)

    series = funding['funding_rate'] if 'funding_rate' in funding else funding.iloc[:, 0]
    series = series.sort_index()
    return series.reindex(index.union(series.index)).ffill().reindex(index)


def align_oi_to_bars(open_interest: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill open interest onto a bar index, same no-peek rule."""
    if open_interest is None or open_interest.empty:
        return pd.Series(np.nan, index=index)

    series = (open_interest['open_interest'] if 'open_interest' in open_interest
              else open_interest.iloc[:, 0])
    series = series.sort_index()
    return series.reindex(index.union(series.index)).ffill().reindex(index)
