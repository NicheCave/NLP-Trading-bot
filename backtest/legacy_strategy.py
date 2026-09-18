# backtest/legacy_strategy.py — the v6 strategy, made replayable.
# ─────────────────────────────────────────────────────────────
# This wraps the EXISTING indicator stack (market_analyzer + signal_generator)
# without changing a single threshold, so the backtest measures the strategy
# as it actually trades today. That number is the baseline every later change
# has to beat.
#
# Two variants are exposed, and the difference between them is itself a
# measurement worth having:
#
#   allow_shorts=False  the bot as it stands — SELL just means "don't be
#                       long", because spot can't short. Roughly half the
#                       indicator stack votes bearish into the void.
#   allow_shorts=True   identical signals, but SELL opens a short. This is
#                       the single change futures makes possible, isolated.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

from datetime import timezone
from typing import Optional

import pandas as pd

import config
from indicators import atr as calc_atr
from market_analyzer import analyze_frames
from signal_generator import signal_from_analysis

from .engine import Intent


def btc_trend_series(df_1h: pd.DataFrame) -> pd.Series:
    """
    Vectorised equivalent of market_analyzer._trend_from_ema_df.

    The live bot recomputes the anchor trend per ticker per cycle behind a
    5-minute cache; over a two-year replay that would be ~17.5k redundant
    passes. Same thresholds, computed once.
    """
    fast = df_1h['close'].ewm(span=config.EMA_FAST, adjust=False).mean()
    slow = df_1h['close'].ewm(span=config.EMA_SLOW, adjust=False).mean()

    trend = pd.Series('sideways', index=df_1h.index, dtype=object)
    trend[fast > slow * 1.002] = 'uptrend'
    trend[fast < slow * 0.998] = 'downtrend'
    # Not enough history for the EMAs to mean anything yet.
    trend.iloc[:config.EMA_SLOW + 5] = 'sideways'
    return trend


def legacy_intent(symbol: str,
                  hist_1h: pd.DataFrame,
                  hist_1d: pd.DataFrame,
                  ts: pd.Timestamp,
                  btc_trend: Optional[str],
                  allow_shorts: bool = False,
                  stop_pct: float = 0.05,
                  take_pct: float = 0.07) -> Optional[Intent]:
    """
    Run one bar of the legacy strategy and translate it into an Intent.

    `hist_1h` / `hist_1d` must contain only bars closed at or before `ts`.
    """
    analysis = analyze_frames(symbol, hist_1h, hist_1d, btc_trend)
    if analysis is None:
        return None

    signal = signal_from_analysis(symbol, analysis, ts.to_pydatetime().replace(tzinfo=timezone.utc))
    if signal.action == 'HOLD':
        return None
    if signal.action == 'SELL' and not allow_shorts:
        return None

    close = float(hist_1h['close'].iloc[-1])
    direction = 'long' if signal.action == 'BUY' else 'short'

    # The live bot's fixed percentage stop/target, expressed as prices.
    if direction == 'long':
        stop = close * (1 - stop_pct)
        take = close * (1 + take_pct)
    else:
        stop = close * (1 + stop_pct)
        take = close * (1 - take_pct)

    atr_result = calc_atr(hist_1h, period=config.ATR_PERIOD)
    atr_value = atr_result.value if atr_result is not None else None

    return Intent(
        direction=direction,
        confidence=signal.confidence,
        stop_price=stop,
        take_price=take,
        atr=atr_value,
        reason=signal.reason[:200],
        tags=_tags(signal),
    )


def _tags(signal) -> list[str]:
    """
    Feature tags for per-tag win-rate attribution.

    Mirrors EdgeTracker._extract_tags in main.py so the backtest and the live
    edge log speak the same vocabulary.
    """
    tags: list[str] = []
    reason = signal.reason.lower()

    for needle, tag in (
        ('choch', 'CHoCH'),
        ('liquidity sweep', 'sweep'),
        ('order block', 'OB'),
        ('fvg', 'FVG'),
        ('bos', 'BOS'),
        ('avwap', 'AVWAP'),
        ('alligator', 'Alligator'),
        ('ao zero-line', 'AO_cross'),
        ('fractal', 'Fractal'),
        ('wave 3', 'W3_confirmed'),
        ('bear market', 'bear_ctx'),
    ):
        if needle in reason:
            tags.append(tag)

    if signal.trap_flip:
        tags.append('TrapFlip')
    if signal.wave5_caution:
        tags.append('W5_caution')

    if signal.confidence >= 0.8:
        tags.append('conf_high')
    elif signal.confidence >= 0.5:
        tags.append('conf_mid')
    else:
        tags.append('conf_low')

    return tags
