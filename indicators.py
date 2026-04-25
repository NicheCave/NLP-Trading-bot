# indicators.py — pure technical indicator functions + SMC detectors
# ─────────────────────────────────────────────────────────────
# All functions take a pandas DataFrame (OHLCV) and return
# a scalar or small dict — no side effects, no state.
#
# Changes from v1:
#   - MACD: fixed divergence to require zero-line cross, not just histogram expansion
#   - RSI:  fixed neutral zone — bearish_zone now casts 0 vote (not -0.5)
#   - trend_structure: extended regression window from 30 → 80 bars for daily use
#   - NEW: order_blocks()       — institutional OB detection
#   - NEW: fair_value_gaps()    — FVG / imbalance detection
#   - NEW: break_of_structure() — BOS / CHoCH event detection
#   - NEW: liquidity_sweep()    — stop-hunt detection
# ─────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────
# TREND INDICATORS
# ──────────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


@dataclass
class EMASignal:
    fast: float
    slow: float
    cross: str        # 'golden' | 'death' | 'above' | 'below'
    slope_pct: float  # % change of fast EMA over last 3 bars


def ema_cross(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> Optional[EMASignal]:
    if len(df) < slow + 5:
        return None
    close = df['close']
    f = ema(close, fast)
    s = ema(close, slow)

    prev_above = f.iloc[-2] > s.iloc[-2]
    curr_above = f.iloc[-1] > s.iloc[-1]

    if not prev_above and curr_above:
        cross = 'golden'
    elif prev_above and not curr_above:
        cross = 'death'
    elif curr_above:
        cross = 'above'
    else:
        cross = 'below'

    slope_pct = (f.iloc[-1] - f.iloc[-4]) / f.iloc[-4] * 100 if f.iloc[-4] != 0 else 0.0

    return EMASignal(fast=float(f.iloc[-1]), slow=float(s.iloc[-1]),
                     cross=cross, slope_pct=slope_pct)


@dataclass
class MACDResult:
    macd: float       # MACD line value
    signal: float     # signal line value
    hist: float       # histogram (macd - signal)
    prev_hist: float  # previous bar histogram
    divergence: str   # 'bullish_cross' | 'bearish_cross' | 'bullish' | 'bearish' | 'neutral'
    # FIX: divergence now requires zero-line cross for _cross variants;
    #      plain 'bullish'/'bearish' = histogram expanding on correct side (weaker signal)


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[MACDResult]:
    if len(df) < slow + signal + 5:
        return None
    close = df['close']
    macd_line   = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist        = macd_line - signal_line

    h_curr  = hist.iloc[-1]
    h_prev  = hist.iloc[-2]
    h_prev2 = hist.iloc[-3] if len(hist) >= 3 else h_prev

    # Primary: zero-line cross (strongest signal — MACD line crosses signal line)
    if h_prev <= 0 and h_curr > 0:
        divergence = 'bullish_cross'
    elif h_prev >= 0 and h_curr < 0:
        divergence = 'bearish_cross'
    # Secondary: histogram expanding in the correct direction (momentum confirmation)
    elif h_curr > 0 and h_curr > h_prev > h_prev2:
        divergence = 'bullish'
    elif h_curr < 0 and h_curr < h_prev < h_prev2:
        divergence = 'bearish'
    else:
        divergence = 'neutral'

    return MACDResult(
        macd=float(macd_line.iloc[-1]),
        signal=float(signal_line.iloc[-1]),
        hist=float(h_curr),
        prev_hist=float(h_prev),
        divergence=divergence,
    )


# ──────────────────────────────────────────────────────────────
# MOMENTUM INDICATORS
# ──────────────────────────────────────────────────────────────

@dataclass
class RSIResult:
    value: float
    zone: str  # 'oversold' | 'overbought' | 'bullish_zone' | 'bearish_zone' | 'neutral'
    # FIX: 'neutral' is now used for RSI 45–55 (truly flat); bearish_zone no longer
    #      auto-votes -1 in market_analyzer — it casts 0 (see market_analyzer.py).


def rsi(df: pd.DataFrame, period: int = 14) -> Optional[RSIResult]:
    if len(df) < period + 2:
        return None
    close    = df['close']
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val  = float((100 - 100 / (1 + rs)).iloc[-1])

    if rsi_val <= 30:
        zone = 'oversold'
    elif rsi_val >= 70:
        zone = 'overbought'
    elif rsi_val >= 55:
        zone = 'bullish_zone'
    elif rsi_val <= 45:
        zone = 'bearish_zone'
    else:
        zone = 'neutral'   # 45–55: genuinely flat, cast no vote

    return RSIResult(value=rsi_val, zone=zone)


# ──────────────────────────────────────────────────────────────
# VOLATILITY INDICATORS
# ──────────────────────────────────────────────────────────────

@dataclass
class BollingerResult:
    upper: float
    middle: float
    lower: float
    width_pct: float      # band width as % of middle
    price_position: float # 0.0 = at lower band, 1.0 = at upper band
    signal: str           # 'squeeze' | 'breakout_up' | 'breakout_down' | 'normal'


def bollinger_bands(df: pd.DataFrame, period: int = 20,
                    std_dev: float = 2.0) -> Optional[BollingerResult]:
    if len(df) < period + 2:
        return None
    close = df['close']
    mid   = sma(close, period)
    std   = close.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std

    w_pct = float((upper.iloc[-1] - lower.iloc[-1]) / mid.iloc[-1] * 100)
    price = close.iloc[-1]
    band_range = upper.iloc[-1] - lower.iloc[-1]
    pos = float((price - lower.iloc[-1]) / band_range) if band_range != 0 else 0.5

    if w_pct < 4.0:
        sig = 'squeeze'
    elif price > upper.iloc[-1]:
        sig = 'breakout_up'
    elif price < lower.iloc[-1]:
        sig = 'breakout_down'
    else:
        sig = 'normal'

    return BollingerResult(
        upper=float(upper.iloc[-1]),
        middle=float(mid.iloc[-1]),
        lower=float(lower.iloc[-1]),
        width_pct=w_pct,
        price_position=pos,
        signal=sig,
    )


@dataclass
class ATRResult:
    value: float   # Average True Range ($)
    pct: float     # ATR as % of close
    regime: str    # 'low' | 'normal' | 'high'


def atr(df: pd.DataFrame, period: int = 14) -> Optional[ATRResult]:
    if len(df) < period + 5:
        return None
    high       = df['high']
    low        = df['low']
    close      = df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_series      = tr.ewm(com=period - 1, adjust=False).mean()
    atr_val         = float(atr_series.iloc[-1])
    pct             = atr_val / float(close.iloc[-1]) * 100
    recent_atr_avg  = float(atr_series.tail(50).mean())

    if atr_val < recent_atr_avg * 0.75:
        regime = 'low'
    elif atr_val > recent_atr_avg * 1.5:
        regime = 'high'
    else:
        regime = 'normal'

    return ATRResult(value=atr_val, pct=pct, regime=regime)


# ──────────────────────────────────────────────────────────────
# VOLUME INDICATORS
# ──────────────────────────────────────────────────────────────

@dataclass
class VolumeResult:
    current: float
    avg_20: float
    ratio: float   # current / avg_20
    spike: bool    # True if ratio >= 1.5
    direction: str # 'bullish' | 'bearish' | 'neutral' — based on candle close vs open
    # FIX: volume now carries its own direction, independent of EMA state


def volume_analysis(df: pd.DataFrame) -> Optional[VolumeResult]:
    if 'volume' not in df.columns or len(df) < 21:
        return None
    vol   = df['volume']
    avg   = float(vol.rolling(20).mean().iloc[-1])
    curr  = float(vol.iloc[-1])
    ratio = curr / avg if avg > 0 else 1.0

    # Determine direction from the candle itself (not from EMA state)
    last_open  = float(df['open'].iloc[-1])
    last_close = float(df['close'].iloc[-1])
    if last_close > last_open * 1.001:
        direction = 'bullish'
    elif last_close < last_open * 0.999:
        direction = 'bearish'
    else:
        direction = 'neutral'

    return VolumeResult(current=curr, avg_20=avg, ratio=ratio,
                        spike=ratio >= 1.5, direction=direction)


# ──────────────────────────────────────────────────────────────
# SWING LEVEL DETECTION
# ──────────────────────────────────────────────────────────────

@dataclass
class SwingLevel:
    level: float
    label: str         # 'support' | 'resistance'
    distance_pct: float
    direction: str     # 'above' | 'below'


def detect_swing_levels(df: pd.DataFrame, lookback: int = 50) -> list[SwingLevel]:
    if len(df) < lookback:
        return []

    close   = df['close']
    high    = df['high']
    low     = df['low']
    current = float(close.iloc[-1])
    window  = 5
    levels  = []

    for i in range(window, len(df) - window):
        if all(high.iloc[i] >= high.iloc[i - j] for j in range(1, window + 1)) and \
           all(high.iloc[i] >= high.iloc[i + j] for j in range(1, window + 1)):
            lvl = float(high.iloc[i])
            levels.append(SwingLevel(
                level=lvl, label='resistance',
                distance_pct=abs(lvl - current) / current * 100,
                direction='above' if lvl > current else 'below',
            ))

        if all(low.iloc[i] <= low.iloc[i - j] for j in range(1, window + 1)) and \
           all(low.iloc[i] <= low.iloc[i + j] for j in range(1, window + 1)):
            lvl = float(low.iloc[i])
            levels.append(SwingLevel(
                level=lvl, label='support',
                distance_pct=abs(lvl - current) / current * 100,
                direction='below' if lvl < current else 'above',
            ))

    merged = []
    for lvl in sorted(levels, key=lambda x: x.level):
        if not merged or abs(lvl.level - merged[-1].level) / merged[-1].level > 0.01:
            merged.append(lvl)

    supports = sorted([l for l in merged if l.label == 'support'],   key=lambda x: x.distance_pct)[:3]
    resists  = sorted([l for l in merged if l.label == 'resistance'], key=lambda x: x.distance_pct)[:3]
    return supports + resists


# ──────────────────────────────────────────────────────────────
# FIBONACCI RETRACEMENTS
# ──────────────────────────────────────────────────────────────

FIBO_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]


@dataclass
class FiboResult:
    swing_high: float
    swing_low: float
    levels: dict         # ratio -> price level
    nearest_level: float # ratio of nearest fib
    nearest_price: float # price of that fib level
    distance_pct: float  # % distance from current price to nearest fib
    at_support: bool     # True if price is near a fib AND below it (fib acting as resistance-turned-support)
    at_resistance: bool  # True if price is near a fib AND above it


def fibonacci_retracements(df: pd.DataFrame, lookback: int = 100) -> Optional[FiboResult]:
    """
    FIX: Previously the support/resistance logic was inverted.
    Now correctly identifies whether price is approaching a fib from below
    (resistance) or has pulled back to a fib from above (support).
    """
    if len(df) < lookback:
        return None

    window     = df.tail(lookback)
    swing_high = float(window['high'].max())
    swing_low  = float(window['low'].min())
    current    = float(df['close'].iloc[-1])
    rng        = swing_high - swing_low

    if rng == 0:
        return None

    # Fib levels are retracements FROM high DOWN toward low
    # fib_price = swing_high - ratio * range
    # So ratio=0.382 → price is 38.2% of the way down from the high
    levels = {ratio: swing_high - ratio * rng for ratio in FIBO_LEVELS}

    nearest       = min(FIBO_LEVELS, key=lambda r: abs(levels[r] - current))
    nearest_price = levels[nearest]
    distance_pct  = abs(nearest_price - current) / current * 100

    # Price is BELOW the fib level → fib is acting as resistance above
    # Price is ABOVE the fib level → price has retraced to fib and may bounce (support)
    at_support    = current >= nearest_price and distance_pct < 2.0
    at_resistance = current < nearest_price  and distance_pct < 2.0

    return FiboResult(
        swing_high=swing_high,
        swing_low=swing_low,
        levels=levels,
        nearest_level=nearest,
        nearest_price=nearest_price,
        distance_pct=distance_pct,
        at_support=at_support,
        at_resistance=at_resistance,
    )


# ──────────────────────────────────────────────────────────────
# TREND STRUCTURE
# FIX: regression window extended from 30 → 80 bars for daily charts
# ──────────────────────────────────────────────────────────────

@dataclass
class TrendStructure:
    direction: str    # 'uptrend' | 'downtrend' | 'ranging'
    strength: float   # R² — 0.0 (weak) to 1.0 (strong)
    higher_highs: bool
    higher_lows: bool
    channel_break: str  # 'broke_above' | 'broke_below' | 'none'


def trend_structure(df: pd.DataFrame, pivots: int = 5,
                    regression_bars: int = 80) -> Optional[TrendStructure]:
    """
    FIX: regression_bars default raised to 80 (was 30).
    On a daily chart 30 bars = 30 days — too short to identify a major trend.
    80 bars = ~3 months, which captures real market structure cycles.
    """
    n = min(regression_bars, len(df))
    if n < 10:
        return None

    close = df['close']
    x     = np.arange(n)
    y     = close.iloc[-n:].values

    coeffs = np.polyfit(x, y, 1)
    slope  = coeffs[0]
    y_pred = coeffs[0] * x + coeffs[1]
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2     = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    highs = df['high'].rolling(3).max().dropna().iloc[-pivots:]
    lows  = df['low'].rolling(3).min().dropna().iloc[-pivots:]
    hh    = bool(highs.is_monotonic_increasing)
    hl    = bool(lows.is_monotonic_increasing)

    if slope > 0 and r2 > 0.5:
        direction = 'uptrend'
    elif slope < 0 and r2 > 0.5:
        direction = 'downtrend'
    else:
        direction = 'ranging'

    return TrendStructure(
        direction=direction,
        strength=r2,
        higher_highs=hh,
        higher_lows=hl,
        channel_break='none',
    )


# ══════════════════════════════════════════════════════════════
# SMART MONEY CONCEPTS (SMC) — new in v2
# ══════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────
# 1. ORDER BLOCKS
# An order block is the last opposing candle before a significant
# impulsive move. Institutions place limit orders here — price
# tends to react strongly when it returns to these zones.
# ──────────────────────────────────────────────────────────────

@dataclass
class OrderBlock:
    ob_type: str        # 'bullish' | 'bearish'
    top: float          # upper boundary of the OB
    bottom: float       # lower boundary
    mid: float          # midpoint
    index: int          # bar index in df where the OB was formed
    distance_pct: float # % from current price to OB mid
    mitigated: bool     # True if price has already returned to the OB


def order_blocks(df: pd.DataFrame, lookback: int = 100,
                 impulse_pct: float = 1.5) -> list[OrderBlock]:
    """
    Detect bullish and bearish order blocks within `lookback` bars.

    Bullish OB: last bearish candle before a strong up-move (impulse_pct% or more).
    Bearish OB: last bullish candle before a strong down-move.

    Only un-mitigated (untested) OBs are most relevant — we flag both
    but mitigated ones carry lower weight in market_analyzer.
    """
    if len(df) < 10:
        return []

    df_slice = df.iloc[-lookback:].reset_index(drop=True)
    current  = float(df['close'].iloc[-1])
    obs      = []

    for i in range(1, len(df_slice) - 1):
        candle_open  = float(df_slice['open'].iloc[i])
        candle_close = float(df_slice['close'].iloc[i])
        next_open    = float(df_slice['open'].iloc[i + 1])
        next_close   = float(df_slice['close'].iloc[i + 1])

        # Measure impulse of the NEXT candle
        next_move_pct = abs(next_close - next_open) / next_open * 100 if next_open != 0 else 0

        if next_move_pct < impulse_pct:
            continue  # not an impulsive move — skip

        # Bullish OB: current candle is bearish, next candle is strongly bullish
        if candle_close < candle_open and next_close > next_open:
            top    = max(candle_open, candle_close)
            bottom = min(candle_open, candle_close)
            mid    = (top + bottom) / 2
            # Mitigated if price has since traded back into the OB
            subsequent_lows = df_slice['low'].iloc[i + 1:].min() if i + 1 < len(df_slice) else current
            mitigated = float(subsequent_lows) <= top

            obs.append(OrderBlock(
                ob_type='bullish',
                top=top, bottom=bottom, mid=mid,
                index=i,
                distance_pct=abs(current - mid) / current * 100,
                mitigated=mitigated,
            ))

        # Bearish OB: current candle is bullish, next candle is strongly bearish
        elif candle_close > candle_open and next_close < next_open:
            top    = max(candle_open, candle_close)
            bottom = min(candle_open, candle_close)
            mid    = (top + bottom) / 2
            subsequent_highs = df_slice['high'].iloc[i + 1:].max() if i + 1 < len(df_slice) else current
            mitigated = float(subsequent_highs) >= bottom

            obs.append(OrderBlock(
                ob_type='bearish',
                top=top, bottom=bottom, mid=mid,
                index=i,
                distance_pct=abs(current - mid) / current * 100,
                mitigated=mitigated,
            ))

    # Sort by recency (highest index = most recent) and proximity
    obs.sort(key=lambda x: (-x.index, x.distance_pct))

    # Return the 3 most relevant of each type
    bullish = [o for o in obs if o.ob_type == 'bullish'][:3]
    bearish = [o for o in obs if o.ob_type == 'bearish'][:3]
    return bullish + bearish


# ──────────────────────────────────────────────────────────────
# 2. FAIR VALUE GAPS (FVG / Imbalances)
# A 3-candle sequence where candle 1 high and candle 3 low don't
# overlap leaves a gap price is magnetically drawn to fill.
# ──────────────────────────────────────────────────────────────

@dataclass
class FairValueGap:
    fvg_type: str     # 'bullish' | 'bearish'
    top: float        # upper boundary of the gap
    bottom: float     # lower boundary
    mid: float
    distance_pct: float
    filled: bool      # True if price has already closed into the gap


def fair_value_gaps(df: pd.DataFrame, lookback: int = 50) -> list[FairValueGap]:
    """
    Bullish FVG: candle[i-1].high < candle[i+1].low — gap left above candle i-1.
    Bearish FVG: candle[i-1].low  > candle[i+1].high — gap left below candle i-1.
    """
    if len(df) < 5:
        return []

    df_slice = df.iloc[-lookback:].reset_index(drop=True)
    current  = float(df['close'].iloc[-1])
    fvgs     = []

    for i in range(1, len(df_slice) - 1):
        h_prev = float(df_slice['high'].iloc[i - 1])
        l_prev = float(df_slice['low'].iloc[i - 1])
        h_next = float(df_slice['high'].iloc[i + 1])
        l_next = float(df_slice['low'].iloc[i + 1])

        # Bullish FVG: gap is above the previous candle (price moved up fast)
        if h_prev < l_next:
            top    = l_next
            bottom = h_prev
            mid    = (top + bottom) / 2
            # Filled if current price has traded back into the gap
            filled = current <= top and current >= bottom
            fvgs.append(FairValueGap(
                fvg_type='bullish',
                top=top, bottom=bottom, mid=mid,
                distance_pct=abs(current - mid) / current * 100,
                filled=filled,
            ))

        # Bearish FVG: gap is below the previous candle (price moved down fast)
        elif l_prev > h_next:
            top    = l_prev
            bottom = h_next
            mid    = (top + bottom) / 2
            filled = current >= bottom and current <= top
            fvgs.append(FairValueGap(
                fvg_type='bearish',
                top=top, bottom=bottom, mid=mid,
                distance_pct=abs(current - mid) / current * 100,
                filled=filled,
            ))

    # Most recent and closest unfilled gaps are most relevant
    fvgs.sort(key=lambda x: (x.filled, x.distance_pct))
    bullish = [f for f in fvgs if f.fvg_type == 'bullish' and not f.filled][:2]
    bearish = [f for f in fvgs if f.fvg_type == 'bearish' and not f.filled][:2]
    return bullish + bearish


# ──────────────────────────────────────────────────────────────
# 3. BREAK OF STRUCTURE (BOS) / CHANGE OF CHARACTER (CHoCH)
# BOS  = price breaks beyond the last swing in the same direction → trend continuation
# CHoCH = price breaks against the prevailing structure → potential reversal
# ──────────────────────────────────────────────────────────────

@dataclass
class StructureBreak:
    event: str          # 'bos_bullish' | 'bos_bearish' | 'choch_bullish' | 'choch_bearish'
    broken_level: float # the swing high/low that was broken
    strength: float     # % of the break beyond the level
    bars_ago: int       # how many bars since the event


def break_of_structure(df: pd.DataFrame, swing_lookback: int = 20) -> Optional[StructureBreak]:
    """
    Identify the most recent BOS or CHoCH event within the last `swing_lookback` bars.
    Returns the single most recent event or None.

    Logic:
    - Find the last significant swing high and swing low
    - If current close > last swing high → BOS bullish (continuation if uptrend, CHoCH if downtrend)
    - If current close < last swing low  → BOS bearish (continuation if downtrend, CHoCH if uptrend)
    """
    if len(df) < swing_lookback + 5:
        return None

    window  = df.iloc[-swing_lookback - 5:]
    current = float(df['close'].iloc[-1])
    wsize   = 3  # pivot window

    # Find most recent swing high and swing low in the lookback window
    swing_highs = []
    swing_lows  = []

    for i in range(wsize, len(window) - wsize):
        h = float(window['high'].iloc[i])
        l = float(window['low'].iloc[i])

        if all(h >= float(window['high'].iloc[i - j]) for j in range(1, wsize + 1)) and \
           all(h >= float(window['high'].iloc[i + j]) for j in range(1, wsize + 1)):
            swing_highs.append((i, h))

        if all(l <= float(window['low'].iloc[i - j]) for j in range(1, wsize + 1)) and \
           all(l <= float(window['low'].iloc[i + j]) for j in range(1, wsize + 1)):
            swing_lows.append((i, l))

    if not swing_highs and not swing_lows:
        return None

    # Get most recent swing high and low
    last_sh = swing_highs[-1] if swing_highs else None
    last_sl = swing_lows[-1]  if swing_lows  else None

    # Determine prior trend from swing sequence
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        sh_rising = swing_highs[-1][1] > swing_highs[-2][1]
        sl_rising = swing_lows[-1][1]  > swing_lows[-2][1]
        prior_uptrend   = sh_rising and sl_rising
        prior_downtrend = not sh_rising and not sl_rising
    else:
        prior_uptrend   = False
        prior_downtrend = False

    total_bars = len(df)

    # Check if current candle broke a swing
    if last_sh and current > last_sh[1]:
        strength  = (current - last_sh[1]) / last_sh[1] * 100
        bars_ago  = total_bars - (total_bars - swing_lookback - 5 + last_sh[0])
        event     = 'bos_bullish' if prior_uptrend else 'choch_bullish'
        return StructureBreak(event=event, broken_level=last_sh[1],
                              strength=strength, bars_ago=max(0, bars_ago))

    if last_sl and current < last_sl[1]:
        strength  = (last_sl[1] - current) / last_sl[1] * 100
        bars_ago  = total_bars - (total_bars - swing_lookback - 5 + last_sl[0])
        event     = 'bos_bearish' if prior_downtrend else 'choch_bearish'
        return StructureBreak(event=event, broken_level=last_sl[1],
                              strength=strength, bars_ago=max(0, bars_ago))

    return None


# ──────────────────────────────────────────────────────────────
# 4. LIQUIDITY SWEEPS (stop-hunt detection)
# Institutions push price through a cluster of retail stop-losses
# (below a swing low / above a swing high) then reverse sharply.
# A sweep + immediate reversal = high-quality entry signal.
# ──────────────────────────────────────────────────────────────

@dataclass
class LiquiditySweep:
    sweep_type: str     # 'bullish_sweep' (swept lows, reversal up) | 'bearish_sweep' (swept highs, reversal down)
    swept_level: float  # the swing level that was swept
    wick_pct: float     # size of the sweep wick as % of price
    bars_ago: int       # 0 = current candle, 1 = previous candle


def liquidity_sweep(df: pd.DataFrame, lookback: int = 30) -> Optional[LiquiditySweep]:
    """
    Detect a liquidity sweep: price wicks through a swing level but closes back on the other side.

    Bullish sweep: low of a candle breaks below a prior swing low,
                   but close recovers ABOVE that swing low → stop-hunt, reversal likely up.

    Bearish sweep: high of a candle breaks above a prior swing high,
                   but close falls BACK below that swing high → stop-hunt, reversal likely down.

    Check the last 3 candles for a recent sweep.
    """
    if len(df) < lookback + 5:
        return None

    # Get recent swing levels using a smaller window
    recent_df     = df.iloc[-lookback:]
    current_price = float(df['close'].iloc[-1])
    swing_window  = 5

    recent_highs = []
    recent_lows  = []

    for i in range(swing_window, len(recent_df) - swing_window):
        h = float(recent_df['high'].iloc[i])
        l = float(recent_df['low'].iloc[i])
        if all(h >= float(recent_df['high'].iloc[i - j]) for j in range(1, swing_window + 1)) and \
           all(h >= float(recent_df['high'].iloc[i + j]) for j in range(1, swing_window + 1)):
            recent_highs.append(h)
        if all(l <= float(recent_df['low'].iloc[i - j]) for j in range(1, swing_window + 1)) and \
           all(l <= float(recent_df['low'].iloc[i + j]) for j in range(1, swing_window + 1)):
            recent_lows.append(l)

    if not recent_highs and not recent_lows:
        return None

    # Check last 3 candles for a sweep
    for offset in range(min(3, len(df))):
        idx   = -(1 + offset)
        c_low   = float(df['low'].iloc[idx])
        c_high  = float(df['high'].iloc[idx])
        c_close = float(df['close'].iloc[idx])
        c_open  = float(df['open'].iloc[idx])

        # Bullish sweep: wick broke below a swing low but candle closed above it
        if recent_lows:
            nearest_low = min(recent_lows, key=lambda x: abs(x - c_low))
            if c_low < nearest_low and c_close > nearest_low:
                wick_pct = (nearest_low - c_low) / nearest_low * 100
                return LiquiditySweep(
                    sweep_type='bullish_sweep',
                    swept_level=nearest_low,
                    wick_pct=wick_pct,
                    bars_ago=offset,
                )

        # Bearish sweep: wick broke above a swing high but candle closed below it
        if recent_highs:
            nearest_high = min(recent_highs, key=lambda x: abs(x - c_high))
            if c_high > nearest_high and c_close < nearest_high:
                wick_pct = (c_high - nearest_high) / nearest_high * 100
                return LiquiditySweep(
                    sweep_type='bearish_sweep',
                    swept_level=nearest_high,
                    wick_pct=wick_pct,
                    bars_ago=offset,
                )

    return None