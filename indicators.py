# indicators.py — all technical indicator functions (core TA + SMC + v3 Advanced TA)
# ─────────────────────────────────────────────────────────────
# Core TA  : ema_cross, rsi, macd, bollinger_bands, atr
# SMC      : order_block, fair_value_gap, bos_choch, liquidity_sweep
# v3 Adv TA: avwap_pinch, alligator, awesome_oscillator, fractals,
#             volume_relative, trap_detector, elliott_divergence
# ─────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════════════
# CORE TA — EMA CROSS
# ══════════════════════════════════════════════════════════════

@dataclass
class EMACrossResult:
    fast_val:       float
    slow_val:       float
    fast_above_slow: bool
    cross_up:       bool   # fast crossed above slow on this bar
    cross_down:     bool   # fast crossed below slow on this bar


def ema_cross(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> Optional[EMACrossResult]:
    if len(df) < slow + 2:
        return None
    close = df['close']
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()

    curr_fast = float(fast_ema.iloc[-1])
    curr_slow = float(slow_ema.iloc[-1])
    prev_fast = float(fast_ema.iloc[-2])
    prev_slow = float(slow_ema.iloc[-2])

    cross_up   = prev_fast <= prev_slow and curr_fast > curr_slow
    cross_down = prev_fast >= prev_slow and curr_fast < curr_slow

    return EMACrossResult(
        fast_val=curr_fast,
        slow_val=curr_slow,
        fast_above_slow=curr_fast > curr_slow,
        cross_up=cross_up,
        cross_down=cross_down,
    )


# ══════════════════════════════════════════════════════════════
# CORE TA — RSI
# ══════════════════════════════════════════════════════════════

@dataclass
class RSIResult:
    value: float
    prev_value: float


def rsi(df: pd.DataFrame, period: int = 14) -> Optional[RSIResult]:
    if len(df) < period + 2:
        return None
    close = df['close']
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - 100 / (1 + rs)
    return RSIResult(
        value=float(rsi_series.iloc[-1]),
        prev_value=float(rsi_series.iloc[-2]),
    )


# ══════════════════════════════════════════════════════════════
# CORE TA — MACD
# ══════════════════════════════════════════════════════════════

@dataclass
class MACDResult:
    macd_line:  float
    signal_line: float
    histogram:  float
    cross_up:   bool
    cross_down: bool


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[MACDResult]:
    if len(df) < slow + signal + 2:
        return None
    close = df['close']
    ema_fast   = close.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line

    curr_macd = float(macd_line.iloc[-1])
    curr_sig  = float(signal_line.iloc[-1])
    prev_macd = float(macd_line.iloc[-2])
    prev_sig  = float(signal_line.iloc[-2])

    cross_up   = prev_macd <= prev_sig and curr_macd > curr_sig
    cross_down = prev_macd >= prev_sig and curr_macd < curr_sig

    return MACDResult(
        macd_line=curr_macd,
        signal_line=curr_sig,
        histogram=float(histogram.iloc[-1]),
        cross_up=cross_up,
        cross_down=cross_down,
    )


# ══════════════════════════════════════════════════════════════
# CORE TA — BOLLINGER BANDS
# ══════════════════════════════════════════════════════════════

@dataclass
class BollingerResult:
    upper:  float
    middle: float
    lower:  float
    width:  float   # (upper - lower) / middle * 100


def bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> Optional[BollingerResult]:
    if len(df) < period + 2:
        return None
    close  = df['close']
    middle = close.rolling(period).mean()
    stddev = close.rolling(period).std()
    upper  = middle + std * stddev
    lower  = middle - std * stddev

    mid_val = float(middle.iloc[-1])
    up_val  = float(upper.iloc[-1])
    lo_val  = float(lower.iloc[-1])

    width = (up_val - lo_val) / mid_val * 100 if mid_val != 0 else 0.0

    return BollingerResult(upper=up_val, middle=mid_val, lower=lo_val, width=width)


# ══════════════════════════════════════════════════════════════
# CORE TA — ATR
# ══════════════════════════════════════════════════════════════

@dataclass
class ATRResult:
    value: float


def atr(df: pd.DataFrame, period: int = 14) -> Optional[ATRResult]:
    if len(df) < period + 2:
        return None
    high  = df['high']
    low   = df['low']
    prev  = df['close'].shift(1)
    tr    = pd.concat([(high - low),
                       (high - prev).abs(),
                       (low  - prev).abs()], axis=1).max(axis=1)
    atr_val = float(tr.ewm(com=period - 1, adjust=False).mean().iloc[-1])
    return ATRResult(value=atr_val)


# ══════════════════════════════════════════════════════════════
# SMC — ORDER BLOCK
# ══════════════════════════════════════════════════════════════

@dataclass
class OrderBlockResult:
    bullish_ob:   bool
    bearish_ob:   bool
    bullish_level: float
    bearish_level: float
    near_bullish: bool   # price within proximity_pct of the OB
    near_bearish: bool


def order_block(df: pd.DataFrame,
                impulse_pct: float = 1.2,
                proximity_pct: float = 2.0) -> Optional[OrderBlockResult]:
    """
    A bullish OB is the last bearish candle before a strong bullish impulse.
    A bearish OB is the last bullish candle before a strong bearish impulse.
    """
    if len(df) < 10:
        return None

    current = float(df['close'].iloc[-1])
    bull_level = 0.0
    bear_level = 0.0
    bull_ob    = False
    bear_ob    = False

    for i in range(2, min(len(df) - 1, 30)):
        idx = -(i + 1)  # look-back bar
        nxt = -i        # next bar (the impulse bar)

        o_curr = float(df['open'].iloc[idx])
        c_curr = float(df['close'].iloc[idx])
        o_nxt  = float(df['open'].iloc[nxt])
        c_nxt  = float(df['close'].iloc[nxt])

        if o_curr == 0 or o_nxt == 0:
            continue

        move_pct = abs(c_nxt - o_nxt) / o_nxt * 100

        # Bullish OB: bearish candle followed by a strong bullish impulse
        if c_curr < o_curr and c_nxt > o_nxt and move_pct >= impulse_pct:
            bull_level = float(df['close'].iloc[idx])
            bull_ob    = True
            break

    for i in range(2, min(len(df) - 1, 30)):
        idx = -(i + 1)
        nxt = -i

        o_curr = float(df['open'].iloc[idx])
        c_curr = float(df['close'].iloc[idx])
        o_nxt  = float(df['open'].iloc[nxt])
        c_nxt  = float(df['close'].iloc[nxt])

        if o_curr == 0 or o_nxt == 0:
            continue

        move_pct = abs(c_nxt - o_nxt) / o_nxt * 100

        # Bearish OB: bullish candle followed by a strong bearish impulse
        if c_curr > o_curr and c_nxt < o_nxt and move_pct >= impulse_pct:
            bear_level = float(df['close'].iloc[idx])
            bear_ob    = True
            break

    near_bull = bull_ob and abs(current - bull_level) / bull_level * 100 <= proximity_pct \
                if bull_level != 0 else False
    near_bear = bear_ob and abs(current - bear_level) / bear_level * 100 <= proximity_pct \
                if bear_level != 0 else False

    return OrderBlockResult(
        bullish_ob=bull_ob,
        bearish_ob=bear_ob,
        bullish_level=bull_level,
        bearish_level=bear_level,
        near_bullish=near_bull,
        near_bearish=near_bear,
    )


# ══════════════════════════════════════════════════════════════
# SMC — FAIR VALUE GAP
# ══════════════════════════════════════════════════════════════

@dataclass
class FVGResult:
    bullish_fvg:      bool
    bearish_fvg:      bool
    bull_high:        float
    bull_low:         float
    bear_high:        float
    bear_low:         float
    price_in_bullish: bool
    price_in_bearish: bool


def fair_value_gap(df: pd.DataFrame, proximity_pct: float = 1.5) -> Optional[FVGResult]:
    """
    FVG = three-candle pattern where candle 1's wick and candle 3's wick
    don't overlap, leaving an imbalance gap.

    Bullish FVG: candle 3 low > candle 1 high  (gap to the upside)
    Bearish FVG: candle 3 high < candle 1 low  (gap to the downside)
    """
    if len(df) < 5:
        return None

    current     = float(df['close'].iloc[-1])
    bull_fvg    = False
    bear_fvg    = False
    bull_hi     = 0.0
    bull_lo     = 0.0
    bear_hi     = 0.0
    bear_lo     = 0.0

    for i in range(2, min(len(df) - 2, 30)):
        c1_hi = float(df['high'].iloc[-(i + 2)])
        c1_lo = float(df['low'].iloc[-(i + 2)])
        c3_hi = float(df['high'].iloc[-i])
        c3_lo = float(df['low'].iloc[-i])

        # Bullish FVG
        if c3_lo > c1_hi:
            bull_lo = c1_hi
            bull_hi = c3_lo
            bull_fvg = True
            break

    for i in range(2, min(len(df) - 2, 30)):
        c1_hi = float(df['high'].iloc[-(i + 2)])
        c1_lo = float(df['low'].iloc[-(i + 2)])
        c3_hi = float(df['high'].iloc[-i])
        c3_lo = float(df['low'].iloc[-i])

        # Bearish FVG
        if c3_hi < c1_lo:
            bear_hi = c1_lo
            bear_lo = c3_hi
            bear_fvg = True
            break

    in_bull = bull_fvg and bull_lo <= current <= bull_hi
    in_bear = bear_fvg and bear_lo <= current <= bear_hi

    return FVGResult(
        bullish_fvg=bull_fvg,
        bearish_fvg=bear_fvg,
        bull_high=bull_hi,
        bull_low=bull_lo,
        bear_high=bear_hi,
        bear_low=bear_lo,
        price_in_bullish=in_bull,
        price_in_bearish=in_bear,
    )


# ══════════════════════════════════════════════════════════════
# SMC — BOS / CHoCH
# ══════════════════════════════════════════════════════════════

@dataclass
class BOSCHoCHResult:
    bullish_bos:   bool   # Break of Structure upward
    bearish_bos:   bool   # Break of Structure downward
    bullish_choch: bool   # Change of Character (reversal) upward
    bearish_choch: bool   # Change of Character (reversal) downward


def bos_choch(df: pd.DataFrame, max_bars: int = 3) -> Optional[BOSCHoCHResult]:
    """
    BOS  : price breaks the most recent swing high/low in the SAME direction as trend.
    CHoCH: price breaks in the OPPOSITE direction — potential trend reversal.

    We use a simplified 3-bar swing high/low for speed.
    """
    if len(df) < 10:
        return None

    # Find last swing high and low (3-bar pivot) in the last 20 bars
    window = df.iloc[-20:].reset_index(drop=True)
    n = len(window)
    current = float(df['close'].iloc[-1])

    swing_highs = []
    swing_lows  = []

    for i in range(1, n - 1):
        h = float(window['high'].iloc[i])
        l = float(window['low'].iloc[i])
        if h > float(window['high'].iloc[i - 1]) and h > float(window['high'].iloc[i + 1]):
            swing_highs.append(h)
        if l < float(window['low'].iloc[i - 1]) and l < float(window['low'].iloc[i + 1]):
            swing_lows.append(l)

    bull_bos = bear_bos = bull_choch = bear_choch = False

    # Determine recent trend from last N closes
    recent_closes = df['close'].iloc[-6:]
    rising = float(recent_closes.iloc[-1]) > float(recent_closes.iloc[0])

    if swing_highs:
        last_high = swing_highs[-1]
        if current > last_high * 1.001:           # broke above swing high
            if rising:
                bull_bos   = True                 # BOS — same direction as trend
            else:
                bull_choch = True                 # CHoCH — reversal upward

    if swing_lows:
        last_low = swing_lows[-1]
        if current < last_low * 0.999:            # broke below swing low
            if not rising:
                bear_bos   = True                 # BOS — same direction
            else:
                bear_choch = True                 # CHoCH — reversal downward

    return BOSCHoCHResult(
        bullish_bos=bull_bos,
        bearish_bos=bear_bos,
        bullish_choch=bull_choch,
        bearish_choch=bear_choch,
    )


# ══════════════════════════════════════════════════════════════
# SMC — LIQUIDITY SWEEP
# ══════════════════════════════════════════════════════════════

@dataclass
class LiquiditySweepResult:
    bull_sweep:   bool    # swept a low then reversed up
    bear_sweep:   bool    # swept a high then reversed down
    swept_level:  float


def liquidity_sweep(df: pd.DataFrame, max_bars: int = 2) -> Optional[LiquiditySweepResult]:
    """
    A liquidity sweep (stop hunt) occurs when price briefly breaks a key
    swing high/low (triggering stops) then reverses sharply.

    Bullish sweep: wick below a swing low but closed above it → long setup.
    Bearish sweep: wick above a swing high but closed below it → short setup.
    """
    if len(df) < 10:
        return None

    window = df.iloc[-20:].reset_index(drop=True)
    n = len(window)

    swing_highs = []
    swing_lows  = []

    for i in range(1, n - max_bars - 1):
        h = float(window['high'].iloc[i])
        l = float(window['low'].iloc[i])
        if h > float(window['high'].iloc[i - 1]) and h > float(window['high'].iloc[i + 1]):
            swing_highs.append((i, h))
        if l < float(window['low'].iloc[i - 1]) and l < float(window['low'].iloc[i + 1]):
            swing_lows.append((i, l))

    bull_sweep  = False
    bear_sweep  = False
    swept_level = 0.0

    # Check last max_bars candles for sweeps
    for offset in range(1, max_bars + 1):
        bar_low   = float(df['low'].iloc[-offset])
        bar_high  = float(df['high'].iloc[-offset])
        bar_close = float(df['close'].iloc[-offset])

        if swing_lows:
            level = swing_lows[-1][1]
            # Wick below swing low but closed above it
            if bar_low < level * 0.999 and bar_close > level:
                bull_sweep  = True
                swept_level = level
                break

        if swing_highs:
            level = swing_highs[-1][1]
            # Wick above swing high but closed below it
            if bar_high > level * 1.001 and bar_close < level:
                bear_sweep  = True
                swept_level = level
                break

    return LiquiditySweepResult(
        bull_sweep=bull_sweep,
        bear_sweep=bear_sweep,
        swept_level=swept_level,
    )


# ══════════════════════════════════════════════════════════════
# BRIAN SHANNON — ANCHORED VWAP
# ══════════════════════════════════════════════════════════════

@dataclass
class AVWAPResult:
    anchor_bar:  int
    value:       float
    slope_pct:   float
    price_above: bool


def anchored_vwap(df: pd.DataFrame, anchor_bar: int) -> Optional[AVWAPResult]:
    if anchor_bar < 0:
        anchor_bar = max(0, len(df) + anchor_bar)
    if anchor_bar >= len(df) - 1:
        return None

    segment = df.iloc[anchor_bar:].copy()
    if len(segment) < 2:
        return None

    typical_price = (segment['high'] + segment['low'] + segment['close']) / 3
    tp_vol  = typical_price * segment['volume']
    cumvol  = segment['volume'].cumsum()
    if float(cumvol.iloc[-1]) == 0:
        return None

    avwap_series = tp_vol.cumsum() / cumvol
    value   = float(avwap_series.iloc[-1])
    current = float(df['close'].iloc[-1])

    if len(avwap_series) >= 5:
        prev_val  = float(avwap_series.iloc[-5])
        slope_pct = (value - prev_val) / prev_val * 100 if prev_val != 0 else 0.0
    else:
        slope_pct = 0.0

    return AVWAPResult(
        anchor_bar=anchor_bar,
        value=value,
        slope_pct=slope_pct,
        price_above=current > value,
    )


@dataclass
class AVWAPPinchResult:
    pinch_active: bool
    high_avwap:   float
    low_avwap:    float
    width_pct:    float
    breakout:     str    # 'bullish' | 'bearish' | 'none'
    slope_filter: bool


def avwap_pinch(df: pd.DataFrame, lookback: int = 60) -> Optional[AVWAPPinchResult]:
    if len(df) < lookback + 10:
        return None

    window  = df.iloc[-lookback:]
    current = float(df['close'].iloc[-1])

    swing_high_idx = None
    swing_low_idx  = None
    wsize = 3

    for i in range(wsize, len(window) - wsize):
        h = float(window['high'].iloc[i])
        l = float(window['low'].iloc[i])

        if all(h >= float(window['high'].iloc[i - j]) for j in range(1, wsize + 1)) and \
           all(h >= float(window['high'].iloc[i + j]) for j in range(1, wsize + 1)):
            swing_high_idx = len(df) - lookback + i

        if all(l <= float(window['low'].iloc[i - j]) for j in range(1, wsize + 1)) and \
           all(l <= float(window['low'].iloc[i + j]) for j in range(1, wsize + 1)):
            swing_low_idx = len(df) - lookback + i

    if swing_high_idx is None or swing_low_idx is None:
        return None

    avwap_from_high = anchored_vwap(df, swing_high_idx)
    avwap_from_low  = anchored_vwap(df, swing_low_idx)

    if avwap_from_high is None or avwap_from_low is None:
        return None

    top_avwap = max(avwap_from_high.value, avwap_from_low.value)
    bot_avwap = min(avwap_from_high.value, avwap_from_low.value)

    width_pct    = (top_avwap - bot_avwap) / current * 100 if current != 0 else 0.0
    between      = bot_avwap <= current <= top_avwap
    pinch_active = between and width_pct < 3.0

    if current > top_avwap * 1.002:
        breakout = 'bullish'
    elif current < bot_avwap * 0.998:
        breakout = 'bearish'
    else:
        breakout = 'none'

    slope_filter = avwap_from_low.slope_pct > 0

    return AVWAPPinchResult(
        pinch_active=pinch_active,
        high_avwap=top_avwap,
        low_avwap=bot_avwap,
        width_pct=width_pct,
        breakout=breakout,
        slope_filter=slope_filter,
    )


# ══════════════════════════════════════════════════════════════
# BILL WILLIAMS — THE ALLIGATOR
# ══════════════════════════════════════════════════════════════

@dataclass
class AlligatorResult:
    jaw:              float
    teeth:            float
    lips:             float
    phase:            str    # 'sleeping'|'awakening'|'eating_bull'|'eating_bear'|'sated'
    lips_above_jaw:   bool
    lips_above_teeth: bool


def _smma(series: pd.Series, period: int) -> pd.Series:
    result = series.copy().astype(float)
    result.iloc[:period] = np.nan
    seed = float(series.iloc[:period].mean())
    result.iloc[period - 1] = seed
    for i in range(period, len(series)):
        result.iloc[i] = (result.iloc[i - 1] * (period - 1) + series.iloc[i]) / period
    return result


def alligator(df: pd.DataFrame) -> Optional[AlligatorResult]:
    if len(df) < 30:
        return None

    median    = (df['high'] + df['low']) / 2
    jaw_raw   = _smma(median, 13)
    teeth_raw = _smma(median, 8)
    lips_raw  = _smma(median, 5)

    if len(jaw_raw.dropna()) < 9:
        return None

    jaw_val   = float(jaw_raw.iloc[-8])   if len(jaw_raw)   >= 8 else float(jaw_raw.iloc[-1])
    teeth_val = float(teeth_raw.iloc[-5]) if len(teeth_raw) >= 5 else float(teeth_raw.iloc[-1])
    lips_val  = float(lips_raw.iloc[-3])  if len(lips_raw)  >= 3 else float(lips_raw.iloc[-1])

    vals       = sorted([jaw_val, teeth_val, lips_val])
    spread_pct = (vals[2] - vals[0]) / vals[1] * 100 if vals[1] != 0 else 0

    lips_above_jaw   = lips_val > jaw_val
    lips_above_teeth = lips_val > teeth_val
    in_order_bull    = lips_val > teeth_val > jaw_val
    in_order_bear    = lips_val < teeth_val < jaw_val

    if spread_pct < 0.3:
        phase = 'sleeping'
    elif in_order_bull and spread_pct > 0.6:
        if len(lips_raw) >= 4:
            prev_lips = float(lips_raw.iloc[-4])
            phase = 'eating_bull' if lips_val > prev_lips else 'sated'
        else:
            phase = 'eating_bull'
    elif in_order_bear and spread_pct > 0.6:
        if len(lips_raw) >= 4:
            prev_lips = float(lips_raw.iloc[-4])
            phase = 'eating_bear' if lips_val < prev_lips else 'sated'
        else:
            phase = 'eating_bear'
    elif spread_pct < 0.6:
        phase = 'awakening'
    else:
        phase = 'sated'

    return AlligatorResult(
        jaw=jaw_val,
        teeth=teeth_val,
        lips=lips_val,
        phase=phase,
        lips_above_jaw=lips_above_jaw,
        lips_above_teeth=lips_above_teeth,
    )


# ══════════════════════════════════════════════════════════════
# BILL WILLIAMS — AWESOME OSCILLATOR (AO)
# ══════════════════════════════════════════════════════════════

@dataclass
class AOResult:
    value:      float
    prev_value: float
    signal:     str          # 'bullish_cross'|'bearish_cross'|'bullish'|'bearish'|'neutral'
    twin_peaks: Optional[str]  # 'bullish_twin'|'bearish_twin'|None


def awesome_oscillator(df: pd.DataFrame, fast: int = 5, slow: int = 34) -> Optional[AOResult]:
    if len(df) < slow + 5:
        return None

    midprice = (df['high'] + df['low']) / 2
    ao = midprice.rolling(fast).mean() - midprice.rolling(slow).mean()

    curr  = float(ao.iloc[-1])
    prev  = float(ao.iloc[-2])

    if prev <= 0 and curr > 0:
        signal = 'bullish_cross'
    elif prev >= 0 and curr < 0:
        signal = 'bearish_cross'
    elif curr > 0 and curr > prev:
        signal = 'bullish'
    elif curr < 0 and curr < prev:
        signal = 'bearish'
    else:
        signal = 'neutral'

    twin_peaks = None
    if len(ao) >= 6:
        w = ao.iloc[-6:].values
        lows_below = [(i, v) for i, v in enumerate(w) if v < 0]
        if len(lows_below) >= 2:
            i1, v1 = lows_below[-2]
            i2, v2 = lows_below[-1]
            if v2 > v1 and i2 > i1:
                twin_peaks = 'bullish_twin'
        highs_above = [(i, v) for i, v in enumerate(w) if v > 0]
        if len(highs_above) >= 2:
            i1, v1 = highs_above[-2]
            i2, v2 = highs_above[-1]
            if v2 < v1 and i2 > i1:
                twin_peaks = 'bearish_twin'

    return AOResult(value=curr, prev_value=prev, signal=signal, twin_peaks=twin_peaks)


# ══════════════════════════════════════════════════════════════
# BILL WILLIAMS — FRACTALS
# ══════════════════════════════════════════════════════════════

@dataclass
class FractalResult:
    last_up_fractal:   Optional[float]
    last_down_fractal: Optional[float]
    bars_since_up:     int
    bars_since_down:   int
    breakout_above_up: bool
    breakout_below_dn: bool


def fractals(df: pd.DataFrame, lookback: int = 60) -> Optional[FractalResult]:
    if len(df) < 10:
        return None

    window  = df.iloc[-lookback:].reset_index(drop=True)
    n       = len(window)
    current = float(df['close'].iloc[-1])

    last_up_fractal   = None
    last_up_idx       = None
    last_down_fractal = None
    last_down_idx     = None

    for i in range(2, n - 2):
        h = float(window['high'].iloc[i])
        l = float(window['low'].iloc[i])

        if (h > float(window['high'].iloc[i - 1]) and
            h > float(window['high'].iloc[i - 2]) and
            h > float(window['high'].iloc[i + 1]) and
            h > float(window['high'].iloc[i + 2])):
            last_up_fractal = h
            last_up_idx     = i

        if (l < float(window['low'].iloc[i - 1]) and
            l < float(window['low'].iloc[i - 2]) and
            l < float(window['low'].iloc[i + 1]) and
            l < float(window['low'].iloc[i + 2])):
            last_down_fractal = l
            last_down_idx     = i

    bars_since_up   = (n - 1 - last_up_idx)   if last_up_idx   is not None else 999
    bars_since_down = (n - 1 - last_down_idx)  if last_down_idx is not None else 999

    breakout_above_up = last_up_fractal is not None and current > last_up_fractal * 1.001
    breakout_below_dn = last_down_fractal is not None and current < last_down_fractal * 0.999

    return FractalResult(
        last_up_fractal=last_up_fractal,
        last_down_fractal=last_down_fractal,
        bars_since_up=bars_since_up,
        bars_since_down=bars_since_down,
        breakout_above_up=breakout_above_up,
        breakout_below_dn=breakout_below_dn,
    )


# ══════════════════════════════════════════════════════════════
# MIKE BELLAFIORE — RELATIVE VOLUME FILTER
# ══════════════════════════════════════════════════════════════

@dataclass
class RelativeVolumeResult:
    rvol:        float
    is_in_play:  bool
    average_vol: float
    current_vol: float


def volume_relative(df: pd.DataFrame,
                    lookback: int = 20,
                    threshold: float = 1.5) -> Optional[RelativeVolumeResult]:
    if 'volume' not in df.columns or len(df) < lookback + 1:
        return None

    vol     = df['volume']
    avg     = float(vol.iloc[-(lookback + 1):-1].mean())
    current = float(vol.iloc[-1])

    if avg == 0:
        return None

    rvol = current / avg
    return RelativeVolumeResult(
        rvol=rvol,
        is_in_play=rvol >= threshold,
        average_vol=avg,
        current_vol=current,
    )


# ══════════════════════════════════════════════════════════════
# JACK SCHWAGER — BREAKOUT TRAP DETECTOR
# ══════════════════════════════════════════════════════════════

@dataclass
class TrapResult:
    trap_type:    Optional[str]   # 'bull_trap' | 'bear_trap' | None
    trap_level:   float
    reversal_pct: float
    bars_ago:     int
    signal:       str             # 'sell' | 'buy' | 'none'


def trap_detector(df: pd.DataFrame,
                  lookback: int = 30,
                  min_breakout_pct: float = 0.3,
                  max_trap_bars: int = 5) -> Optional[TrapResult]:
    if len(df) < lookback + 5:
        return None

    window = df.iloc[-lookback:].reset_index(drop=True)
    n      = len(window)
    current_close = float(df['close'].iloc[-1])

    swing_highs = []
    swing_lows  = []
    wsize = 3

    for i in range(wsize, n - wsize):
        h = float(window['high'].iloc[i])
        l = float(window['low'].iloc[i])

        if all(h >= float(window['high'].iloc[i - j]) for j in range(1, wsize + 1)) and \
           all(h >= float(window['high'].iloc[i + j]) for j in range(1, wsize + 1)):
            swing_highs.append((i, h))

        if all(l <= float(window['low'].iloc[i - j]) for j in range(1, wsize + 1)) and \
           all(l <= float(window['low'].iloc[i + j]) for j in range(1, wsize + 1)):
            swing_lows.append((i, l))

    for offset in range(1, max_trap_bars + 1):
        idx = -(1 + offset)
        if abs(idx) > len(df):
            break

        c_high  = float(df['high'].iloc[idx])
        c_low   = float(df['low'].iloc[idx])
        c_close = float(df['close'].iloc[idx])

        if swing_highs:
            level = swing_highs[-1][1]
            if c_high > level * (1 + min_breakout_pct / 100) and c_close < level:
                reversal_pct = (c_high - current_close) / c_high * 100
                return TrapResult(
                    trap_type='bull_trap',
                    trap_level=level,
                    reversal_pct=reversal_pct,
                    bars_ago=offset,
                    signal='sell',
                )

        if swing_lows:
            level = swing_lows[-1][1]
            if c_low < level * (1 - min_breakout_pct / 100) and c_close > level:
                reversal_pct = (current_close - c_low) / c_low * 100
                return TrapResult(
                    trap_type='bear_trap',
                    trap_level=level,
                    reversal_pct=reversal_pct,
                    bars_ago=offset,
                    signal='buy',
                )

    return TrapResult(trap_type=None, trap_level=0.0, reversal_pct=0.0, bars_ago=0, signal='none')


# ══════════════════════════════════════════════════════════════
# ELLIOTT WAVE — DIVERGENCE DETECTOR
# ══════════════════════════════════════════════════════════════

@dataclass
class ElliottDivergenceResult:
    wave_context:   str    # 'wave3_likely'|'wave5_warning'|'wave_a_likely'|'none'
    rsi_divergence: bool
    price_high:     float
    rsi_at_high:    float


def elliott_divergence(df: pd.DataFrame,
                       rsi_period: int = 14,
                       lookback: int = 50) -> Optional[ElliottDivergenceResult]:
    if len(df) < lookback + rsi_period + 5:
        return None

    close    = df['close']
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=rsi_period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=rsi_period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - 100 / (1 + rs)

    window_price = df['high'].iloc[-lookback:].reset_index(drop=True)
    window_rsi   = rsi_series.iloc[-lookback:].reset_index(drop=True)
    n     = len(window_price)
    wsize = 4

    price_highs = []
    for i in range(wsize, n - wsize):
        h = float(window_price.iloc[i])
        if all(h >= float(window_price.iloc[i - j]) for j in range(1, wsize + 1)) and \
           all(h >= float(window_price.iloc[i + j]) for j in range(1, wsize + 1)):
            price_highs.append((i, h, float(window_rsi.iloc[i])))

    if len(price_highs) < 2:
        # Try lows for Wave A
        price_lows = []
        window_low = df['low'].iloc[-lookback:].reset_index(drop=True)
        for i in range(wsize, n - wsize):
            l = float(window_low.iloc[i])
            if all(l <= float(window_low.iloc[i - j]) for j in range(1, wsize + 1)) and \
               all(l <= float(window_low.iloc[i + j]) for j in range(1, wsize + 1)):
                price_lows.append((i, l, float(window_rsi.iloc[i])))

        if len(price_lows) >= 2:
            _, prev_l, prev_rsi = price_lows[-2]
            _, last_l, last_rsi = price_lows[-1]
            if last_l < prev_l and last_rsi > prev_rsi:
                return ElliottDivergenceResult(
                    wave_context='wave_a_likely',
                    rsi_divergence=True,
                    price_high=last_l,
                    rsi_at_high=last_rsi,
                )

        return ElliottDivergenceResult(
            wave_context='none',
            rsi_divergence=False,
            price_high=float(df['high'].iloc[-1]),
            rsi_at_high=float(rsi_series.iloc[-1]),
        )

    prev_swing = price_highs[-2]
    last_swing = price_highs[-1]

    price_new_high = last_swing[1] > prev_swing[1]
    rsi_new_high   = last_swing[2] > prev_swing[2]

    if price_new_high and rsi_new_high:
        wave_context = 'wave3_likely'
        divergence   = False
    elif price_new_high and not rsi_new_high:
        wave_context = 'wave5_warning'
        divergence   = True
    else:
        wave_context = 'none'
        divergence   = False

    return ElliottDivergenceResult(
        wave_context=wave_context,
        rsi_divergence=divergence,
        price_high=last_swing[1],
        rsi_at_high=last_swing[2],
    )