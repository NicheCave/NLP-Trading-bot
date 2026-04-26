# market_analyzer.py — multi-timeframe TA + SMC + Advanced TA analysis (v3)
# ─────────────────────────────────────────────────────────────
# Produces an AnalysisResult by running all indicator votes across
# 1m / 1h / 1d timeframes and aggregating into bull/bear/net scores.
#
# v3 additions (integrated from market_analyzer_additions.py):
#   - AnalysisResult extended with alligator_sleeping, wave5_warning,
#     trap_active, trap_signal flags
#   - INDICATOR_WEIGHTS extended with 7 new v3 indicators
#   - analyze() extended with vote blocks 16-22 (AVWAP, Alligator, AO,
#     Fractals, RVOL, Trap Detector, Elliott Divergence)
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

import config
from market_data import fetch_multi_timeframe
from indicators import (
    # ── core TA ──────────────────────────────────────────────
    ema_cross,
    rsi,
    macd,
    bollinger_bands,
    atr,
    # ── SMC ──────────────────────────────────────────────────
    order_block,
    fair_value_gap,
    bos_choch,
    liquidity_sweep,
    # ── v3 Advanced TA ───────────────────────────────────────
    avwap_pinch,
    alligator,
    awesome_oscillator,
    fractals,
    volume_relative,
    trap_detector,
    elliott_divergence,
    # ── result types (v3) ────────────────────────────────────
    AVWAPPinchResult,
    AlligatorResult,
    AOResult,
    FractalResult,
    RelativeVolumeResult,
    TrapResult,
    ElliottDivergenceResult,
)


# ══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════

@dataclass
class IndicatorVote:
    """A single indicator's directional vote."""
    name:   str
    vote:   int    # +1 = bull, -1 = bear, 0 = neutral
    weight: float
    reason: str


@dataclass
class AnalysisResult:
    """Aggregated multi-indicator analysis for one ticker."""
    ticker:      str
    trend:       str          # 'uptrend' | 'downtrend' | 'sideways'
    bull_score:  float        # 0.0–1.0
    bear_score:  float        # 0.0–1.0
    net_score:   float        # -1.0 to +1.0  (bull - bear, weighted)
    votes:       list[IndicatorVote] = field(default_factory=list)
    summary:     str = ''

    # ── v3 flags (checked by signal_generator.py) ────────────
    alligator_sleeping: bool = False   # True → market choppy → force HOLD
    wave5_warning:      bool = False   # True → Elliott W5 divergence → reduce size
    trap_active:        bool = False   # True → trap pattern present → flip signal
    trap_signal:        str  = 'none'  # 'buy' | 'sell' | 'none'


# ══════════════════════════════════════════════════════════════
# INDICATOR WEIGHTS
# ══════════════════════════════════════════════════════════════

INDICATOR_WEIGHTS: dict[str, float] = {
    # ── Core TA ──────────────────────────────────────────────
    'ema_cross':           2.0,
    'rsi':                 1.5,
    'macd':                1.5,
    'bollinger':           1.0,
    'atr_trend':           0.5,

    # ── SMC ──────────────────────────────────────────────────
    'smc_order_block':     2.5,
    'smc_fvg':             2.0,
    'smc_bos_choch':       2.5,
    'smc_liquidity':       2.0,

    # ── v3: Brian Shannon — Anchored VWAP ────────────────────
    'avwap_pinch':         2.5,

    # ── v3: Bill Williams — Natural Order System ─────────────
    'williams_alligator':  2.0,
    'williams_ao':         1.5,
    'williams_fractals':   1.5,

    # ── v3: Bellafiore — volume filter ───────────────────────
    'rvol_filter':         1.5,

    # ── v3: Schwager — trap detector ─────────────────────────
    'trap_detector':       3.0,   # highest weight — fastest moves

    # ── v3: Elliott Wave — wave personality ──────────────────
    'elliott_divergence':  2.0,
}


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _aggregate(votes: list[IndicatorVote]) -> tuple[float, float, float]:
    """
    Aggregate votes into (bull_score, bear_score, net_score).

    bull_score = weighted bull votes / total weight (0-1)
    bear_score = weighted bear votes / total weight (0-1)
    net_score  = (bull_weighted - bear_weighted) / total_weight  (-1 to +1)
    """
    total_weight = sum(v.weight for v in votes)
    if total_weight == 0:
        return 0.0, 0.0, 0.0

    bull_w = sum(v.weight for v in votes if v.vote == +1)
    bear_w = sum(v.weight for v in votes if v.vote == -1)

    bull_score = bull_w / total_weight
    bear_score = bear_w / total_weight
    net_score  = (bull_w - bear_w) / total_weight

    return bull_score, bear_score, net_score


def _trend_from_ema(df_1h) -> str:
    """Determine broad trend from 1H EMA relationship."""
    if df_1h.empty or len(df_1h) < config.EMA_SLOW + 5:
        return 'sideways'
    try:
        import pandas as pd
        fast = df_1h['close'].ewm(span=config.EMA_FAST, adjust=False).mean()
        slow = df_1h['close'].ewm(span=config.EMA_SLOW, adjust=False).mean()
        if float(fast.iloc[-1]) > float(slow.iloc[-1]) * 1.002:
            return 'uptrend'
        if float(fast.iloc[-1]) < float(slow.iloc[-1]) * 0.998:
            return 'downtrend'
        return 'sideways'
    except Exception:
        return 'sideways'


# ══════════════════════════════════════════════════════════════
# MAIN ANALYSIS FUNCTION
# ══════════════════════════════════════════════════════════════

def analyze(ticker: str) -> Optional[AnalysisResult]:
    """
    Run full multi-timeframe analysis for a ticker and return AnalysisResult.

    Returns None if data is insufficient for a meaningful signal.
    """
    bars = fetch_multi_timeframe(ticker)
    df_1m = bars.get('1m')
    df_1h = bars.get('1h')
    df_1d = bars.get('1d')

    # Need at least 1H data to do anything meaningful
    if df_1h is None or df_1h.empty or len(df_1h) < 30:
        logger.warning(f"Insufficient 1H data for {ticker} — skipping analysis")
        return None

    votes: list[IndicatorVote] = []
    trend = _trend_from_ema(df_1h)

    # ══════════════════════════════════════════════════════════
    # CORE TA LAYER (votes 1–5)
    # ══════════════════════════════════════════════════════════

    # ── 1. EMA Cross — 1H ────────────────────────────────────
    try:
        ema = ema_cross(df_1h, fast=config.EMA_FAST, slow=config.EMA_SLOW)
        if ema is not None:
            if ema.cross_up:
                votes.append(IndicatorVote('ema_cross', +1, INDICATOR_WEIGHTS['ema_cross'],
                    f'EMA{config.EMA_FAST} crossed above EMA{config.EMA_SLOW} (1H)'))
            elif ema.cross_down:
                votes.append(IndicatorVote('ema_cross', -1, INDICATOR_WEIGHTS['ema_cross'],
                    f'EMA{config.EMA_FAST} crossed below EMA{config.EMA_SLOW} (1H)'))
            elif ema.fast_above_slow:
                votes.append(IndicatorVote('ema_cross', +1, INDICATOR_WEIGHTS['ema_cross'] * 0.5,
                    f'EMA{config.EMA_FAST} above EMA{config.EMA_SLOW} — bullish bias (1H)'))
            else:
                votes.append(IndicatorVote('ema_cross', -1, INDICATOR_WEIGHTS['ema_cross'] * 0.5,
                    f'EMA{config.EMA_FAST} below EMA{config.EMA_SLOW} — bearish bias (1H)'))
    except Exception as e:
        logger.debug(f"EMA cross error ({ticker}): {e}")

    # ── 2. RSI — 1H ──────────────────────────────────────────
    try:
        rsi_result = rsi(df_1h, period=config.RSI_PERIOD)
        if rsi_result is not None:
            val = rsi_result.value
            if val <= 30:
                votes.append(IndicatorVote('rsi', +1, INDICATOR_WEIGHTS['rsi'],
                    f'RSI oversold ({val:.1f}) — 1H'))
            elif val >= 70:
                votes.append(IndicatorVote('rsi', -1, INDICATOR_WEIGHTS['rsi'],
                    f'RSI overbought ({val:.1f}) — 1H'))
            elif val >= 55:
                votes.append(IndicatorVote('rsi', +1, INDICATOR_WEIGHTS['rsi'] * 0.4,
                    f'RSI bullish zone ({val:.1f}) — 1H'))
            elif val <= 45:
                votes.append(IndicatorVote('rsi', -1, INDICATOR_WEIGHTS['rsi'] * 0.4,
                    f'RSI bearish zone ({val:.1f}) — 1H'))
            else:
                votes.append(IndicatorVote('rsi', 0, INDICATOR_WEIGHTS['rsi'],
                    f'RSI neutral ({val:.1f}) — 1H'))
    except Exception as e:
        logger.debug(f"RSI error ({ticker}): {e}")

    # ── 3. MACD — 1H ─────────────────────────────────────────
    try:
        macd_result = macd(df_1h,
                           fast=config.MACD_FAST,
                           slow=config.MACD_SLOW,
                           signal=config.MACD_SIGNAL)
        if macd_result is not None:
            if macd_result.cross_up:
                votes.append(IndicatorVote('macd', +1, INDICATOR_WEIGHTS['macd'],
                    f'MACD bullish cross (hist={macd_result.histogram:+.4f}) — 1H'))
            elif macd_result.cross_down:
                votes.append(IndicatorVote('macd', -1, INDICATOR_WEIGHTS['macd'],
                    f'MACD bearish cross (hist={macd_result.histogram:+.4f}) — 1H'))
            elif macd_result.histogram > 0:
                votes.append(IndicatorVote('macd', +1, INDICATOR_WEIGHTS['macd'] * 0.5,
                    f'MACD histogram positive ({macd_result.histogram:+.4f}) — 1H'))
            else:
                votes.append(IndicatorVote('macd', -1, INDICATOR_WEIGHTS['macd'] * 0.5,
                    f'MACD histogram negative ({macd_result.histogram:+.4f}) — 1H'))
    except Exception as e:
        logger.debug(f"MACD error ({ticker}): {e}")

    # ── 4. Bollinger Bands — 1H ───────────────────────────────
    try:
        bb = bollinger_bands(df_1h, period=config.BB_PERIOD, std=config.BB_STD)
        if bb is not None:
            current_close = float(df_1h['close'].iloc[-1])
            if current_close <= bb.lower:
                votes.append(IndicatorVote('bollinger', +1, INDICATOR_WEIGHTS['bollinger'],
                    f'Price at/below lower BB (${bb.lower:,.0f}) — 1H'))
            elif current_close >= bb.upper:
                votes.append(IndicatorVote('bollinger', -1, INDICATOR_WEIGHTS['bollinger'],
                    f'Price at/above upper BB (${bb.upper:,.0f}) — 1H'))
            else:
                votes.append(IndicatorVote('bollinger', 0, INDICATOR_WEIGHTS['bollinger'],
                    f'Price inside BBands — 1H'))
    except Exception as e:
        logger.debug(f"Bollinger error ({ticker}): {e}")

    # ── 5. ATR trend confirmation — 1D ───────────────────────
    try:
        if df_1d is not None and not df_1d.empty and len(df_1d) >= config.ATR_PERIOD + 5:
            atr_result = atr(df_1d, period=config.ATR_PERIOD)
            if atr_result is not None:
                # High ATR = expanding volatility. Combine with trend for vote.
                atr_pct = atr_result.value / float(df_1d['close'].iloc[-1]) * 100
                if atr_pct > 3.0 and trend == 'uptrend':
                    votes.append(IndicatorVote('atr_trend', +1, INDICATOR_WEIGHTS['atr_trend'],
                        f'High ATR ({atr_pct:.1f}%) in uptrend — 1D'))
                elif atr_pct > 3.0 and trend == 'downtrend':
                    votes.append(IndicatorVote('atr_trend', -1, INDICATOR_WEIGHTS['atr_trend'],
                        f'High ATR ({atr_pct:.1f}%) in downtrend — 1D'))
                else:
                    votes.append(IndicatorVote('atr_trend', 0, INDICATOR_WEIGHTS['atr_trend'],
                        f'ATR ({atr_pct:.1f}%) — low conviction'))
    except Exception as e:
        logger.debug(f"ATR error ({ticker}): {e}")

    # ══════════════════════════════════════════════════════════
    # SMC LAYER (votes 6–15)
    # ══════════════════════════════════════════════════════════

    # ── 6. Order Block — 1H ──────────────────────────────────
    try:
        ob = order_block(df_1h,
                         impulse_pct=config.SMC_OB_IMPULSE_PCT,
                         proximity_pct=config.SMC_OB_PROXIMITY_PCT)
        if ob is not None:
            if ob.bullish_ob and ob.near_bullish:
                votes.append(IndicatorVote('smc_order_block', +1, INDICATOR_WEIGHTS['smc_order_block'],
                    f'Bullish OB at ${ob.bullish_level:,.0f} — price in proximity — 1H'))
            elif ob.bearish_ob and ob.near_bearish:
                votes.append(IndicatorVote('smc_order_block', -1, INDICATOR_WEIGHTS['smc_order_block'],
                    f'Bearish OB at ${ob.bearish_level:,.0f} — price in proximity — 1H'))
            else:
                votes.append(IndicatorVote('smc_order_block', 0, INDICATOR_WEIGHTS['smc_order_block'],
                    'No order block proximity — 1H'))
    except Exception as e:
        logger.debug(f"Order block error ({ticker}): {e}")

    # ── 7. Fair Value Gap — 1H ───────────────────────────────
    try:
        fvg = fair_value_gap(df_1h, proximity_pct=config.SMC_FVG_PROXIMITY_PCT)
        if fvg is not None:
            if fvg.bullish_fvg and fvg.price_in_bullish:
                votes.append(IndicatorVote('smc_fvg', +1, INDICATOR_WEIGHTS['smc_fvg'],
                    f'Bullish FVG [{fvg.bull_low:,.0f}–{fvg.bull_high:,.0f}] — price inside — 1H'))
            elif fvg.bearish_fvg and fvg.price_in_bearish:
                votes.append(IndicatorVote('smc_fvg', -1, INDICATOR_WEIGHTS['smc_fvg'],
                    f'Bearish FVG [{fvg.bear_low:,.0f}–{fvg.bear_high:,.0f}] — price inside — 1H'))
            else:
                votes.append(IndicatorVote('smc_fvg', 0, INDICATOR_WEIGHTS['smc_fvg'],
                    'No FVG proximity — 1H'))
    except Exception as e:
        logger.debug(f"FVG error ({ticker}): {e}")

    # ── 8. BOS / CHoCH — 1H ──────────────────────────────────
    try:
        bos = bos_choch(df_1h, max_bars=config.SMC_BOS_MAX_BARS)
        if bos is not None:
            if bos.bullish_bos:
                votes.append(IndicatorVote('smc_bos_choch', +1, INDICATOR_WEIGHTS['smc_bos_choch'],
                    f'Bullish BOS — structure break up — 1H'))
            elif bos.bearish_bos:
                votes.append(IndicatorVote('smc_bos_choch', -1, INDICATOR_WEIGHTS['smc_bos_choch'],
                    f'Bearish BOS — structure break down — 1H'))
            elif bos.bullish_choch:
                votes.append(IndicatorVote('smc_bos_choch', +1, INDICATOR_WEIGHTS['smc_bos_choch'],
                    f'CHoCH bullish — trend reversal signal — 1H'))
            elif bos.bearish_choch:
                votes.append(IndicatorVote('smc_bos_choch', -1, INDICATOR_WEIGHTS['smc_bos_choch'],
                    f'CHoCH bearish — trend reversal signal — 1H'))
            else:
                votes.append(IndicatorVote('smc_bos_choch', 0, INDICATOR_WEIGHTS['smc_bos_choch'],
                    'No BOS/CHoCH — 1H'))
    except Exception as e:
        logger.debug(f"BOS/CHoCH error ({ticker}): {e}")

    # ── 9. Liquidity Sweep — 1H ──────────────────────────────
    try:
        sweep = liquidity_sweep(df_1h, max_bars=config.SMC_SWEEP_MAX_BARS)
        if sweep is not None:
            if sweep.bull_sweep:
                votes.append(IndicatorVote('smc_liquidity', +1, INDICATOR_WEIGHTS['smc_liquidity'],
                    f'Bullish liquidity sweep (swept low at ${sweep.swept_level:,.0f}) — 1H'))
            elif sweep.bear_sweep:
                votes.append(IndicatorVote('smc_liquidity', -1, INDICATOR_WEIGHTS['smc_liquidity'],
                    f'Bearish liquidity sweep (swept high at ${sweep.swept_level:,.0f}) — 1H'))
            else:
                votes.append(IndicatorVote('smc_liquidity', 0, INDICATOR_WEIGHTS['smc_liquidity'],
                    'No liquidity sweep — 1H'))
    except Exception as e:
        logger.debug(f"Liquidity sweep error ({ticker}): {e}")

    # Also run key indicators on 1D for higher-timeframe confluence
    # ── 10–15. Daily timeframe repeats (order block, FVG, BOS) ─
    if df_1d is not None and not df_1d.empty and len(df_1d) >= 20:
        try:
            ob_1d = order_block(df_1d,
                                impulse_pct=config.SMC_OB_IMPULSE_PCT,
                                proximity_pct=config.SMC_OB_PROXIMITY_PCT * 1.5)
            if ob_1d is not None:
                if ob_1d.bullish_ob and ob_1d.near_bullish:
                    votes.append(IndicatorVote('smc_order_block', +1,
                        INDICATOR_WEIGHTS['smc_order_block'] * 0.7,
                        f'Bullish OB (1D) at ${ob_1d.bullish_level:,.0f}'))
                elif ob_1d.bearish_ob and ob_1d.near_bearish:
                    votes.append(IndicatorVote('smc_order_block', -1,
                        INDICATOR_WEIGHTS['smc_order_block'] * 0.7,
                        f'Bearish OB (1D) at ${ob_1d.bearish_level:,.0f}'))
        except Exception as e:
            logger.debug(f"1D OB error ({ticker}): {e}")

    # ══════════════════════════════════════════════════════════
    # ADVANCED TA LAYER (v3) — votes 16–22
    # ══════════════════════════════════════════════════════════

    # ── 16. AVWAP Pinch (Brian Shannon) — 1H ─────────────────
    pinch_1h = None
    try:
        pinch_1h = avwap_pinch(df_1h, lookback=config.AVWAP_LOOKBACK) if not df_1h.empty else None
        if pinch_1h:
            if pinch_1h.breakout == 'bullish' and pinch_1h.slope_filter:
                votes.append(IndicatorVote(
                    'avwap_pinch', +1, INDICATOR_WEIGHTS['avwap_pinch'],
                    f'AVWAP pinch bullish breakout (width={pinch_1h.width_pct:.1f}%) '
                    f'— slope filter ✓ (avg buyer in profit)'
                ))
            elif pinch_1h.breakout == 'bullish' and not pinch_1h.slope_filter:
                votes.append(IndicatorVote(
                    'avwap_pinch', +1, INDICATOR_WEIGHTS['avwap_pinch'] * 0.5,
                    f'AVWAP pinch bullish breakout but slope filter ✗ — reduced weight'
                ))
            elif pinch_1h.breakout == 'bearish':
                votes.append(IndicatorVote(
                    'avwap_pinch', -1, INDICATOR_WEIGHTS['avwap_pinch'],
                    f'AVWAP pinch bearish breakout (width={pinch_1h.width_pct:.1f}%)'
                ))
            elif pinch_1h.pinch_active:
                votes.append(IndicatorVote(
                    'avwap_pinch', 0, INDICATOR_WEIGHTS['avwap_pinch'],
                    f'AVWAP pinch ACTIVE (width={pinch_1h.width_pct:.1f}%) — no breakout yet'
                ))
            else:
                votes.append(IndicatorVote(
                    'avwap_pinch', 0, INDICATOR_WEIGHTS['avwap_pinch'],
                    'No AVWAP pinch pattern'
                ))
    except Exception as e:
        logger.debug(f"AVWAP pinch error ({ticker}): {e}")

    # ── 17. Williams Alligator (Bill Williams) — 1H ──────────
    gator_1h = None
    try:
        gator_1h = alligator(df_1h) if not df_1h.empty else None
        if gator_1h:
            if gator_1h.phase == 'sleeping':
                votes.append(IndicatorVote(
                    'williams_alligator', 0, INDICATOR_WEIGHTS['williams_alligator'],
                    'Alligator SLEEPING — choppy market, no edge (hard HOLD)'
                ))
            elif gator_1h.phase == 'eating_bull':
                votes.append(IndicatorVote(
                    'williams_alligator', +1, INDICATOR_WEIGHTS['williams_alligator'],
                    f'Alligator EATING BULL (lips={gator_1h.lips:.0f} > '
                    f'teeth={gator_1h.teeth:.0f} > jaw={gator_1h.jaw:.0f})'
                ))
            elif gator_1h.phase == 'eating_bear':
                votes.append(IndicatorVote(
                    'williams_alligator', -1, INDICATOR_WEIGHTS['williams_alligator'],
                    f'Alligator EATING BEAR (lips={gator_1h.lips:.0f} < '
                    f'teeth={gator_1h.teeth:.0f} < jaw={gator_1h.jaw:.0f})'
                ))
            elif gator_1h.phase == 'awakening':
                direction = +1 if gator_1h.lips_above_jaw else -1
                votes.append(IndicatorVote(
                    'williams_alligator', direction,
                    INDICATOR_WEIGHTS['williams_alligator'] * 0.5,
                    f'Alligator AWAKENING ({"bull" if direction > 0 else "bear"}) — half weight'
                ))
            else:  # sated
                votes.append(IndicatorVote(
                    'williams_alligator', 0, INDICATOR_WEIGHTS['williams_alligator'],
                    'Alligator SATED — trend losing momentum'
                ))
    except Exception as e:
        logger.debug(f"Alligator error ({ticker}): {e}")

    # ── 18. Awesome Oscillator (Bill Williams) — 1H ──────────
    try:
        ao_1h = awesome_oscillator(df_1h) if not df_1h.empty else None
        if ao_1h:
            if ao_1h.signal == 'bullish_cross':
                votes.append(IndicatorVote(
                    'williams_ao', +1, INDICATOR_WEIGHTS['williams_ao'],
                    f'AO zero-line cross BULLISH (val={ao_1h.value:+.4f})'
                ))
            elif ao_1h.signal == 'bearish_cross':
                votes.append(IndicatorVote(
                    'williams_ao', -1, INDICATOR_WEIGHTS['williams_ao'],
                    f'AO zero-line cross BEARISH (val={ao_1h.value:+.4f})'
                ))
            elif ao_1h.signal == 'bullish':
                votes.append(IndicatorVote(
                    'williams_ao', +1, INDICATOR_WEIGHTS['williams_ao'] * 0.6,
                    f'AO above zero and rising ({ao_1h.value:+.4f})'
                ))
            elif ao_1h.signal == 'bearish':
                votes.append(IndicatorVote(
                    'williams_ao', -1, INDICATOR_WEIGHTS['williams_ao'] * 0.6,
                    f'AO below zero and falling ({ao_1h.value:+.4f})'
                ))
            else:
                votes.append(IndicatorVote(
                    'williams_ao', 0, INDICATOR_WEIGHTS['williams_ao'],
                    f'AO neutral ({ao_1h.value:+.4f})'
                ))
    except Exception as e:
        logger.debug(f"AO error ({ticker}): {e}")

    # ── 19. Williams Fractals — 1H ────────────────────────────
    try:
        frac_1h = fractals(df_1h, lookback=60) if not df_1h.empty else None
        if frac_1h:
            if frac_1h.breakout_above_up and frac_1h.bars_since_up <= 10:
                votes.append(IndicatorVote(
                    'williams_fractals', +1, INDICATOR_WEIGHTS['williams_fractals'],
                    f'Fractal breakout UP — broke ${frac_1h.last_up_fractal:,.0f} '
                    f'({frac_1h.bars_since_up} bars ago)'
                ))
            elif frac_1h.breakout_below_dn and frac_1h.bars_since_down <= 10:
                votes.append(IndicatorVote(
                    'williams_fractals', -1, INDICATOR_WEIGHTS['williams_fractals'],
                    f'Fractal breakdown DN — broke ${frac_1h.last_down_fractal:,.0f} '
                    f'({frac_1h.bars_since_down} bars ago)'
                ))
            else:
                votes.append(IndicatorVote(
                    'williams_fractals', 0, INDICATOR_WEIGHTS['williams_fractals'],
                    'No recent fractal breakout'
                ))
    except Exception as e:
        logger.debug(f"Fractals error ({ticker}): {e}")

    # ── 20. Relative Volume / "In Play" Filter (Bellafiore) — 1H ─
    try:
        rvol_1h = volume_relative(df_1h, lookback=20,
                                  threshold=config.RVOL_MIN_THRESHOLD) if not df_1h.empty else None
        if rvol_1h:
            if rvol_1h.is_in_play:
                last_open_v  = float(df_1h['open'].iloc[-1])
                last_close_v = float(df_1h['close'].iloc[-1])
                if last_close_v > last_open_v * 1.001:
                    votes.append(IndicatorVote(
                        'rvol_filter', +1, INDICATOR_WEIGHTS['rvol_filter'],
                        f'RVOL {rvol_1h.rvol:.1f}x — IN PLAY (bull candle, smart money present)'
                    ))
                elif last_close_v < last_open_v * 0.999:
                    votes.append(IndicatorVote(
                        'rvol_filter', -1, INDICATOR_WEIGHTS['rvol_filter'],
                        f'RVOL {rvol_1h.rvol:.1f}x — IN PLAY (bear candle, smart money present)'
                    ))
                else:
                    votes.append(IndicatorVote(
                        'rvol_filter', 0, INDICATOR_WEIGHTS['rvol_filter'],
                        f'RVOL {rvol_1h.rvol:.1f}x — in play but neutral candle'
                    ))
            else:
                votes.append(IndicatorVote(
                    'rvol_filter', 0, INDICATOR_WEIGHTS['rvol_filter'],
                    f'RVOL {rvol_1h.rvol:.1f}x — NOT in play (< {config.RVOL_MIN_THRESHOLD}x avg)'
                ))
    except Exception as e:
        logger.debug(f"RVOL error ({ticker}): {e}")

    # ── 21. Trap Detector (Schwager) — 1H ────────────────────
    trap_1h = None
    try:
        trap_1h = trap_detector(df_1h, lookback=30,
                                min_breakout_pct=0.3,
                                max_trap_bars=config.TRAP_MAX_BARS) if not df_1h.empty else None
        if trap_1h and trap_1h.trap_type is not None:
            if trap_1h.trap_type == 'bull_trap' and trap_1h.bars_ago <= config.TRAP_MAX_BARS:
                votes.append(IndicatorVote(
                    'trap_detector', -1, INDICATOR_WEIGHTS['trap_detector'],
                    f'BULL TRAP detected at ${trap_1h.trap_level:,.0f} '
                    f'({trap_1h.bars_ago} bars ago, reversal={trap_1h.reversal_pct:.1f}%) '
                    f'— FLIP TO SELL'
                ))
            elif trap_1h.trap_type == 'bear_trap' and trap_1h.bars_ago <= config.TRAP_MAX_BARS:
                votes.append(IndicatorVote(
                    'trap_detector', +1, INDICATOR_WEIGHTS['trap_detector'],
                    f'BEAR TRAP detected at ${trap_1h.trap_level:,.0f} '
                    f'({trap_1h.bars_ago} bars ago, reversal={trap_1h.reversal_pct:.1f}%) '
                    f'— FLIP TO BUY'
                ))
            else:
                votes.append(IndicatorVote(
                    'trap_detector', 0, INDICATOR_WEIGHTS['trap_detector'],
                    f'Trap detected but stale ({trap_1h.bars_ago} bars ago)'
                ))
        else:
            votes.append(IndicatorVote(
                'trap_detector', 0, INDICATOR_WEIGHTS['trap_detector'],
                'No trap pattern detected'
            ))
    except Exception as e:
        logger.debug(f"Trap detector error ({ticker}): {e}")

    # ── 22. Elliott Wave Divergence — 1H ─────────────────────
    ew_1h = None
    try:
        ew_1h = elliott_divergence(df_1h, lookback=config.ELLIOTT_LOOKBACK) if not df_1h.empty else None
        if ew_1h:
            if ew_1h.wave_context == 'wave3_likely':
                votes.append(IndicatorVote(
                    'elliott_divergence', +1, INDICATOR_WEIGHTS['elliott_divergence'],
                    f'Elliott Wave 3 likely — RSI confirming price high '
                    f'(rsi={ew_1h.rsi_at_high:.1f} at ${ew_1h.price_high:,.0f})'
                ))
            elif ew_1h.wave_context == 'wave5_warning':
                votes.append(IndicatorVote(
                    'elliott_divergence', 0, INDICATOR_WEIGHTS['elliott_divergence'],
                    f'Elliott Wave 5 WARNING — price high but RSI diverging '
                    f'(rsi={ew_1h.rsi_at_high:.1f}) — reduce position size'
                ))
            elif ew_1h.wave_context == 'wave_a_likely':
                votes.append(IndicatorVote(
                    'elliott_divergence', +1, INDICATOR_WEIGHTS['elliott_divergence'] * 0.7,
                    f'Elliott Wave A — buyers absorbing selling pressure '
                    f'(rsi divergence at ${ew_1h.price_high:,.0f})'
                ))
            else:
                votes.append(IndicatorVote(
                    'elliott_divergence', 0, INDICATOR_WEIGHTS['elliott_divergence'],
                    'No Elliott divergence pattern'
                ))
    except Exception as e:
        logger.debug(f"Elliott divergence error ({ticker}): {e}")

    # ══════════════════════════════════════════════════════════
    # AGGREGATE
    # ══════════════════════════════════════════════════════════

    bull_score, bear_score, net_score = _aggregate(votes)

    # Build a compact summary string
    bull_votes = [v for v in votes if v.vote == +1]
    bear_votes = [v for v in votes if v.vote == -1]
    summary_parts = []
    if bull_votes:
        summary_parts.append('Bull: ' + '; '.join(v.name for v in bull_votes[:3]))
    if bear_votes:
        summary_parts.append('Bear: ' + '; '.join(v.name for v in bear_votes[:3]))
    summary = ' | '.join(summary_parts) or 'No directional signals'

    # ── v3 flags ─────────────────────────────────────────────
    alligator_sleeping = (gator_1h is not None and gator_1h.phase == 'sleeping')
    wave5_warning      = (ew_1h is not None and ew_1h.wave_context == 'wave5_warning')
    trap_active        = (
        trap_1h is not None and
        trap_1h.trap_type is not None and
        trap_1h.bars_ago <= config.TRAP_MAX_BARS
    )
    trap_sig           = (trap_1h.signal if trap_active and trap_1h else 'none')

    logger.debug(
        f"Analysis {ticker}: trend={trend} bull={bull_score:.0%} "
        f"bear={bear_score:.0%} net={net_score:+.3f} votes={len(votes)}"
    )

    return AnalysisResult(
        ticker=ticker,
        trend=trend,
        bull_score=bull_score,
        bear_score=bear_score,
        net_score=net_score,
        votes=votes,
        summary=summary,
        alligator_sleeping=alligator_sleeping,
        wave5_warning=wave5_warning,
        trap_active=trap_active,
        trap_signal=trap_sig,
    )