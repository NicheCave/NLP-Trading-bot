# market_analyzer.py — multi-timeframe TA + SMC engine (v2)
# ─────────────────────────────────────────────────────────────
# Changes from v1:
#   - Parallelised bar fetching (ThreadPoolExecutor) — ~2s faster per cycle
#   - RSI: bearish_zone now casts 0 (neutral), not -0.5 (was silently penalising flat markets)
#   - RSI: neutral zone (45–55) explicitly casts 0
#   - MACD: bullish_cross / bearish_cross carry higher weight than plain bullish / bearish
#   - Volume: now uses candle direction (from VolumeResult.direction), not EMA state
#   - Fibonacci: uses fixed at_support / at_resistance flags — logic was inverted in v1
#   - Trend regression window now 80 bars (passed through to trend_structure)
#   - NEW: SMC vote layer — order blocks, FVGs, BOS/CHoCH, liquidity sweeps
#   - TOTAL_WEIGHT constant removed (was computed but never used)
# ─────────────────────────────────────────────────────────────
from __future__ import annotations
from dataclasses import dataclass, field
from loguru import logger
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from market_data import fetch_multi_timeframe, get_current_price
from indicators import (
    ema_cross, macd, rsi, bollinger_bands, atr,
    volume_analysis, detect_swing_levels, fibonacci_retracements,
    trend_structure, order_blocks, fair_value_gaps,
    break_of_structure, liquidity_sweep,
    EMASignal, MACDResult, RSIResult,
    BollingerResult, ATRResult, VolumeResult, SwingLevel,
    FiboResult, TrendStructure,
)


# ──────────────────────────────────────────────────────────────
# INDICATOR WEIGHTS
# ──────────────────────────────────────────────────────────────

INDICATOR_WEIGHTS: dict[str, float] = {
    # Trend — most reliable for crypto swing trades
    'ema_cross_1h':       3.0,
    'macd_1h':            2.5,
    'trend_1d':           2.5,
    # Momentum
    'rsi_1h':             2.0,
    'rsi_1d':             1.5,
    # Volatility / structure
    'bollinger_1h':       1.5,
    'fibo_proximity':     2.0,
    'swing_proximity':    1.5,
    # Volume — now independent of EMA state
    'volume_1h':          1.5,
    # Short-term trigger
    'ema_cross_1m':       1.0,
    'macd_1m':            1.0,
    # ── SMC layer (new in v2) ─────────────────────────────────
    'smc_order_block':    2.5,   # price at/near an unmitigated OB
    'smc_fvg':            1.5,   # price at an unfilled FVG
    'smc_bos_choch':      2.0,   # BOS or CHoCH event (recent)
    'smc_liquidity':      2.5,   # liquidity sweep + reversal (highest confluence)
}


@dataclass
class IndicatorVote:
    name: str
    vote: int       # +1, 0, -1
    weight: float
    reason: str


@dataclass
class AnalysisResult:
    ticker: str
    price: float
    bull_score: float    # 0.0–1.0
    bear_score: float    # 0.0–1.0
    net_score: float     # -1.0 to +1.0
    votes: list[IndicatorVote] = field(default_factory=list)
    atr_pct: float = 0.0
    nearest_support: float = 0.0
    nearest_resistance: float = 0.0
    trend: str = 'ranging'
    summary: str = ''


def _fetch_bars_parallel(ticker: str) -> dict[str, pd.DataFrame]:
    """
    Fetch 1m / 1h / 1d bars concurrently to cut cycle time by ~2s.
    Falls back to sequential if the executor fails.
    """
    from market_data import fetch_bars
    from alpaca.data.timeframe import TimeFrame

    tasks = {
        '1m': (ticker, TimeFrame.Minute, 100),
        '1h': (ticker, TimeFrame.Hour,   200),
        '1d': (ticker, TimeFrame.Day,    200),
    }
    results: dict[str, pd.DataFrame] = {}

    try:
        with ThreadPoolExecutor(max_workers=3) as ex:
            future_map = {
                ex.submit(fetch_bars, t, tf, lim): key
                for key, (t, tf, lim) in tasks.items()
            }
            for fut in as_completed(future_map):
                key = future_map[fut]
                try:
                    results[key] = fut.result()
                except Exception as e:
                    logger.warning(f"Parallel bar fetch failed for {key}: {e}")
                    results[key] = pd.DataFrame()
    except Exception as e:
        logger.warning(f"ThreadPoolExecutor failed — falling back to sequential: {e}")
        results = fetch_multi_timeframe(ticker)

    return results


def analyze(ticker: str) -> Optional[AnalysisResult]:
    """
    Run full multi-timeframe TA + SMC analysis for a crypto ticker.
    Returns AnalysisResult or None if insufficient data.
    """
    bars  = _fetch_bars_parallel(ticker)
    df_1m = bars.get('1m', pd.DataFrame())
    df_1h = bars.get('1h', pd.DataFrame())
    df_1d = bars.get('1d', pd.DataFrame())

    price = get_current_price(ticker)
    if price is None:
        logger.warning(f"Cannot analyze {ticker} — no price data")
        return None

    votes: list[IndicatorVote] = []

    # ── 1. EMA Cross — 1H ─────────────────────────────────────
    ec_1h = ema_cross(df_1h) if not df_1h.empty else None
    if ec_1h:
        if ec_1h.cross == 'golden':
            votes.append(IndicatorVote('ema_cross_1h', +1, INDICATOR_WEIGHTS['ema_cross_1h'],
                'Golden cross 1H'))
        elif ec_1h.cross == 'death':
            votes.append(IndicatorVote('ema_cross_1h', -1, INDICATOR_WEIGHTS['ema_cross_1h'],
                'Death cross 1H'))
        elif ec_1h.cross == 'above' and ec_1h.slope_pct > 0.1:
            votes.append(IndicatorVote('ema_cross_1h', +1, INDICATOR_WEIGHTS['ema_cross_1h'],
                f'EMA fast above slow, rising {ec_1h.slope_pct:.2f}%/bar'))
        elif ec_1h.cross == 'below' and ec_1h.slope_pct < -0.1:
            votes.append(IndicatorVote('ema_cross_1h', -1, INDICATOR_WEIGHTS['ema_cross_1h'],
                f'EMA fast below slow, falling {ec_1h.slope_pct:.2f}%/bar'))
        else:
            votes.append(IndicatorVote('ema_cross_1h', 0, INDICATOR_WEIGHTS['ema_cross_1h'],
                'EMA flat/neutral'))

    # ── 2. MACD — 1H ──────────────────────────────────────────
    # FIX: bullish_cross carries full weight, plain 'bullish' carries 60% (weaker signal)
    mc_1h = macd(df_1h) if not df_1h.empty else None
    if mc_1h:
        if mc_1h.divergence == 'bullish_cross':
            votes.append(IndicatorVote('macd_1h', +1, INDICATOR_WEIGHTS['macd_1h'],
                f'MACD zero-line cross bullish 1H (hist={mc_1h.hist:+.4f})'))
        elif mc_1h.divergence == 'bearish_cross':
            votes.append(IndicatorVote('macd_1h', -1, INDICATOR_WEIGHTS['macd_1h'],
                f'MACD zero-line cross bearish 1H (hist={mc_1h.hist:+.4f})'))
        elif mc_1h.divergence == 'bullish':
            votes.append(IndicatorVote('macd_1h', +1, INDICATOR_WEIGHTS['macd_1h'] * 0.6,
                f'MACD hist expanding bullish 1H ({mc_1h.hist:+.4f})'))
        elif mc_1h.divergence == 'bearish':
            votes.append(IndicatorVote('macd_1h', -1, INDICATOR_WEIGHTS['macd_1h'] * 0.6,
                f'MACD hist expanding bearish 1H ({mc_1h.hist:+.4f})'))
        else:
            votes.append(IndicatorVote('macd_1h', 0, INDICATOR_WEIGHTS['macd_1h'],
                'MACD neutral 1H'))

    # ── 3. Trend Structure — 1D ────────────────────────────────
    ts_1d = trend_structure(df_1d, regression_bars=80) if not df_1d.empty else None
    if ts_1d:
        if ts_1d.direction == 'uptrend' and ts_1d.strength > 0.6:
            votes.append(IndicatorVote('trend_1d', +1, INDICATOR_WEIGHTS['trend_1d'],
                f'Daily uptrend R²={ts_1d.strength:.2f}'))
        elif ts_1d.direction == 'downtrend' and ts_1d.strength > 0.6:
            votes.append(IndicatorVote('trend_1d', -1, INDICATOR_WEIGHTS['trend_1d'],
                f'Daily downtrend R²={ts_1d.strength:.2f} (bear market)'))
        else:
            votes.append(IndicatorVote('trend_1d', 0, INDICATOR_WEIGHTS['trend_1d'],
                'No clear daily trend'))

    # ── 4. RSI — 1H ───────────────────────────────────────────
    # FIX: bearish_zone → 0 vote (not -0.5); neutral → 0 vote
    rsi_1h = rsi(df_1h) if not df_1h.empty else None
    if rsi_1h:
        if rsi_1h.zone == 'oversold':
            votes.append(IndicatorVote('rsi_1h', +1, INDICATOR_WEIGHTS['rsi_1h'],
                f'RSI 1H oversold ({rsi_1h.value:.1f})'))
        elif rsi_1h.zone == 'overbought':
            votes.append(IndicatorVote('rsi_1h', -1, INDICATOR_WEIGHTS['rsi_1h'],
                f'RSI 1H overbought ({rsi_1h.value:.1f})'))
        elif rsi_1h.zone == 'bullish_zone':
            votes.append(IndicatorVote('rsi_1h', +1, INDICATOR_WEIGHTS['rsi_1h'] * 0.5,
                f'RSI 1H bullish zone ({rsi_1h.value:.1f})'))
        else:
            # bearish_zone and neutral both cast 0 — neither is a strong directional signal
            votes.append(IndicatorVote('rsi_1h', 0, INDICATOR_WEIGHTS['rsi_1h'],
                f'RSI 1H neutral/weak ({rsi_1h.value:.1f})'))

    # ── 5. RSI — 1D ───────────────────────────────────────────
    rsi_1d = rsi(df_1d) if not df_1d.empty else None
    if rsi_1d:
        if rsi_1d.zone == 'oversold':
            votes.append(IndicatorVote('rsi_1d', +1, INDICATOR_WEIGHTS['rsi_1d'],
                f'RSI 1D oversold ({rsi_1d.value:.1f}) — strong bounce candidate'))
        elif rsi_1d.zone == 'overbought':
            votes.append(IndicatorVote('rsi_1d', -1, INDICATOR_WEIGHTS['rsi_1d'],
                f'RSI 1D overbought ({rsi_1d.value:.1f})'))
        else:
            votes.append(IndicatorVote('rsi_1d', 0, INDICATOR_WEIGHTS['rsi_1d'],
                f'RSI 1D neutral ({rsi_1d.value:.1f})'))

    # ── 6. Bollinger Bands — 1H ───────────────────────────────
    bb_1h = bollinger_bands(df_1h) if not df_1h.empty else None
    if bb_1h:
        if bb_1h.signal == 'breakout_up':
            votes.append(IndicatorVote('bollinger_1h', +1, INDICATOR_WEIGHTS['bollinger_1h'],
                'BB breakout above upper band'))
        elif bb_1h.signal == 'breakout_down':
            votes.append(IndicatorVote('bollinger_1h', -1, INDICATOR_WEIGHTS['bollinger_1h'],
                'BB breakdown below lower band'))
        elif bb_1h.signal == 'squeeze':
            votes.append(IndicatorVote('bollinger_1h', 0, INDICATOR_WEIGHTS['bollinger_1h'],
                f'BB squeeze ({bb_1h.width_pct:.1f}%) — breakout imminent'))
        elif bb_1h.price_position < 0.2:
            votes.append(IndicatorVote('bollinger_1h', +1, INDICATOR_WEIGHTS['bollinger_1h'],
                'Price at lower BB — mean reversion potential'))
        elif bb_1h.price_position > 0.8:
            votes.append(IndicatorVote('bollinger_1h', -1, INDICATOR_WEIGHTS['bollinger_1h'],
                'Price at upper BB — potential rejection'))
        else:
            votes.append(IndicatorVote('bollinger_1h', 0, INDICATOR_WEIGHTS['bollinger_1h'],
                'Price mid-bands'))

    # ── 7. Fibonacci Proximity — 1D ───────────────────────────
    # FIX: uses at_support / at_resistance from corrected FiboResult
    fibo = fibonacci_retracements(df_1d) if not df_1d.empty else None
    if fibo:
        if fibo.nearest_level in [0.382, 0.5, 0.618]:
            if fibo.at_support:
                votes.append(IndicatorVote('fibo_proximity', +1, INDICATOR_WEIGHTS['fibo_proximity'],
                    f'At {fibo.nearest_level:.1%} Fib support — whale buy zone ({fibo.distance_pct:.1f}% away)'))
            elif fibo.at_resistance:
                votes.append(IndicatorVote('fibo_proximity', -1, INDICATOR_WEIGHTS['fibo_proximity'],
                    f'Approaching {fibo.nearest_level:.1%} Fib resistance ({fibo.distance_pct:.1f}% away)'))
            else:
                votes.append(IndicatorVote('fibo_proximity', 0, INDICATOR_WEIGHTS['fibo_proximity'],
                    f'Near {fibo.nearest_level:.1%} Fib but not close enough ({fibo.distance_pct:.1f}% away)'))
        else:
            votes.append(IndicatorVote('fibo_proximity', 0, INDICATOR_WEIGHTS['fibo_proximity'],
                f'Near minor Fib {fibo.nearest_level:.1%} ({fibo.distance_pct:.1f}% away)'))

    # ── 8. Swing Level Proximity — 1D ─────────────────────────
    swings             = detect_swing_levels(df_1d) if not df_1d.empty else []
    nearest_support    = 0.0
    nearest_resistance = 0.0

    if swings:
        supports = [s for s in swings if s.label == 'support'    and s.direction == 'below']
        resists  = [r for r in swings if r.label == 'resistance' and r.direction == 'above']

        if supports:
            nearest_support = supports[0].level
            if supports[0].distance_pct < 3.0:
                votes.append(IndicatorVote('swing_proximity', +1, INDICATOR_WEIGHTS['swing_proximity'],
                    f'Near major support ${nearest_support:,.0f} ({supports[0].distance_pct:.1f}% below)'))
            else:
                votes.append(IndicatorVote('swing_proximity', 0, INDICATOR_WEIGHTS['swing_proximity'],
                    f'Support ${nearest_support:,.0f} is {supports[0].distance_pct:.1f}% away'))

        if resists:
            nearest_resistance = resists[0].level
            if resists[0].distance_pct < 3.0:
                votes.append(IndicatorVote('swing_proximity', -1, INDICATOR_WEIGHTS['swing_proximity'],
                    f'Approaching resistance ${nearest_resistance:,.0f} ({resists[0].distance_pct:.1f}% above)'))
    else:
        votes.append(IndicatorVote('swing_proximity', 0, INDICATOR_WEIGHTS['swing_proximity'],
            'No clear swing levels detected'))

    # ── 9. Volume — 1H ────────────────────────────────────────
    # FIX: uses VolumeResult.direction (candle body), not EMA state
    vol_1h = volume_analysis(df_1h) if not df_1h.empty else None
    if vol_1h:
        if vol_1h.spike:
            if vol_1h.direction == 'bullish':
                votes.append(IndicatorVote('volume_1h', +1, INDICATOR_WEIGHTS['volume_1h'],
                    f'Volume spike x{vol_1h.ratio:.1f} on bullish candle'))
            elif vol_1h.direction == 'bearish':
                votes.append(IndicatorVote('volume_1h', -1, INDICATOR_WEIGHTS['volume_1h'],
                    f'Volume spike x{vol_1h.ratio:.1f} on bearish candle'))
            else:
                votes.append(IndicatorVote('volume_1h', 0, INDICATOR_WEIGHTS['volume_1h'],
                    f'Volume spike x{vol_1h.ratio:.1f} — doji/neutral candle'))
        else:
            votes.append(IndicatorVote('volume_1h', 0, INDICATOR_WEIGHTS['volume_1h'],
                f'Normal volume (x{vol_1h.ratio:.1f} avg)'))

    # ── 10. EMA Cross — 1M ────────────────────────────────────
    ec_1m = ema_cross(df_1m) if not df_1m.empty else None
    if ec_1m:
        if ec_1m.cross == 'golden':
            votes.append(IndicatorVote('ema_cross_1m', +1, INDICATOR_WEIGHTS['ema_cross_1m'],
                'Golden cross 1M — short-term trigger'))
        elif ec_1m.cross == 'death':
            votes.append(IndicatorVote('ema_cross_1m', -1, INDICATOR_WEIGHTS['ema_cross_1m'],
                'Death cross 1M — short-term trigger'))
        else:
            votes.append(IndicatorVote('ema_cross_1m', 0, INDICATOR_WEIGHTS['ema_cross_1m'],
                'No 1M cross signal'))

    # ── 11. MACD — 1M ─────────────────────────────────────────
    mc_1m = macd(df_1m) if not df_1m.empty else None
    if mc_1m:
        if mc_1m.divergence in ('bullish_cross', 'bullish'):
            votes.append(IndicatorVote('macd_1m', +1, INDICATOR_WEIGHTS['macd_1m'],
                f'MACD 1M bullish ({mc_1m.divergence})'))
        elif mc_1m.divergence in ('bearish_cross', 'bearish'):
            votes.append(IndicatorVote('macd_1m', -1, INDICATOR_WEIGHTS['macd_1m'],
                f'MACD 1M bearish ({mc_1m.divergence})'))
        else:
            votes.append(IndicatorVote('macd_1m', 0, INDICATOR_WEIGHTS['macd_1m'],
                'MACD 1M neutral'))

    # ══════════════════════════════════════════════════════════
    # SMC VOTE LAYER (new in v2)
    # ══════════════════════════════════════════════════════════

    # ── 12. Order Blocks — 1H ─────────────────────────────────
    obs_1h = order_blocks(df_1h, impulse_pct=1.2) if not df_1h.empty else []
    if obs_1h:
        # Find the nearest unmitigated OB
        unmitigated = [o for o in obs_1h if not o.mitigated]
        if unmitigated:
            nearest_ob = min(unmitigated, key=lambda x: x.distance_pct)
            if nearest_ob.distance_pct < 2.0:
                if nearest_ob.ob_type == 'bullish':
                    votes.append(IndicatorVote('smc_order_block', +1, INDICATOR_WEIGHTS['smc_order_block'],
                        f'Price at bullish OB ${nearest_ob.bottom:,.0f}–${nearest_ob.top:,.0f} '
                        f'({nearest_ob.distance_pct:.1f}% away)'))
                else:
                    votes.append(IndicatorVote('smc_order_block', -1, INDICATOR_WEIGHTS['smc_order_block'],
                        f'Price at bearish OB ${nearest_ob.bottom:,.0f}–${nearest_ob.top:,.0f} '
                        f'({nearest_ob.distance_pct:.1f}% away)'))
            else:
                votes.append(IndicatorVote('smc_order_block', 0, INDICATOR_WEIGHTS['smc_order_block'],
                    f'Nearest OB {nearest_ob.distance_pct:.1f}% away — not in zone'))
        else:
            votes.append(IndicatorVote('smc_order_block', 0, INDICATOR_WEIGHTS['smc_order_block'],
                'All nearby OBs mitigated'))

    # ── 13. Fair Value Gaps — 1H ──────────────────────────────
    fvgs_1h = fair_value_gaps(df_1h, lookback=40) if not df_1h.empty else []
    if fvgs_1h:
        nearest_fvg = min(fvgs_1h, key=lambda x: x.distance_pct)
        if nearest_fvg.distance_pct < 1.5:
            if nearest_fvg.fvg_type == 'bullish':
                # Bullish FVG below current price = support/magnet from below
                votes.append(IndicatorVote('smc_fvg', +1, INDICATOR_WEIGHTS['smc_fvg'],
                    f'Bullish FVG below at ${nearest_fvg.bottom:,.0f}–${nearest_fvg.top:,.0f} '
                    f'(fill magnet, {nearest_fvg.distance_pct:.1f}% away)'))
            else:
                # Bearish FVG above current price = resistance/magnet from above
                votes.append(IndicatorVote('smc_fvg', -1, INDICATOR_WEIGHTS['smc_fvg'],
                    f'Bearish FVG above at ${nearest_fvg.bottom:,.0f}–${nearest_fvg.top:,.0f} '
                    f'(fill magnet, {nearest_fvg.distance_pct:.1f}% away)'))
        else:
            votes.append(IndicatorVote('smc_fvg', 0, INDICATOR_WEIGHTS['smc_fvg'],
                f'Nearest FVG {nearest_fvg.distance_pct:.1f}% away'))

    # ── 14. Break of Structure — 1H ───────────────────────────
    bos_1h = break_of_structure(df_1h, swing_lookback=20) if not df_1h.empty else None
    if bos_1h:
        if bos_1h.bars_ago <= 3:  # only act on fresh breaks
            if bos_1h.event in ('bos_bullish', 'choch_bullish'):
                label = 'BOS' if 'bos' in bos_1h.event else 'CHoCH'
                votes.append(IndicatorVote('smc_bos_choch', +1, INDICATOR_WEIGHTS['smc_bos_choch'],
                    f'{label} bullish 1H — broke ${bos_1h.broken_level:,.0f} '
                    f'({bos_1h.bars_ago} bars ago, +{bos_1h.strength:.2f}%)'))
            elif bos_1h.event in ('bos_bearish', 'choch_bearish'):
                label = 'BOS' if 'bos' in bos_1h.event else 'CHoCH'
                votes.append(IndicatorVote('smc_bos_choch', -1, INDICATOR_WEIGHTS['smc_bos_choch'],
                    f'{label} bearish 1H — broke ${bos_1h.broken_level:,.0f} '
                    f'({bos_1h.bars_ago} bars ago, -{bos_1h.strength:.2f}%)'))
        else:
            votes.append(IndicatorVote('smc_bos_choch', 0, INDICATOR_WEIGHTS['smc_bos_choch'],
                f'Last structure break was {bos_1h.bars_ago} bars ago — stale'))
    else:
        votes.append(IndicatorVote('smc_bos_choch', 0, INDICATOR_WEIGHTS['smc_bos_choch'],
            'No recent structure break detected'))

    # ── 15. Liquidity Sweep — 1H ──────────────────────────────
    sweep_1h = liquidity_sweep(df_1h, lookback=30) if not df_1h.empty else None
    if sweep_1h:
        if sweep_1h.bars_ago <= 2:  # sweep must be very recent
            if sweep_1h.sweep_type == 'bullish_sweep':
                votes.append(IndicatorVote('smc_liquidity', +1, INDICATOR_WEIGHTS['smc_liquidity'],
                    f'Bullish liquidity sweep — wicked below ${sweep_1h.swept_level:,.0f} '
                    f'and recovered ({sweep_1h.bars_ago} bars ago, wick={sweep_1h.wick_pct:.2f}%)'))
            else:
                votes.append(IndicatorVote('smc_liquidity', -1, INDICATOR_WEIGHTS['smc_liquidity'],
                    f'Bearish liquidity sweep — wicked above ${sweep_1h.swept_level:,.0f} '
                    f'and rejected ({sweep_1h.bars_ago} bars ago, wick={sweep_1h.wick_pct:.2f}%)'))
        else:
            votes.append(IndicatorVote('smc_liquidity', 0, INDICATOR_WEIGHTS['smc_liquidity'],
                f'Liquidity sweep {sweep_1h.bars_ago} bars ago — too old to act on'))
    else:
        votes.append(IndicatorVote('smc_liquidity', 0, INDICATOR_WEIGHTS['smc_liquidity'],
            'No recent liquidity sweep'))

    # ── Aggregate ─────────────────────────────────────────────
    weighted_bull = sum(v.vote * v.weight for v in votes if v.vote > 0)
    weighted_bear = sum(abs(v.vote) * v.weight for v in votes if v.vote < 0)
    total_w       = sum(v.weight for v in votes) or 1.0

    bull_score = weighted_bull / total_w
    bear_score = weighted_bear / total_w
    net_score  = (weighted_bull - weighted_bear) / total_w

    atr_res = atr(df_1h) if not df_1h.empty else None
    atr_pct = atr_res.pct if atr_res else 2.0

    trend_dir = ts_1d.direction if ts_1d else 'ranging'

    # Top 5 non-neutral reasons for the summary
    summary_votes = sorted([v for v in votes if v.vote != 0],
                            key=lambda x: x.weight, reverse=True)
    summary = ' | '.join(v.reason for v in summary_votes[:5])

    logger.info(
        f"{ticker} @ ${price:,.2f} | net={net_score:+.3f} "
        f"bull={bull_score:.2f} bear={bear_score:.2f} "
        f"atr={atr_pct:.1f}% trend={trend_dir}"
    )

    return AnalysisResult(
        ticker=ticker,
        price=price,
        bull_score=bull_score,
        bear_score=bear_score,
        net_score=net_score,
        votes=votes,
        atr_pct=atr_pct,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        trend=trend_dir,
        summary=summary,
    )