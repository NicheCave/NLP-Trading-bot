# backtest/feature_stream.py — precomputed, parameter-free per-bar features.
# ─────────────────────────────────────────────────────────────
# Walk-forward tuning needs to score hundreds of parameter sets. Recomputing
# the whole indicator stack each time would cost ~45 minutes per set, which
# makes honest out-of-sample tuning impossible in practice — and that is
# exactly how hand-tuned, overfitted configs like the v6 one come about.
#
# The split that makes it tractable: almost everything expensive is
# parameter-FREE. Indicator votes, block scores, regime labels, ATR and
# funding z-scores depend only on price history. What the tunable
# parameters actually change is how those numbers are WEIGHTED, GATED and
# turned into stops — all cheap arithmetic.
#
# So: compute the expensive part once into a table, then score parameter
# sets against the table in milliseconds.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

import config
from data import cache

from .legacy_strategy import btc_trend_series

# Block columns stored per bar. Scores are the raw within-family means,
# BEFORE any regime multiplier — the multiplier is a tunable parameter.
BLOCK_NAMES = ['trend', 'momentum', 'mean_reversion', 'structure',
               'volume', 'reversal', 'context']

WINDOW = config.ANALYSIS_WINDOW_BARS


def _stream_worker(args: tuple) -> pd.DataFrame:
    """Compute the feature row for a set of bar positions."""
    symbol, df_1h, df_1d, btc_trend, positions, funding, open_interest = args

    logger.remove()

    from features.blocks import FACTOR_BLOCKS
    from features.derivatives import analyse_derivatives
    from features.regime import classify_regime
    from indicators import atr as calc_atr
    from market_analyzer import analyze_frames

    daily_index = df_1d.index if df_1d is not None and not df_1d.empty else None
    rows, index = [], []

    for i in positions:
        ts = df_1h.index[i]
        hist_1h = df_1h.iloc[max(0, i - WINDOW + 1): i + 1]

        if daily_index is not None:
            cutoff = ts - pd.Timedelta(days=1)
            hist_1d = df_1d.iloc[:daily_index.searchsorted(cutoff, side='right')]
        else:
            hist_1d = pd.DataFrame()

        anchor = None if config.is_anchor(symbol) else (
            btc_trend.get(ts) if btc_trend is not None else None)

        row = {'close': float(hist_1h['close'].iloc[-1]),
               'btc_trend': anchor or 'unknown'}

        try:
            regime = classify_regime(hist_1h)
            row['regime'] = regime.regime.value
            row['regime_tradeable'] = regime.tradeable
            row['adx'] = regime.adx
            row['efficiency'] = regime.efficiency
            row['vol_pct'] = regime.vol_percentile

            analysis = analyze_frames(symbol, hist_1h, hist_1d, anchor)
            if analysis is None:
                raise ValueError('no analysis')

            # Raw within-family aggregation — no regime weighting yet.
            grouped: dict[str, list] = {}
            for vote in analysis.votes:
                block = FACTOR_BLOCKS.get(vote.name)
                if block:
                    grouped.setdefault(block, []).append(vote)

            for name in BLOCK_NAMES:
                votes = grouped.get(name, [])
                weight = sum(v.weight for v in votes)
                row[f'{name}_score'] = (
                    sum(v.vote * v.weight for v in votes) / weight if weight > 0 else 0.0)
                row[f'{name}_n'] = sum(1 for v in votes if v.vote != 0)

            atr_result = calc_atr(hist_1h, period=config.ATR_PERIOD)
            row['atr'] = atr_result.value if atr_result else np.nan

            hist_funding = funding.loc[:ts] if funding is not None else None
            hist_oi = open_interest.loc[:ts] if open_interest is not None else None
            derivatives = analyse_derivatives(hist_funding, hist_oi, hist_1h['close'])
            row['funding_z'] = derivatives.funding_z
            row['funding_rate'] = derivatives.funding_rate
            row['oi_bias'] = derivatives.oi_bias
            row['deriv_available'] = derivatives.available

        except Exception:
            row.setdefault('regime', 'unknown')
            row.setdefault('regime_tradeable', False)
            for name in BLOCK_NAMES:
                row.setdefault(f'{name}_score', 0.0)
                row.setdefault(f'{name}_n', 0)
            row.setdefault('atr', np.nan)
            row.setdefault('funding_z', 0.0)
            row.setdefault('funding_rate', 0.0)
            row.setdefault('oi_bias', 0)
            row.setdefault('deriv_available', False)
            row.setdefault('adx', 0.0)
            row.setdefault('efficiency', 0.0)
            row.setdefault('vol_pct', 0.5)

        rows.append(row)
        index.append(ts)

    return pd.DataFrame(rows, index=pd.DatetimeIndex(index, name='ts'))


def build_feature_stream(symbols: list[str],
                         data: dict[str, dict[str, pd.DataFrame]],
                         warmup: int = 300,
                         use_cache: bool = True,
                         workers: Optional[int] = None) -> dict[str, pd.DataFrame]:
    """Compute (or load) the per-bar feature table for every symbol."""
    anchor_1h = data.get(config.ANCHOR_SYMBOL, {}).get('1h')
    btc_trend = (btc_trend_series(anchor_1h)
                 if anchor_1h is not None and not anchor_1h.empty else None)

    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    out: dict[str, pd.DataFrame] = {}

    from features.derivatives import align_funding_to_bars, align_oi_to_bars

    for symbol in symbols:
        path = cache.cache_path('features', symbol, 'v7')
        df_1h = data[symbol]['1h']
        df_1d = data[symbol].get('1d', pd.DataFrame())
        n = len(df_1h)

        cached = cache.load(path) if use_cache else None
        have = set(cached.index) if cached is not None else set()
        todo = [i for i in range(warmup, n) if df_1h.index[i] not in have]

        if not todo:
            logger.info(f"features {symbol}: {len(cached)} bars (cached)")
            out[symbol] = cached
            continue

        funding = align_funding_to_bars(data[symbol].get('funding'), df_1h.index)
        open_interest = align_oi_to_bars(data[symbol].get('oi'), df_1h.index)

        logger.info(f"features {symbol}: computing {len(todo)} bars across {workers} workers…")
        buckets: list[list[int]] = [[] for _ in range(workers)]
        for k, i in enumerate(todo):
            buckets[k % workers].append(i)
        tasks = [(symbol, df_1h, df_1d, btc_trend, bucket, funding, open_interest)
                 for bucket in buckets if bucket]

        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_stream_worker, tasks))

        fresh = pd.concat(parts).sort_index()
        df = fresh if cached is None else pd.concat([cached, fresh]).sort_index()
        df = df[~df.index.duplicated(keep='last')]
        out[symbol] = df

        if use_cache:
            cache.save(df, path)
        logger.info(f"features {symbol}: {len(df)} bars ready")

    return out


# ══════════════════════════════════════════════════════════════
# SCORING — cheap, runs per parameter set
# ══════════════════════════════════════════════════════════════

def intents_from_features(features: pd.DataFrame, params, symbol: str) -> pd.DataFrame:
    """
    Apply a parameter set to a precomputed feature table.

    Vectorised where it matters, so a full 2-year, 3-symbol scoring pass
    takes well under a second — which is what makes a real walk-forward
    search possible instead of a hand-tuned guess.
    """
    from features.blocks import BLOCK_BASE_WEIGHTS
    from features.regime import REGIME_BLOCK_WEIGHTS, Regime

    n = len(features)
    net = np.zeros(n)
    total_weight = np.zeros(n)
    agree_weight = np.zeros(n)
    active_weight = np.zeros(n)
    active_blocks = np.zeros(n, dtype=int)

    regimes = features['regime'].to_numpy()

    # Per-regime multiplier lookup, applied per block.
    for block in BLOCK_NAMES:
        score = features[f'{block}_score'].to_numpy(dtype=float)
        counts = features[f'{block}_n'].to_numpy(dtype=float)
        base = BLOCK_BASE_WEIGHTS.get(block, 1.0)

        multiplier = np.ones(n)
        for regime in Regime:
            mask = regimes == regime.value
            if mask.any():
                multiplier[mask] = REGIME_BLOCK_WEIGHTS[regime].get(block, 1.0)

        weight = base * multiplier
        net += score * weight
        total_weight += weight
        has_votes = counts > 0
        active_weight += np.where(has_votes, weight, 0.0)
        active_blocks += has_votes.astype(int)

    # Derivatives block, when data exists for that bar.
    deriv_score = np.zeros(n)
    if 'funding_z' in features:
        funding_z = features['funding_z'].to_numpy(dtype=float)
        oi_bias = features['oi_bias'].to_numpy(dtype=float)
        available = features['deriv_available'].to_numpy(dtype=bool)
        extreme = np.abs(funding_z) >= params.funding_extreme_z

        # use_funding_gate governs the funding CONTRIBUTION as well as the
        # crowding veto further down — mirroring strategy_v7._derivatives_block.
        # Applying it unconditionally here made the tuner score a different
        # strategy from the one that trades whenever the flag was off.
        if params.use_funding_gate:
            deriv_score = np.where(available & extreme,
                                   -0.6 * np.sign(funding_z), 0.0)
        deriv_score = deriv_score + np.where(available, 0.4 * oi_bias, 0.0)
        deriv_score = np.clip(deriv_score, -1.0, 1.0)

        weight = BLOCK_BASE_WEIGHTS.get('derivatives', 0.9)
        has = deriv_score != 0
        net += deriv_score * weight * has
        total_weight += weight * has
        active_weight += weight * has
        active_blocks += has.astype(int)

    net_score = np.divide(net, total_weight, out=np.zeros(n), where=total_weight > 0)
    direction_sign = np.sign(net_score)

    # Agreement: share of ACTIVE block weight pointing the signal's way.
    for block in BLOCK_NAMES:
        score = features[f'{block}_score'].to_numpy(dtype=float)
        counts = features[f'{block}_n'].to_numpy(dtype=float)
        base = BLOCK_BASE_WEIGHTS.get(block, 1.0)
        multiplier = np.ones(n)
        for regime in Regime:
            mask = regimes == regime.value
            if mask.any():
                multiplier[mask] = REGIME_BLOCK_WEIGHTS[regime].get(block, 1.0)
        weight = base * multiplier
        agree_weight += np.where((counts > 0) & (score * direction_sign > 0), weight, 0.0)
    agree_weight += np.where(deriv_score * direction_sign > 0,
                             BLOCK_BASE_WEIGHTS.get('derivatives', 0.9), 0.0)

    agreement = np.divide(agree_weight, active_weight,
                          out=np.zeros(n), where=active_weight > 0)

    net_abs = np.abs(net_score)
    confidence = 1.0 / (1.0 + np.exp(-(params.conf_edge_weight * net_abs
                                       + params.conf_agree_weight * (agreement - 0.5)
                                       - params.conf_bias)))

    # ── gates ─────────────────────────────────────────────────
    tradeable = features['regime_tradeable'].to_numpy(dtype=bool)
    allowed = tradeable & (net_abs >= params.min_net)
    allowed &= agreement >= params.min_agreement
    allowed &= active_blocks >= params.min_active_blocks
    allowed &= confidence >= params.confidence_floor

    if not params.trade_squeeze:
        allowed &= regimes != Regime.SQUEEZE.value

    is_long = direction_sign > 0
    # Counter-trend entries need a steeper bar.
    counter = ((regimes == Regime.TREND_UP.value) & ~is_long) | \
              ((regimes == Regime.TREND_DOWN.value) & is_long)
    allowed &= ~counter | (net_abs >= params.counter_trend_min_net)

    # Symmetric BTC anchor.
    if params.use_btc_anchor and not config.is_anchor(symbol):
        btc = features['btc_trend'].to_numpy()
        allowed &= ~(is_long & (btc == 'downtrend'))
        allowed &= ~(~is_long & (btc == 'uptrend'))

    # Session windows, when enabled (see V7Params.use_kill_zones).
    if getattr(params, 'use_kill_zones', False):
        hours = features.index.hour.to_numpy()
        in_zone = np.zeros(n, dtype=bool)
        for _, start, end in config.KILL_ZONES:
            in_zone |= (hours >= start) & (hours < end)
        allowed &= in_zone

    # Crowded funding against the trade, outside a trend.
    if params.use_funding_gate and 'funding_z' in features:
        funding_z = features['funding_z'].to_numpy(dtype=float)
        crowded = (np.abs(funding_z) >= params.funding_extreme_z) & \
                  ((is_long & (funding_z > 0)) | (~is_long & (funding_z < 0)))
        in_trend = np.isin(regimes, [Regime.TREND_UP.value, Regime.TREND_DOWN.value])
        allowed &= ~(crowded & ~in_trend)

    # ── stops ─────────────────────────────────────────────────
    close = features['close'].to_numpy(dtype=float)
    atr = features['atr'].to_numpy(dtype=float)
    allowed &= np.isfinite(atr) & (atr > 0)

    distance = np.clip(atr * params.atr_stop_mult,
                       close * config.MIN_STOP_PCT,
                       close * config.MAX_STOP_PCT)
    reward = distance * (params.atr_target_mult / params.atr_stop_mult)

    stop = np.where(is_long, close - distance, close + distance)
    take = np.where(is_long, close + reward, close - reward)

    return pd.DataFrame({
        'direction': np.where(allowed, np.where(is_long, 'long', 'short'), 'flat'),
        'confidence': np.where(allowed, confidence, 0.0),
        'stop_price': np.where(allowed, stop, np.nan),
        'take_price': np.where(allowed, take, np.nan),
        'atr': np.where(allowed, atr, np.nan),
        'reason': '',
        'tags': [[] for _ in range(n)],
    }, index=features.index)
