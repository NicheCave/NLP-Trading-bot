# backtest/signals.py — precompute and cache the per-bar intent stream.
# ─────────────────────────────────────────────────────────────
# Signal generation is by far the slowest part of a replay (~54 ms/bar for
# the legacy stack — roughly 45 minutes for 3 symbols over 2 years), while
# the portfolio simulation itself takes seconds.
#
# So the two are separated: generate intents ONCE, cache them to parquet,
# then re-run the engine instantly across as many risk settings as we like.
# Without this split, walk-forward tuning in Phase 5 would be unusable.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import pandas as pd
from loguru import logger

import config
from data import cache

from .legacy_strategy import btc_trend_series, legacy_intent

# Bars of history handed to the indicator stack at each step. Must match the
# live path (config.ANALYSIS_WINDOW_BARS) or the backtest measures a
# different strategy from the one that actually trades.
WINDOW = config.ANALYSIS_WINDOW_BARS

INTENT_COLUMNS = ['direction', 'confidence', 'stop_price', 'take_price', 'atr', 'reason', 'tags']


def _worker(args: tuple) -> pd.DataFrame:
    """
    Compute intents for one contiguous slice of a symbol's timeline.

    Module-level (not a closure) because Windows spawns fresh interpreters
    for process pools and can only pickle importable callables.
    """
    (symbol, df_1h, df_1d, btc_trend, positions, spec) = args

    # The strategy logs a line per bar at INFO. Across ~17.5k bars × N
    # workers that is both unreadable and a real slowdown, so worker
    # processes stay silent and only the parent reports progress.
    logger.remove()

    strategy = spec.get('strategy', 'legacy')
    funding = spec.get('funding')
    open_interest = spec.get('open_interest')
    params = spec.get('params')

    if strategy == 'v7':
        from strategy_v7 import v7_intent

    rows: list[dict] = []
    index: list[pd.Timestamp] = []
    daily_index = df_1d.index if df_1d is not None and not df_1d.empty else None

    for i in positions:
        ts = df_1h.index[i]
        hist_1h = df_1h.iloc[max(0, i - WINDOW + 1): i + 1]

        # Only daily bars that had CLOSED by this moment. A daily bar opening
        # at date D is not complete until D+1.
        if daily_index is not None:
            cutoff = ts - pd.Timedelta(days=1)
            hist_1d = df_1d.iloc[:daily_index.searchsorted(cutoff, side='right')]
        else:
            hist_1d = pd.DataFrame()

        anchor = None if config.is_anchor(symbol) else (
            btc_trend.get(ts) if btc_trend is not None else None
        )

        try:
            if strategy == 'v7':
                # Slice derivatives series to what had already PRINTED at ts.
                hist_funding = funding.loc[:ts] if funding is not None else None
                hist_oi = open_interest.loc[:ts] if open_interest is not None else None
                intent = v7_intent(symbol, hist_1h, hist_1d, ts, anchor,
                                   funding=hist_funding, open_interest=hist_oi,
                                   params=params)
            else:
                intent = legacy_intent(symbol, hist_1h, hist_1d, ts, anchor,
                                       allow_shorts=spec.get('allow_shorts', False),
                                       stop_pct=spec.get('stop_pct', 0.05),
                                       take_pct=spec.get('take_pct', 0.07))
        except Exception as e:
            logger.debug(f"intent error {symbol} @ {ts}: {e}")
            intent = None

        index.append(ts)
        if intent is None:
            rows.append({'direction': 'flat', 'confidence': 0.0, 'stop_price': None,
                         'take_price': None, 'atr': None, 'reason': '', 'tags': []})
        else:
            rows.append({
                'direction': intent.direction,
                'confidence': intent.confidence,
                'stop_price': intent.stop_price,
                'take_price': intent.take_price,
                'atr': intent.atr,
                'reason': intent.reason,
                'tags': intent.tags,
            })

    return pd.DataFrame(rows, index=pd.DatetimeIndex(index, name='ts'))


def build_intents(symbols: list[str],
                  data: dict[str, dict[str, pd.DataFrame]],
                  variant: str = 'legacy_long_only',
                  strategy: str = 'legacy',
                  allow_shorts: bool = False,
                  stop_pct: float = 0.05,
                  take_pct: float = 0.07,
                  params=None,
                  warmup: int = 300,
                  use_cache: bool = True,
                  workers: Optional[int] = None) -> dict[str, pd.DataFrame]:
    """
    Build (or load) the intent stream for every symbol.

    Args:
        variant:  cache key — change it whenever the strategy logic changes,
                  or you will silently score stale signals.
        strategy: 'legacy' (the v6 stack) or 'v7' (regime-routed blocks)
    """
    anchor_1h = data.get(config.ANCHOR_SYMBOL, {}).get('1h')
    btc_trend = btc_trend_series(anchor_1h) if anchor_1h is not None and not anchor_1h.empty else None

    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    out: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        path = cache.cache_path('intents', symbol, variant)
        df_1h = data[symbol]['1h']
        df_1d = data[symbol].get('1d', pd.DataFrame())
        n = len(df_1h)
        if n <= warmup:
            logger.warning(f"{symbol}: only {n} bars, need > {warmup}")
            continue

        cached = cache.load(path) if use_cache else None

        # A cache hit is only valid if it COVERS the bars being asked for.
        # Matching on the variant name alone means a short exploratory run
        # silently satisfies a later long one, and the backtest then scores
        # a fraction of the requested period without saying so.
        have = set(cached.index) if cached is not None else set()
        todo = [i for i in range(warmup, n) if df_1h.index[i] not in have]

        if not todo:
            logger.info(f"intents[{variant}] {symbol}: {len(cached)} bars (cached)")
            out[symbol] = cached
            continue

        if cached is not None:
            logger.info(f"intents[{variant}] {symbol}: cache covers "
                        f"{len(have)} bars, computing {len(todo)} more…")
        else:
            logger.info(f"intents[{variant}] {symbol}: computing {len(todo)} bars "
                        f"across {workers} workers…")

        # Round-robin the outstanding bars so every worker gets a similar
        # amount of work even when the gap is fragmented.
        # Derivatives series, forward-filled onto the bar index. Alignment
        # happens once here rather than per bar in every worker.
        spec = {
            'strategy': strategy,
            'allow_shorts': allow_shorts,
            'stop_pct': stop_pct,
            'take_pct': take_pct,
            'params': params,
        }
        if strategy == 'v7':
            from features.derivatives import align_funding_to_bars, align_oi_to_bars
            spec['funding'] = align_funding_to_bars(
                data[symbol].get('funding'), df_1h.index)
            spec['open_interest'] = align_oi_to_bars(
                data[symbol].get('oi'), df_1h.index)

        buckets: list[list[int]] = [[] for _ in range(workers)]
        for k, i in enumerate(todo):
            buckets[k % workers].append(i)
        tasks = [
            (symbol, df_1h, df_1d, btc_trend, bucket, spec)
            for bucket in buckets if bucket
        ]

        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_worker, tasks))

        fresh = pd.concat(parts).sort_index()
        df = fresh if cached is None else pd.concat([cached, fresh]).sort_index()
        df = df[~df.index.duplicated(keep='last')]
        out[symbol] = df

        if use_cache:
            # `tags` is a list column — parquet handles it, but only if the
            # empty case is a real list rather than None.
            df = df.copy()
            df['tags'] = df['tags'].apply(lambda v: list(v) if isinstance(v, (list, tuple)) else [])
            cache.save(df, path)

        actionable = int((df['direction'] != 'flat').sum())
        logger.info(f"intents[{variant}] {symbol}: {actionable}/{len(df)} actionable bars")

    return out
