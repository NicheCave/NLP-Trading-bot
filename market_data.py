# market_data.py — OHLCV access for the live bot (Bybit perpetuals).
# ─────────────────────────────────────────────────────────────
# Thin convenience layer over data/bybit_data.py, kept because the rest of
# the bot already speaks this interface. All the real work — pagination,
# caching, partial-bar handling — lives in the data package.
#
# Two behavioural changes from the Alpaca version:
#   - The still-forming bar is excluded. Indicators computed on a partial
#     candle flip between cycles and produce entries on levels that never
#     actually printed.
#   - The 1m timeframe is gone. It was fetched every cycle and never read.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from loguru import logger

import config
from data import get_klines, get_last_price

# Internal timeframe label → Bybit interval code.
TIMEFRAMES = {
    '1h': '60',
    '4h': '240',
    '1d': 'D',
}

# How much history each timeframe needs to warm up its slowest indicator.
DEFAULT_DAYS = {
    '1h': 30,     # ~720 bars
    '4h': 120,
    '1d': 400,
}


def fetch_bars(symbol: str, timeframe: str = '1h', days: float | None = None) -> pd.DataFrame:
    """
    Fetch closed OHLCV candles for one perpetual.

    Args:
        symbol:    Bybit symbol, e.g. 'BTCUSDT'
        timeframe: '1h' | '4h' | '1d'
        days:      history window; defaults to DEFAULT_DAYS[timeframe]

    Returns:
        DataFrame[open, high, low, close, volume, turnover] indexed UTC,
        ascending. Empty on failure — callers must handle that.
    """
    interval = TIMEFRAMES.get(timeframe)
    if interval is None:
        logger.warning(f"Unknown timeframe {timeframe!r}")
        return pd.DataFrame()

    try:
        return get_klines(symbol, interval, days=days or DEFAULT_DAYS[timeframe])
    except Exception as e:
        logger.warning(f"Bar fetch error ({symbol}, {timeframe}): {e}")
        return pd.DataFrame()


def fetch_multi_timeframe(symbol: str) -> dict[str, pd.DataFrame]:
    """
    Fetch 1H and 1D bars concurrently.

    Returns {'1h': df, '1d': df}; either may be empty on failure.
    """
    labels = ['1h', '1d']
    results: dict[str, pd.DataFrame] = {}

    with ThreadPoolExecutor(max_workers=len(labels)) as pool:
        futures = {pool.submit(fetch_bars, symbol, label): label for label in labels}
        for future, label in futures.items():
            try:
                results[label] = future.result()
            except Exception as e:
                logger.warning(f"fetch_multi_timeframe error ({symbol}, {label}): {e}")
                results[label] = pd.DataFrame()

    return results


def get_current_price(symbol: str) -> float | None:
    """Latest traded price, or None if unavailable."""
    return get_last_price(symbol)
