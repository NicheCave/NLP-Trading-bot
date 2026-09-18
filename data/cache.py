# data/cache.py — parquet-backed cache for market data.
# ─────────────────────────────────────────────────────────────
# Backtests re-read the same history hundreds of times during parameter
# search. Hitting Bybit every time is slow and rate-limited, so every
# fetch is persisted to cache/<kind>/<symbol>_<suffix>.parquet and
# extended incrementally rather than re-downloaded.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

CACHE_ROOT = Path(__file__).resolve().parent.parent / "cache"


def cache_path(kind: str, symbol: str, suffix: str = "") -> Path:
    """Build (and create) the on-disk path for one cached series."""
    directory = CACHE_ROOT / kind
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{symbol}_{suffix}.parquet" if suffix else f"{symbol}.parquet"
    return directory / name


def load(path: Path) -> Optional[pd.DataFrame]:
    """Read a cached frame, or None when it's absent/unreadable."""
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        return df
    except Exception as e:
        logger.warning(f"cache: could not read {path.name}: {e}")
        return None


def save(df: pd.DataFrame, path: Path) -> None:
    """Persist a frame, ignoring (but logging) any write failure."""
    if df is None or df.empty:
        return
    try:
        df.to_parquet(path)
    except Exception as e:
        logger.warning(f"cache: could not write {path.name}: {e}")


def merge(old: Optional[pd.DataFrame], new: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine cached and freshly-fetched frames.

    Both are indexed by UTC timestamp. Overlapping rows keep the NEW value —
    the final bar of a cached fetch may have been incomplete when it was
    written, so the fresh copy is always the more accurate one.
    """
    frames = [f for f in (old, new) if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0].sort_index()

    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep='last')]
    return combined.sort_index()
