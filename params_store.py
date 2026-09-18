# params_store.py — persist walk-forward-selected strategy parameters.
# ─────────────────────────────────────────────────────────────
# Closes the loop between tuning and trading. Without this the walk-forward
# result is just a number in a log, and the live bot keeps running the
# hand-picked defaults — which is precisely how the v6 config drifted away
# from anything measured.
#
# The saved values are the CONSENSUS across walk-forward windows (median for
# numeric knobs, majority for flags), not the single best-scoring window.
# One window's winner is mostly luck; a setting that keeps getting picked
# across independent periods is far likelier to be real.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import os
import statistics
from dataclasses import asdict, fields, replace
from typing import Optional

from loguru import logger

PARAMS_PATH = "tuned_params.json"


def consensus(selected: list, base) -> tuple:
    """
    Collapse per-window parameter choices into one set.

    Returns (params, stability) where stability maps each field to the
    fraction of windows that agreed with the consensus value — a direct read
    on how much to trust each knob.
    """
    if not selected:
        return base, {}

    values: dict[str, list] = {}
    for params in selected:
        for field in fields(params):
            values.setdefault(field.name, []).append(getattr(params, field.name))

    chosen, stability = {}, {}
    for name, series in values.items():
        if all(isinstance(v, bool) for v in series):
            pick = sum(series) > len(series) / 2
        else:
            # Pick the most frequently selected value, breaking ties toward
            # the one nearest the median.
            #
            # It MUST come from the observed set. Taking a plain median of an
            # even number of grid choices interpolates between them — an
            # earlier version produced atr_stop_mult=2.65 from a grid of
            # {1.8, 2.5, 3.5}, a configuration the search had never scored.
            # Saving that would have put an untested config into live
            # trading, which is exactly what the tuning exists to prevent.
            counts = {v: series.count(v) for v in set(series)}
            best = max(counts.values())
            tied = [v for v, c in counts.items() if c == best]
            if len(tied) == 1:
                pick = tied[0]
            elif all(isinstance(v, (int, float)) for v in tied):
                midpoint = statistics.median(series)
                pick = min(tied, key=lambda v: abs(v - midpoint))
            else:
                pick = tied[0]

        chosen[name] = pick
        stability[name] = sum(1 for v in series if v == pick) / len(series)

    return replace(base, **chosen), stability


def save(params, metadata: Optional[dict] = None, path: str = PARAMS_PATH) -> None:
    try:
        with open(path, 'w') as f:
            json.dump({'params': asdict(params), 'metadata': metadata or {}}, f, indent=2)
        logger.info(f"Saved tuned parameters to {path}")
    except Exception as e:
        logger.warning(f"Could not save tuned parameters: {e}")


def load(base, path: str = PARAMS_PATH):
    """
    Load tuned parameters, falling back to `base` when absent.

    Unknown keys are ignored so an older file can't break a newer strategy.
    """
    if not os.path.exists(path):
        return base, {}
    try:
        with open(path) as f:
            data = json.load(f)
        known = {f.name for f in fields(base)}
        values = {k: v for k, v in data.get('params', {}).items() if k in known}
        logger.info(f"Loaded tuned parameters from {path} "
                    f"({len(values)} fields)")
        return replace(base, **values), data.get('metadata', {})
    except Exception as e:
        logger.warning(f"Could not load tuned parameters: {e}")
        return base, {}
