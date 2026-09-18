# backtest/walkforward.py — honest out-of-sample parameter fitting.
# ─────────────────────────────────────────────────────────────
# Fitting parameters on all the data and reporting the result is how the v6
# config came to exist: a chain of hand-tuned constants ("v6: raised from
# 1.5") each of which improved the number the author was looking at, with
# no way to tell signal from noise.
#
# Walk-forward is the standard defence:
#
#   |<--- in-sample --->|<- OOS ->|
#            |<--- in-sample --->|<- OOS ->|
#                     |<--- in-sample --->|<- OOS ->|
#
# Parameters are chosen on the in-sample window ONLY, then scored on data
# the search never saw. The concatenated out-of-sample results are the
# honest estimate — usually far worse than the in-sample fit, and that gap
# is the single most useful diagnostic available.
#
# A parameter set that looks good in-sample and collapses out-of-sample was
# fitted to noise. Reporting both makes that impossible to miss.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import random
from dataclasses import replace
from typing import Iterable, Optional

import pandas as pd
from loguru import logger

import config

from .engine import BacktestConfig, BacktestEngine, BacktestResult
from .feature_stream import intents_from_features
from .metrics import Metrics, compute_metrics

# Search space. Deliberately coarse — a fine grid over a two-year sample
# finds precision that does not exist.
PARAM_GRID: dict[str, list] = {
    # The first full-sample v7 run fired on 36% of bars and took 3,015
    # trades in two years — its fee bill alone ($10.4k) exceeded the
    # starting equity. Trading cost is the binding constraint here, so the
    # search space is weighted heavily toward SELECTIVITY.
    'min_net':               [0.20, 0.28, 0.35, 0.42, 0.50],
    'min_agreement':         [0.60, 0.70, 0.80, 0.90],
    'counter_trend_min_net': [0.35, 0.45, 0.60],
    'min_active_blocks':     [3, 4, 5],
    # Wider stops mean fewer stop-outs and longer holds, which spreads the
    # fixed round-trip cost over a larger move.
    'atr_stop_mult':         [1.8, 2.5, 3.5],
    'atr_target_mult':       [3.6, 5.0, 7.0, 10.0],
    'conf_bias':             [1.5, 1.9, 2.3],
    'use_funding_gate':      [True, False],
    'trade_squeeze':         [False],
    # Measured on the baseline: in-zone +0.039R vs out-of-zone −0.036R.
    # Suggestive but ~1 SE, so it is searched rather than assumed.
    'use_kill_zones':        [False, True],
}


def sample_params(base, grid: dict, n: int, seed: int = 42) -> list:
    """Random search over the grid — more efficient than exhaustive here."""
    rng = random.Random(seed)
    seen, out = set(), []
    for _ in range(n * 4):
        if len(out) >= n:
            break
        choice = {k: rng.choice(v) for k, v in grid.items()}
        # Keep the payoff ratio sane: a target inside the stop is nonsense.
        if choice['atr_target_mult'] <= choice['atr_stop_mult']:
            continue
        key = tuple(sorted(choice.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(replace(base, **choice))
    return out


def _score(data: dict,
           features: dict[str, pd.DataFrame],
           symbols: list[str],
           params,
           cfg: BacktestConfig,
           start: pd.Timestamp,
           end: pd.Timestamp,
           warmup: int) -> tuple[Metrics, BacktestResult]:
    """Run one parameter set over one date window."""
    intents, sliced = {}, {}
    for symbol in symbols:
        feature_df = features[symbol]
        window = feature_df[(feature_df.index >= start) & (feature_df.index < end)]
        if window.empty:
            continue
        intents[symbol] = intents_from_features(window, params, symbol)

        bars = data[symbol]['1h']
        sliced[symbol] = {
            '1h': bars[(bars.index >= start) & (bars.index < end)],
            '1d': data[symbol].get('1d', pd.DataFrame()),
            'funding': data[symbol].get('funding', pd.DataFrame()),
        }

    if not sliced:
        return Metrics(), BacktestResult([], pd.Series(dtype=float), cfg)

    result = BacktestEngine(cfg).run(sliced, intents=intents, warmup=warmup)
    return compute_metrics(result), result


def walk_forward(data: dict,
                 features: dict[str, pd.DataFrame],
                 symbols: list[str],
                 base_params,
                 cfg: BacktestConfig,
                 is_days: int = 180,
                 oos_days: int = 60,
                 n_samples: int = 60,
                 min_is_trades: int = 25,
                 objective: str = 'expectancy_r',
                 seed: int = 42) -> dict:
    """
    Roll an in-sample fit forward over out-of-sample windows.

    Args:
        objective:      metric maximised in-sample ('expectancy_r' or 'profit_factor')
        min_is_trades:  reject in-sample fits with too few trades to be meaningful —
                        the fastest route to overfitting is a parameter set that
                        takes four trades and wins all of them.

    Returns a dict with the concatenated out-of-sample metrics, the per-window
    detail, and the in-sample/out-of-sample gap.
    """
    candidates = sample_params(base_params, PARAM_GRID, n_samples, seed)
    logger.info(f"Walk-forward: {len(candidates)} parameter sets, "
                f"{is_days}d in-sample → {oos_days}d out-of-sample")

    # Common timeline across symbols.
    starts = [features[s].index[0] for s in symbols if len(features.get(s, []))]
    ends = [features[s].index[-1] for s in symbols if len(features.get(s, []))]
    if not starts:
        raise ValueError("no feature data")

    timeline_start, timeline_end = min(starts), max(ends)

    windows = []
    cursor = timeline_start
    while cursor + pd.Timedelta(days=is_days + oos_days) <= timeline_end:
        is_start = cursor
        is_end = cursor + pd.Timedelta(days=is_days)
        oos_end = is_end + pd.Timedelta(days=oos_days)
        windows.append((is_start, is_end, oos_end))
        cursor = cursor + pd.Timedelta(days=oos_days)

    logger.info(f"Walk-forward: {len(windows)} windows "
                f"({timeline_start.date()} → {timeline_end.date()})")

    all_oos_trades = []
    detail = []

    for k, (is_start, is_end, oos_end) in enumerate(windows, 1):
        best, best_score, best_is = None, -1e9, None

        for params in candidates:
            metrics, _ = _score(data, features, symbols, params, cfg,
                                is_start, is_end, warmup=0)
            if metrics.n_trades < min_is_trades:
                continue
            score = getattr(metrics, objective)
            if score > best_score:
                best, best_score, best_is = params, score, metrics

        if best is None:
            logger.warning(f"  window {k}: no parameter set cleared "
                           f"{min_is_trades} in-sample trades — skipping")
            continue

        oos_metrics, oos_result = _score(data, features, symbols, best, cfg,
                                         is_end, oos_end, warmup=0)
        all_oos_trades.extend(oos_result.trades)

        detail.append({
            'window': k,
            'is_start': is_start, 'is_end': is_end, 'oos_end': oos_end,
            'params': best,
            'is_trades': best_is.n_trades,
            'is_expectancy': best_is.expectancy_r,
            'is_win_rate': best_is.win_rate,
            'oos_trades': oos_metrics.n_trades,
            'oos_expectancy': oos_metrics.expectancy_r,
            'oos_win_rate': oos_metrics.win_rate,
        })

        logger.info(
            f"  window {k}/{len(windows)} {is_end.date()}→{oos_end.date()}: "
            f"IS {best_is.expectancy_r:+.3f}R ({best_is.n_trades}t) → "
            f"OOS {oos_metrics.expectancy_r:+.3f}R ({oos_metrics.n_trades}t)")

    combined = BacktestResult(trades=all_oos_trades,
                              equity_curve=_rebuild_curve(all_oos_trades, cfg),
                              config=cfg)
    oos_metrics = compute_metrics(combined)

    is_mean = (sum(d['is_expectancy'] for d in detail) / len(detail)) if detail else 0.0
    oos_mean = (sum(d['oos_expectancy'] for d in detail) / len(detail)) if detail else 0.0

    return {
        'oos_metrics': oos_metrics,
        'windows': detail,
        'is_mean_expectancy': is_mean,
        'oos_mean_expectancy': oos_mean,
        'overfit_gap': is_mean - oos_mean,
        'candidates': len(candidates),
    }


def _rebuild_curve(trades, cfg: BacktestConfig) -> pd.Series:
    """
    Stitch out-of-sample trades into one equity curve.

    Approximate by construction — the windows are scored independently, so
    this is for drawdown shape, not a precise account statement.
    """
    if not trades:
        return pd.Series(dtype=float)

    ordered = sorted(trades, key=lambda t: t.exit_ts)
    equity = cfg.initial_equity
    points = {}
    for trade in ordered:
        equity += trade.net_pnl
        points[trade.exit_ts] = equity
    return pd.Series(points).sort_index()


def format_walk_forward(report: dict) -> str:
    """Render the walk-forward summary."""
    metrics = report['oos_metrics']
    lines = [
        "",
        "═" * 66,
        "  WALK-FORWARD — OUT-OF-SAMPLE ONLY",
        "═" * 66,
        f"  Windows                {len(report['windows'])}",
        f"  Parameter sets tried   {report['candidates']}",
        f"  OOS trades             {metrics.n_trades}",
        f"  OOS win rate           {metrics.win_rate:.1f}%  "
        f"(break-even {metrics.breakeven_wr:.1f}%)",
        f"  OOS expectancy         {metrics.expectancy_r:+.3f} ± "
        f"{metrics.expectancy_se:.3f} R/trade  (t = {metrics.expectancy_t:+.2f})",
        f"  Distinguishable from 0? "
        f"{'yes' if abs(metrics.expectancy_t) >= 2 else 'NO — within noise'}",
        f"  OOS profit factor      {metrics.profit_factor:.2f}",
        f"  OOS max drawdown       {metrics.max_drawdown:.1f}%",
        "─" * 66,
        f"  Mean in-sample  expectancy   {report['is_mean_expectancy']:+.3f} R",
        f"  Mean out-of-sample expectancy {report['oos_mean_expectancy']:+.3f} R",
        f"  Overfit gap                   {report['overfit_gap']:+.3f} R",
        "",
        "  A large positive gap means the search fitted noise: the in-sample",
        "  numbers were bought by parameters that did not generalise.",
        "═" * 66,
    ]
    return "\n".join(lines)
