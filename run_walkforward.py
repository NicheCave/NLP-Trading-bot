#!/usr/bin/env python
# run_walkforward.py — fit the v7 parameters out-of-sample.
# ─────────────────────────────────────────────────────────────
#   python run_walkforward.py                       # default 180d IS / 60d OOS
#   python run_walkforward.py --samples 120
#   python run_walkforward.py --is-days 240 --oos-days 90
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import sys

from loguru import logger

import config
import params_store
from backtest import BacktestConfig
from backtest.feature_stream import build_feature_stream
from backtest.walkforward import format_walk_forward, walk_forward
from run_backtest import load_data
from strategy_v7 import DEFAULT_PARAMS

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass

logger.remove()
logger.add(sys.stdout, colorize=True, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward tuning for the v7 strategy")
    parser.add_argument('--days', type=float, default=config.BACKTEST_DAYS)
    parser.add_argument('--symbols', default=','.join(config.SYMBOLS))
    parser.add_argument('--is-days', type=int, default=180)
    parser.add_argument('--oos-days', type=int, default=60)
    parser.add_argument('--samples', type=int, default=60)
    parser.add_argument('--objective', default='expectancy_r',
                        choices=['expectancy_r', 'profit_factor'])
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    data = load_data(symbols, args.days, with_oi=True)

    features = build_feature_stream(symbols, data, warmup=config.BACKTEST_WARMUP_BARS)

    cfg = BacktestConfig(
        initial_equity=config.BACKTEST_INITIAL_EQUITY,
        taker_fee=config.TAKER_FEE,
        slippage_bps=config.SLIPPAGE_BPS,
        risk_per_trade=config.RISK_PER_TRADE,
        max_positions=config.MAX_OPEN_POSITIONS,
        max_leverage=config.LEVERAGE,
        time_stop_bars=config.TIME_STOP_HOURS,
        trailing_atr_mult=config.TRAILING_ATR_MULTIPLIER,
        trailing_activate_r=config.TRAILING_ACTIVATE_R,
    )

    report = walk_forward(
        data=data, features=features, symbols=symbols,
        base_params=DEFAULT_PARAMS, cfg=cfg,
        is_days=args.is_days, oos_days=args.oos_days,
        n_samples=args.samples, objective=args.objective,
    )

    print(format_walk_forward(report))

    # Which settings were chosen most often across windows? A parameter that
    # keeps winning is more trustworthy than one that wins once by a nose.
    if report['windows']:
        print("\n  Parameters selected per window")
        print("  " + "─" * 62)
        for window in report['windows']:
            p = window['params']
            print(f"  {window['oos_end'].date()}  "
                  f"net≥{p.min_net:.2f} agree≥{p.min_agreement:.2f} "
                  f"blocks≥{p.min_active_blocks} "
                  f"stop={p.atr_stop_mult}×ATR target={p.atr_target_mult}×ATR "
                  f"→ OOS {window['oos_expectancy']:+.3f}R")

        tuned, stability = params_store.consensus(
            [w['params'] for w in report['windows']], DEFAULT_PARAMS)

        print("\n  Consensus across windows (stability = how often each was chosen)")
        print("  " + "─" * 62)
        for name in sorted(stability, key=lambda k: -stability[k]):
            if stability[name] < 0.99:      # unanimous knobs aren't interesting
                print(f"  {name:<24} {getattr(tuned, name)!s:>8}   "
                      f"{stability[name]:.0%} of windows")

        metrics = report['oos_metrics']
        if metrics.expectancy_r > 0 and metrics.n_trades >= 50:
            params_store.save(tuned, {
                'oos_expectancy_r': metrics.expectancy_r,
                'oos_expectancy_se': metrics.expectancy_se,
                'oos_expectancy_t': metrics.expectancy_t,
                'oos_win_rate': metrics.win_rate,
                'oos_trades': metrics.n_trades,
                'overfit_gap': report['overfit_gap'],
                'windows': len(report['windows']),
                'stability': stability,
            })
            print(f"\n  ✓ Saved to {params_store.PARAMS_PATH} — "
                  f"main.py will pick these up on next start.")

            if abs(metrics.expectancy_t) < 2:
                trades_needed = int((2 * metrics.expectancy_se / metrics.expectancy_r) ** 2
                                    * metrics.n_trades) if metrics.expectancy_r else 0
                print(f"\n  ⚠ But the edge is NOT statistically significant: "
                      f"{metrics.expectancy_r:+.3f} ± {metrics.expectancy_se:.3f} R "
                      f"(t = {metrics.expectancy_t:+.2f}).")
                print(f"    An edge this small needs roughly {trades_needed:,} trades "
                      f"to confirm — about "
                      f"{trades_needed / max(1, metrics.n_trades):.0f}× "
                      f"the sample available here.")
                print(f"    Treat it as a hypothesis to paper-trade, not a proven edge.")
        else:
            print(f"\n  ✗ NOT saved: out-of-sample expectancy "
                  f"{metrics.expectancy_r:+.3f}R over {metrics.n_trades} trades "
                  f"does not justify trading this.")
        print()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
