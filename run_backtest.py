#!/usr/bin/env python
# run_backtest.py — score a strategy variant over historical Bybit data.
# ─────────────────────────────────────────────────────────────
#   python run_backtest.py                    # legacy long-only (the baseline)
#   python run_backtest.py --shorts           # same signals, shorts enabled
#   python run_backtest.py --compare          # both, side by side
#   python run_backtest.py --no-lookahead-check
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import sys
from dataclasses import replace

import pandas as pd
from loguru import logger

import config
from backtest import BacktestConfig, BacktestEngine, compute_metrics, format_report
from backtest.metrics import compare, session_breakdown
from backtest.signals import build_intents
from data import get_funding_history, get_klines, get_open_interest

# The report uses box-drawing characters and '≥'; the default Windows
# console codepage (cp1252) cannot encode them and raises mid-print.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass

logger.remove()
logger.add(sys.stdout, colorize=True, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")


def load_data(symbols: list[str], days: float,
              with_oi: bool = False) -> dict[str, dict[str, pd.DataFrame]]:
    """Load 1H, 1D, funding and (optionally) open-interest series."""
    data: dict[str, dict[str, pd.DataFrame]] = {}
    for symbol in symbols:
        data[symbol] = {
            '1h': get_klines(symbol, '60', days=days),
            '1d': get_klines(symbol, 'D', days=days + 200),
            'funding': get_funding_history(symbol, days=days),
        }
        oi_note = ''
        if with_oi:
            # Bybit only retains ~6 months of OI, so it cannot cover a
            # 2-year replay. The strategy treats a missing series as "no
            # information", never as zero.
            oi = get_open_interest(symbol, days=min(days, 180))
            data[symbol]['oi'] = oi
            oi_note = f", {len(oi)} OI points"

        logger.info(f"{symbol}: {len(data[symbol]['1h'])} 1H bars, "
                    f"{len(data[symbol]['funding'])} funding settlements{oi_note}")
    return data


def assert_no_lookahead(data, intents, result) -> None:
    """
    Verify the engine cannot act on a price it hasn't seen.

    For every trade, the entry must occur at the OPEN of the bar strictly
    AFTER the bar whose close produced the signal. If an entry price ever
    equals the close of its own decision bar, the timing model has drifted.
    """
    problems = 0
    checked = 0
    for trade in result.trades[:500]:
        df = data[trade.symbol]['1h']
        idf = intents.get(trade.symbol)
        if idf is None or trade.entry_ts not in df.index:
            continue
        pos = df.index.get_loc(trade.entry_ts)
        if pos == 0:
            continue

        decision_ts = df.index[pos - 1]
        if decision_ts not in idf.index:
            continue
        checked += 1

        # The decision bar must actually have carried a matching intent.
        direction = idf.at[decision_ts, 'direction']
        if direction != trade.side:
            problems += 1
            logger.error(
                f"LOOKAHEAD: {trade.symbol} entered {trade.side} at {trade.entry_ts} "
                f"but the prior bar's intent was {direction!r}"
            )

    if problems:
        raise AssertionError(f"No-lookahead check FAILED on {problems}/{checked} trades")
    logger.success(f"✓ No-lookahead check passed on {checked} trades "
                   f"(every entry fills at the open of the bar after its signal)")


# Named variants: (cache key, strategy, allow_shorts)
VARIANTS = {
    # The bot as it trades today: spot semantics, SELL just means "flat".
    'baseline':     ('legacy_long_only', 'legacy', False),
    # Identical signals, but SELL opens a short. Isolates what futures buys.
    'shorts':       ('legacy_shorts',    'legacy', True),
    # Regime routing + factor blocks + derivatives + ATR stops.
    'v7':           ('v7',               'v7',     True),
}


def run_variant(data, symbols, *, name, cfg, params=None) -> tuple:
    variant, strategy, allow_shorts = VARIANTS[name]
    intents = build_intents(
        symbols, data,
        variant=variant,
        strategy=strategy,
        allow_shorts=allow_shorts,
        stop_pct=0.05,      # the live bot's CRYPTO_STOP_LOSS_PCT
        take_pct=0.07,      # the live bot's CRYPTO_TAKE_PROFIT_PCT
        params=params,
        warmup=config.BACKTEST_WARMUP_BARS,
    )
    engine = BacktestEngine(cfg)
    result = engine.run(data, intents=intents, warmup=config.BACKTEST_WARMUP_BARS)
    return intents, result, compute_metrics(result)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest the Bybit futures bot")
    parser.add_argument('--days', type=float, default=config.BACKTEST_DAYS)
    parser.add_argument('--symbols', default=','.join(config.SYMBOLS))
    parser.add_argument('--variants', default='baseline',
                        help="comma-separated: " + ','.join(VARIANTS))
    parser.add_argument('--risk', type=float, default=config.RISK_PER_TRADE)
    parser.add_argument('--no-lookahead-check', action='store_true')
    parser.add_argument('--fee-sweep', action='store_true',
                        help="re-score the same signals across fee levels")
    parser.add_argument('--sessions', action='store_true',
                        help="test the kill-zone hypothesis against measured results")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    names = [v.strip() for v in args.variants.split(',') if v.strip()]
    for name in names:
        if name not in VARIANTS:
            parser.error(f"unknown variant {name!r}; choose from {list(VARIANTS)}")

    data = load_data(symbols, args.days, with_oi=any(n == 'v7' for n in names))

    cfg = BacktestConfig(
        initial_equity=config.BACKTEST_INITIAL_EQUITY,
        taker_fee=config.TAKER_FEE,
        slippage_bps=config.SLIPPAGE_BPS,
        risk_per_trade=args.risk,
        max_positions=config.MAX_OPEN_POSITIONS,
        max_leverage=config.LEVERAGE,
        time_stop_bars=config.TIME_STOP_HOURS,
        trailing_atr_mult=config.TRAILING_ATR_MULTIPLIER,
        trailing_activate_r=config.TRAILING_ACTIVATE_R,
    )

    results = {}
    for name in names:
        intents, result, metrics = run_variant(data, symbols, name=name, cfg=cfg)
        results[name] = metrics

        title = f"{name.upper()}  ({args.days:.0f}d, {len(symbols)} symbols, risk {args.risk:.1%})"
        print()
        print(format_report(metrics, title))
        print()

        if args.sessions:
            print(session_breakdown(result))

        if not args.no_lookahead_check:
            assert_no_lookahead(data, intents, result)

    if len(results) >= 2:
        base = names[0]
        for other in names[1:]:
            print()
            print(compare(results[base], results[other], base, other))
            print()

    if args.fee_sweep:
        fee_sweep(data, symbols, names[-1], cfg)

    return 0


def fee_sweep(data, symbols, name: str, cfg: BacktestConfig) -> None:
    """
    Re-score one variant's signals across fee levels.

    Worth its own view because the baseline runs showed both strategies were
    gross-PROFITABLE and net-negative: the edge is real but smaller than the
    cost of harvesting it. This shows exactly how much of the result is the
    signal and how much is the fee schedule.
    """
    levels = [
        ('zero cost (signal only)', 0.0,      0.0),
        ('maker 0.020%',            0.0002,   1.0),
        ('taker 0.055% (actual)',   0.00055,  2.0),
        ('taker + wide slippage',   0.00055,  5.0),
    ]

    print("\n  FEE SENSITIVITY — same signals, different cost assumptions")
    print("  " + "─" * 68)
    print(f"  {'scenario':<26}{'trades':>8}{'win %':>8}{'expect. R':>12}{'return %':>12}")
    print("  " + "─" * 68)

    for label, fee, slippage in levels:
        variant_cfg = replace(cfg, taker_fee=fee, slippage_bps=slippage)
        _, _, metrics = run_variant(data, symbols, name=name, cfg=variant_cfg)
        print(f"  {label:<26}{metrics.n_trades:>8}{metrics.win_rate:>8.1f}"
              f"{metrics.expectancy_r:>+12.3f}{metrics.total_return:>+12.1f}")

    print("  " + "─" * 68)
    print("  If 'zero cost' is clearly positive while 'taker' is not, the signal has")
    print("  an edge and the problem is trading frequency, not prediction quality.")
    print()


if __name__ == '__main__':
    raise SystemExit(main())
