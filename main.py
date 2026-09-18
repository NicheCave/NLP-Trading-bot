#!/usr/bin/env python
# main.py — Bybit USDT-perpetual futures bot (v7).
# ─────────────────────────────────────────────────────────────
#   python main.py --mode paper     # local simulator, real prices, no keys
#   python main.py --mode demo      # Bybit Demo Trading (fake money)
#   python main.py --mode live      # real funds — requires --i-know
#
# Long AND short, regime-routed, ATR-sized, with stop-loss and take-profit
# resident at the exchange rather than enforced by a polling loop.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import schedule
from loguru import logger

import config
import params_store
from brokers import make_broker
from brokers.base import BrokerError
from data import get_funding_history, get_klines
from edge_tracker import EdgeTracker
from executor import Executor
from market_data import fetch_bars
from strategy_v7 import DEFAULT_PARAMS, V7Decision, decide

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass


class TradingBot:
    def __init__(self, mode: str = 'paper', params=None):
        self.mode = mode
        if params is not None:
            self.params, self.tuning = params, {}
        else:
            # Prefer walk-forward-selected parameters when they exist; the
            # hand-picked defaults are only a starting point.
            self.params, self.tuning = params_store.load(DEFAULT_PARAMS)
        self.broker = make_broker(mode)
        self.edge = EdgeTracker()
        self.executor = Executor(self.broker, on_close=self.edge.record_close)
        self.decisions: dict[str, V7Decision] = {}
        self.funding_rates: dict[str, float] = {}

        self.broker.prepare(config.SYMBOLS)

    # ══════════════════════════════════════════════════════════
    # CYCLES
    # ══════════════════════════════════════════════════════════

    def analysis_cycle(self) -> None:
        """Refresh market data and re-decide for every symbol."""
        logger.info(f"════ analysis  {datetime.now().strftime('%Y-%m-%d %H:%M')} ════")

        anchor_trend = self._anchor_trend()

        for symbol in config.SYMBOLS:
            try:
                bars_1h = fetch_bars(symbol, '1h')
                if bars_1h.empty or len(bars_1h) < 100:
                    logger.warning(f"{symbol}: insufficient bars — skipping")
                    continue

                bars_1h = bars_1h.iloc[-config.ANALYSIS_WINDOW_BARS:]
                bars_1d = fetch_bars(symbol, '1d')
                funding = self._funding_series(symbol)

                decision = decide(
                    symbol=symbol,
                    hist_1h=bars_1h,
                    hist_1d=bars_1d,
                    ts=bars_1h.index[-1],
                    btc_trend=None if config.is_anchor(symbol) else anchor_trend,
                    funding=funding,
                    params=self.params,
                )
                self.decisions[symbol] = decision

                if decision.direction != 'flat':
                    logger.success(
                        f"{symbol} → {decision.direction.upper()} "
                        f"conf={decision.confidence:.0%} net={decision.net_score:+.3f} "
                        f"agree={decision.agreement:.0%} [{decision.regime.regime.value}]")
                else:
                    logger.info(f"{symbol} → flat "
                                f"[{decision.regime.regime.value if decision.regime else '?'}] "
                                f"{decision.rejected}")
            except Exception as e:
                logger.error(f"Analysis error for {symbol}: {e}")

        logger.info("════ analysis complete ════\n")

    def execution_cycle(self) -> None:
        """Act on cached decisions."""
        if not self.decisions:
            return

        # Fail closed: without a truthful list of what we already hold, the
        # duplicate-entry guard below is meaningless, so open nothing.
        try:
            positions = self.broker.get_positions()
        except BrokerError as e:
            logger.warning(f"Execution cycle skipped — cannot read positions ({e})")
            return

        # Strongest signal first: with more symbols than position slots,
        # several can fire on the same cycle and only some will fit. Taking
        # them in watchlist order would give the slots to whichever symbol
        # is listed first rather than whichever setup is best.
        ranked = sorted(self.decisions.items(),
                        key=lambda kv: -kv[1].confidence)
        for symbol, decision in ranked:
            if decision.direction == 'flat' or symbol in positions:
                continue
            try:
                intent = self._intent_for(symbol, decision)
                if intent is None:
                    continue
                opened = self.executor.open_position(
                    symbol=symbol,
                    direction=decision.direction,
                    stop_price=intent['stop'],
                    take_price=intent['take'],
                    atr=intent['atr'],
                    confidence=decision.confidence,
                    reason=decision.reason,
                    tags=decision.tags,
                )
                if opened:
                    # Consume the decision so it can't re-fire next cycle.
                    self.decisions[symbol] = V7Decision(blocks={}, tags=[])
            except Exception as e:
                logger.error(f"Execution error for {symbol}: {e}")

    def manage_cycle(self) -> None:
        """Trailing stops, time stops, funding exits, and reconciliation."""
        try:
            self.executor.manage_positions(self.funding_rates)
        except Exception as e:
            logger.error(f"Manage error: {e}")

    def edge_cycle(self) -> None:
        try:
            for line in self.edge.report().split('\n'):
                logger.info(line)
        except Exception as e:
            logger.error(f"Edge report error: {e}")

    def funding_cycle(self) -> None:
        """Refresh the latest funding rate per symbol (settles every 8h)."""
        for symbol in config.SYMBOLS:
            try:
                history = get_funding_history(symbol, days=5)
                if not history.empty:
                    self.funding_rates[symbol] = float(history['funding_rate'].iloc[-1])
            except Exception as e:
                logger.debug(f"Funding refresh failed for {symbol}: {e}")

    # ══════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════

    def _anchor_trend(self) -> Optional[str]:
        """BTC trend, used as the intermarket anchor for altcoins."""
        try:
            bars = fetch_bars(config.ANCHOR_SYMBOL, '1h')
            if bars.empty:
                return None
            fast = bars['close'].ewm(span=config.EMA_FAST, adjust=False).mean()
            slow = bars['close'].ewm(span=config.EMA_SLOW, adjust=False).mean()
            if float(fast.iloc[-1]) > float(slow.iloc[-1]) * 1.002:
                return 'uptrend'
            if float(fast.iloc[-1]) < float(slow.iloc[-1]) * 0.998:
                return 'downtrend'
            return 'sideways'
        except Exception as e:
            logger.debug(f"Anchor trend failed: {e}")
            return None

    def _funding_series(self, symbol: str) -> Optional[pd.Series]:
        try:
            history = get_funding_history(symbol, days=90)
            return history['funding_rate'] if not history.empty else None
        except Exception:
            return None

    def _intent_for(self, symbol: str, decision: V7Decision) -> Optional[dict]:
        """Recompute the ATR stop/target against the CURRENT price."""
        import risk
        from indicators import atr as calc_atr

        bars = fetch_bars(symbol, '1h')
        if bars.empty:
            return None

        atr_result = calc_atr(bars, period=config.ATR_PERIOD)
        if atr_result is None or atr_result.value <= 0:
            return None

        price = self.broker.get_price(symbol)
        if price is None:
            return None

        stop, take = risk.atr_stop_target(
            price, decision.direction, atr_result.value,
            stop_mult=self.params.atr_stop_mult,
            target_mult=self.params.atr_target_mult,
        )
        return {'stop': stop, 'take': take, 'atr': atr_result.value}

    # ══════════════════════════════════════════════════════════
    # RUN
    # ══════════════════════════════════════════════════════════

    def run(self) -> None:
        self._banner()

        schedule.every(config.CHECK_INTERVAL).minutes.do(self._safe, self.analysis_cycle)
        schedule.every(config.EXECUTION_INTERVAL).seconds.do(self._safe, self.execution_cycle)
        schedule.every(config.EXIT_INTERVAL).seconds.do(self._safe, self.manage_cycle)
        schedule.every(config.EDGE_LOG_INTERVAL).minutes.do(self._safe, self.edge_cycle)
        schedule.every(30).minutes.do(self._safe, self.funding_cycle)

        self._safe(self.funding_cycle)
        self._safe(self.analysis_cycle)
        self._safe(self.manage_cycle)

        while True:
            try:
                schedule.run_pending()
            except Exception as e:
                # Last line of defence. The scheduler itself should never be
                # able to take the process down.
                logger.error(f"Scheduler error: {e}")
            time.sleep(5)

    def _safe(self, cycle) -> None:
        """
        Run one scheduled cycle, absorbing anything it throws.

        A trading bot must outlive transient failures. An unhandled
        exception in any single cycle used to propagate through
        schedule.run_pending() and terminate the process — a dropped DNS
        lookup while the machine was idle was enough to kill it, leaving
        positions open with nothing watching them.

        Exchange-side stops mean open positions stay protected regardless,
        but the bot still needs to be alive to manage time stops, trailing
        stops and reconciliation.
        """
        try:
            cycle()
        except BrokerError as e:
            # Expected during an outage; already logged with context.
            logger.warning(f"{cycle.__name__} skipped: {e}")
        except Exception as e:
            logger.error(f"{cycle.__name__} failed: {e}")
            logger.opt(exception=True).debug("traceback")

    def _banner(self) -> None:
        equity = self.broker.get_equity()
        logger.info("🤖 Bybit perpetual futures bot (v7)")
        logger.info(f"   Mode:        {self.mode.upper()}  ({self.broker.name})")
        logger.info(f"   Equity:      {equity:,.2f} {config.QUOTE_CURRENCY}")
        logger.info(f"   Symbols:     {', '.join(config.SYMBOLS)}")
        logger.info(f"   Direction:   LONG + SHORT")
        logger.info(f"   Leverage:    {config.LEVERAGE}× {config.MARGIN_MODE.lower()}")
        logger.info(f"   Risk/trade:  {config.RISK_PER_TRADE:.1%} of equity at the stop")
        # Read from self.params, not config: tuned parameters override the
        # config defaults, and a banner that reports the defaults while the
        # bot trades something else is worse than no banner at all.
        logger.info(f"   Stops:       {self.params.atr_stop_mult}×ATR, "
                    f"target {self.params.atr_target_mult}×ATR "
                    f"({self.params.atr_target_mult / self.params.atr_stop_mult:.1f} R)")
        logger.info(f"   Protection:  exchange-resident, triggered on "
                    f"{config.TRIGGER_PRICE_TYPE}")
        logger.info(f"   Max open:    {config.MAX_OPEN_POSITIONS} "
                    f"(heat cap {config.MAX_PORTFOLIO_HEAT:.0%})")
        logger.info(f"   Time stop:   {config.TIME_STOP_HOURS}h")
        logger.info(f"   Gates:       |net|≥{self.params.min_net:.2f} "
                    f"agreement≥{self.params.min_agreement:.0%} "
                    f"blocks≥{self.params.min_active_blocks}"
                    + ("  kill-zones only" if self.params.use_kill_zones else ""))

        if self.tuning:
            logger.info(f"   Parameters:  walk-forward tuned — out-of-sample "
                        f"{self.tuning.get('oos_expectancy_r', 0):+.3f}R over "
                        f"{self.tuning.get('oos_trades', 0)} trades")
        else:
            logger.warning("   Parameters:  UNTUNED defaults — run "
                           "run_walkforward.py before trusting this")

        if equity <= 0 and self.mode == 'demo':
            logger.warning("   Demo equity is zero — fund it from the Bybit UI, or call "
                           "broker.request_demo_funds()")


class _PybitToLoguru(logging.Handler):
    """
    Route pybit's internal chatter into the debug log instead of the console.

    pybit logs every retry attempt at ERROR, including transient connection
    resets that it then recovers from on the next try — which is the normal
    state of affairs against api-demo.bybit.com. Left alone it fills the
    console with alarming messages about things that worked.

    Genuine failures are unaffected: they surface through BybitFuturesBroker's
    own error handling after the retries are exhausted.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            return
        level = 'DEBUG' if 'Retrying' in message else 'WARNING'
        logger.log(level, f"[pybit] {message}")


def _quieten_pybit() -> None:
    pybit_logger = logging.getLogger('pybit')
    pybit_logger.handlers = [_PybitToLoguru()]
    pybit_logger.propagate = False
    pybit_logger.setLevel(logging.INFO)


def main() -> int:
    parser = argparse.ArgumentParser(description="Bybit futures trading bot")
    parser.add_argument('--mode', default='paper', choices=['paper', 'demo', 'live'])
    parser.add_argument('--log-level', default='INFO')
    parser.add_argument('--i-know', action='store_true',
                        help="required acknowledgement for --mode live")
    parser.add_argument('--fund-demo', type=float, metavar='USDT',
                        help="request demo funds, then exit")
    parser.add_argument('--skip-clock-check', action='store_true',
                        help="start even if the system clock is out of sync")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stdout, colorize=True, level=args.log_level,
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    logger.add("logs/bot_{time:YYYY-MM-DD}.log", rotation="1 day",
               retention="14 days", level="DEBUG", encoding='utf-8')

    _quieten_pybit()

    if args.mode == 'live' and not args.i_know:
        logger.error("--mode live trades real funds. Re-run with --i-know to confirm.")
        return 1

    # Report the clock, but don't refuse to start over it: the broker
    # corrects request timestamps in software (brokers/clock_sync.py), so a
    # wrong system clock is a warning rather than a blocker.
    if args.mode in ('demo', 'live'):
        from brokers.clock_sync import measure_offset
        offset = measure_offset()
        if offset is None:
            logger.warning("Could not reach Bybit to check the clock")
        elif abs(offset) > 1000:
            logger.warning(f"System clock is {offset / 1000:+.1f}s off Bybit — "
                           f"request timestamps will be corrected in software.")
            logger.warning("Worth fixing at the source too (ADMIN PowerShell):")
            logger.warning('    w32tm /config /manualpeerlist:"time.windows.com,0x9" '
                           '/syncfromflags:manual /update')
            logger.warning("    w32tm /resync /force")

    bot = TradingBot(mode=args.mode)

    if args.fund_demo:
        bot.broker.request_demo_funds(args.fund_demo)
        return 0

    try:
        bot.run()
    except KeyboardInterrupt:
        logger.info("Shutting down — open positions and their exchange-side stops "
                    "are left in place.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
