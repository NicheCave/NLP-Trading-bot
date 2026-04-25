# main.py — TA + SMC crypto bot orchestrator (v2)
# ─────────────────────────────────────────────────────────────
# Changes from v1:
#   - Added sync_cycle() on DAILY_SYNC_INTERVAL — daily trade counter now
#     synced once per minute instead of every execution cycle (every 20s).
#     This reduces Alpaca API calls by ~2/min with no accuracy loss.
#   - CHECK_INTERVAL now 5 min (was 2) — see config.py for rationale.
#   - Startup logs updated to reflect SMC layer.
# ─────────────────────────────────────────────────────────────
import schedule
import time
import sys
from loguru import logger
from datetime import datetime
import config
from signal_generator import generate_signal, Signal
from executor import execute_signal, check_exits, sync_daily_counter

# ── Logging setup ─────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, colorize=True, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
logger.add("logs/bot_{time:YYYY-MM-DD}.log", rotation="1 day",
           retention="7 days", level="DEBUG")

# ── Signal cache ──────────────────────────────────────────────
_cached_signals: dict[str, Signal] = {}


def analysis_cycle():
    """
    Slow loop: fetch OHLCV bars → run TA + SMC indicators → cache signals.
    Runs every CHECK_INTERVAL minutes (default: 5 min).
    """
    logger.info(f"════ TA+SMC analysis cycle: {datetime.now().strftime('%Y-%m-%d %H:%M')} ════")

    for ticker in config.WATCHLIST:
        try:
            signal = generate_signal(ticker)
            _cached_signals[ticker] = signal
            logger.info(
                f"Signal cached: {ticker} → {signal.action} "
                f"(bull={signal.bull_pct:.0%} bear={signal.bear_pct:.0%} "
                f"net={signal.avg_score:+.3f} conf={signal.confidence:.0%})"
            )
        except Exception as e:
            logger.error(f"Analysis error for {ticker}: {e}", exc_info=True)

    logger.info("════ TA+SMC analysis cycle complete ════\n")


def execution_cycle():
    """
    Fast loop: read cached signals → execute if actionable.
    Runs every EXECUTION_INTERVAL seconds (default: 20 s).
    """
    if not _cached_signals:
        logger.debug("No cached signals yet — skipping execution cycle")
        return

    logger.debug(f"── Execution check: {datetime.now().strftime('%H:%M:%S')} ──")

    for ticker, signal in list(_cached_signals.items()):
        try:
            if signal.action != 'HOLD':
                logger.info(f"Acting on cached signal: {ticker} → {signal.action}")

            if signal.action == 'SELL':
                from executor import already_in_position
                execute_signal(signal)
                if not already_in_position(ticker):
                    _cached_signals[ticker] = Signal(
                        ticker, 'HOLD', 0.0, 0.0, 0.0, 0.0, 0,
                        "Post-sell hold — waiting for next analysis cycle"
                    )
                    logger.info(f"Signal cleared for {ticker} — position closed")
                else:
                    logger.warning(f"SELL for {ticker} may not have filled — keeping signal")
            else:
                execute_signal(signal)
        except Exception as e:
            logger.error(f"Execution error for {ticker}: {e}", exc_info=True)


def exit_cycle():
    """
    Fastest loop: check open positions for stop-loss / take-profit.
    Runs every EXIT_INTERVAL seconds (default: 30 s).
    """
    try:
        check_exits()
    except Exception as e:
        logger.error(f"Exit check error: {e}")


def sync_cycle():
    """
    Sync the daily trade counter from Alpaca.
    Runs every DAILY_SYNC_INTERVAL seconds (default: 60 s).
    Replaces the per-execution-cycle sync that was wasting API calls.
    """
    try:
        sync_daily_counter()
    except Exception as e:
        logger.error(f"Daily sync error: {e}")


# ── Schedule all loops independently ──────────────────────────
schedule.every(config.CHECK_INTERVAL).minutes.do(analysis_cycle)
schedule.every(config.EXECUTION_INTERVAL).seconds.do(execution_cycle)
schedule.every(config.EXIT_INTERVAL).seconds.do(exit_cycle)
schedule.every(config.DAILY_SYNC_INTERVAL).seconds.do(sync_cycle)

logger.info("🤖 Crypto TA + SMC Bot starting (v2)...")
logger.info(f"   Device:        {config.DEVICE}")
logger.info(f"   Watchlist:     {config.WATCHLIST}")
logger.info(f"   Analysis:      every {config.CHECK_INTERVAL} min  (TA + SMC)")
logger.info(f"   Execution:     every {config.EXECUTION_INTERVAL} s")
logger.info(f"   Exit guard:    every {config.EXIT_INTERVAL} s")
logger.info(f"   Counter sync:  every {config.DAILY_SYNC_INTERVAL} s")
logger.info(f"   Alpaca:        {config.ALPACA_BASE_URL}")

# Run immediately to populate cache
analysis_cycle()
exit_cycle()
sync_cycle()

while True:
    schedule.run_pending()
    time.sleep(5)