# main.py
import schedule
import time
import sys
from loguru import logger
from datetime import datetime
import config
from signal_generator import generate_signal, Signal
from executor import execute_signal, check_exits

# ── Logging setup ─────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, colorize=True, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
logger.add("logs/bot_{time:YYYY-MM-DD}.log", rotation="1 day",
           retention="7 days", level="DEBUG")

# ── Shared signal cache ───────────────────────────────────────
# sentiment_cycle() writes here every 60 min
# execution_cycle() reads here every 5 min
_cached_signals = {}  # ticker → Signal


def sentiment_cycle():
    """Slow loop: fetch news → run FinBERT → cache signals. Runs every 60 min."""
    logger.info(f"════ Sentiment cycle: {datetime.now().strftime('%Y-%m-%d %H:%M')} ════")

    for ticker in config.WATCHLIST:
        try:
            signal = generate_signal(ticker)
            _cached_signals[ticker] = signal
            logger.info(f"Cached signal: {ticker} → {signal.action} | {signal.reason}")
        except Exception as e:
            logger.error(f"Sentiment error for {ticker}: {e}")

    logger.info(f"════ Sentiment cycle complete ════\n")


def execution_cycle():
    """Fast loop: check cached signals → place orders if actionable. Runs every 5 min."""
    if not _cached_signals:
        logger.debug("No cached signals yet — skipping execution cycle")
        return

    logger.info(f"── Execution check: {datetime.now().strftime('%H:%M')} ──")

    for ticker, signal in list(_cached_signals.items()):  # list() so we can modify during iteration
        try:
            if signal.action != 'HOLD':
                logger.info(f"Acting on cached signal: {ticker} → {signal.action}")
            if signal.action == 'SELL':
                from executor import already_in_position
                execute_signal(signal)
                # Only clear if position is actually gone now
                if not already_in_position(ticker):
                    _cached_signals[ticker] = Signal(
                        ticker, 'HOLD', 0.0, 0.0, 0.0, 0.0, 0,
                        "Post-sell hold — waiting for next sentiment cycle"
                    )
                    logger.info(f"Signal cleared for {ticker} — position closed")
                else:
                    logger.warning(f"SELL for {ticker} may have failed — keeping SELL signal")
            else:
                execute_signal(signal)
        except Exception as e:
            logger.error(f"Execution error for {ticker}: {e}")

    check_exits()


# ── Schedule both loops independently ────────────────────────
schedule.every(config.CHECK_INTERVAL).minutes.do(sentiment_cycle)
schedule.every(config.EXECUTION_INTERVAL).seconds.do(execution_cycle)

logger.info("🤖 FinBERT Sentiment Bot starting...")
logger.info(f"   Device:      {config.DEVICE}")
logger.info(f"   Watchlist:   {config.WATCHLIST}")
logger.info(f"   Sentiment:   every {config.CHECK_INTERVAL} min")
logger.info(f"   Execution:   every {config.EXECUTION_INTERVAL} min")
logger.info(f"   Paper trading: {config.ALPACA_BASE_URL}")

# Run sentiment immediately on start to populate cache before first execution check
sentiment_cycle()
check_exits()

while True:
    schedule.run_pending()
    time.sleep(10)  # tighter sleep so 5-min schedule fires on time