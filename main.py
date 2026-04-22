# main.py — bot entry point and scheduler
# Run this file to start the bot: python main.py
# ─────────────────────────────────────────────────────────────
import schedule
import time
import sys
from loguru import logger
from datetime import datetime
import config
from signal_generator import generate_signal
from executor import execute_signal

# ── Configure logging: console + rotating file ───────────────
logger.remove()  # remove default handler
logger.add(sys.stdout, colorize=True, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
logger.add("logs/bot_{time:YYYY-MM-DD}.log", rotation="1 day",
           retention="7 days", level="DEBUG")  # keep 7 days of logs


def run_cycle():
    """One full cycle: fetch → score → signal → execute for all tickers."""
    logger.info(f"════ Cycle start: {datetime.now().strftime('%Y-%m-%d %H:%M')} ════")
    logger.info(f"Watchlist: {', '.join(config.WATCHLIST)}")

    for ticker in config.WATCHLIST:
        logger.info(f"── Processing {ticker} ──")
        try:
            signal = generate_signal(ticker)
            execute_signal(signal)
        except Exception as e:
            logger.error(f"Unhandled error for {ticker}: {e}")

    logger.info(f"════ Cycle complete ════\n")


# ── Schedule: run every N minutes ────────────────────────────
schedule.every(config.CHECK_INTERVAL).minutes.do(run_cycle)

logger.info("🤖 FinBERT Sentiment Bot starting...")
logger.info(f"   Device: {config.DEVICE}")
logger.info(f"   Watchlist: {config.WATCHLIST}")
logger.info(f"   Interval: every {config.CHECK_INTERVAL} minutes")
logger.info(f"   Paper trading: {config.ALPACA_BASE_URL}")

run_cycle()  # run immediately on start

while True:
    schedule.run_pending()
    time.sleep(30)