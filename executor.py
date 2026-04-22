# executor.py — risk-managed order placement via Alpaca
# ─────────────────────────────────────────────────────────────
import alpaca_trade_api as tradeapi
from loguru import logger
from datetime import date
import config
from signal_generator import Signal

# ── Connect to Alpaca API ─────────────────────────────────────
api = tradeapi.REST(
    config.ALPACA_API_KEY,
    config.ALPACA_SECRET_KEY,
    config.ALPACA_BASE_URL,
    api_version='v2'
)

# ── Daily trade counter (resets at midnight) ──────────────────
_today        = date.today()
_daily_trades = 0

def _reset_daily_counter_if_needed():
    global _today, _daily_trades
    today = date.today()
    if today != _today:
        _today = today
        _daily_trades = 0
        logger.info("New day — daily trade counter reset.")

def get_portfolio_value() -> float:
    return float(api.get_account().portfolio_value)

def get_open_positions() -> list:
    return api.list_positions()

def already_in_position(ticker: str) -> bool:
    try:
        pos = api.get_position(ticker)
        return float(pos.qty) != 0
    except:
        return False

def execute_signal(signal: Signal):
    """Execute a trade signal with full risk management checks."""
    global _daily_trades

    # ── Pre-flight checks ─────────────────────────────────────
    _reset_daily_counter_if_needed()

    if signal.action == 'HOLD':
        logger.info(f"HOLD {signal.ticker} — {signal.reason}")
        return

    if _daily_trades >= config.MAX_DAILY_TRADES:
        logger.warning(f"Daily trade limit ({config.MAX_DAILY_TRADES}) reached — skipping {signal.ticker}")
        return

    open_positions = get_open_positions()
    if len(open_positions) >= config.MAX_OPEN_POSITIONS:
        logger.warning(f"Max open positions ({config.MAX_OPEN_POSITIONS}) reached — skipping")
        return

    if signal.action == 'BUY' and already_in_position(signal.ticker):
        logger.info(f"Already holding {signal.ticker} — skipping duplicate BUY")
        return

    # ── Position sizing ───────────────────────────────────────
    portfolio  = get_portfolio_value()
    price      = float(api.get_latest_trade(signal.ticker).price)
    dollars    = portfolio * config.RISK_PER_TRADE * signal.confidence
    qty        = int(dollars / price)

    if qty < 1:
        logger.warning(f"Position size too small for {signal.ticker} — skipping")
        return

    side = 'buy' if signal.action == 'BUY' else 'sell'

    # ── Place the market order ────────────────────────────────
    try:
        order = api.submit_order(
            symbol        = signal.ticker,
            qty           = qty,
            side          = side,
            type          = 'market',
            time_in_force = 'day'
        )
        _daily_trades += 1
        logger.success(
            f"✅ {signal.action} {qty}x {signal.ticker} @ ~${price:.2f} | "
            f"conf={signal.confidence:.0%} | {signal.reason}"
        )

        # ── Place stop-loss order ────────────────────────────────
        if signal.action == 'BUY':
            stop_price = round(price * (1 - config.STOP_LOSS_PCT), 2)
            api.submit_order(
                symbol=signal.ticker, qty=qty, side='sell',
                type='stop', stop_price=stop_price, time_in_force='gtc'
            )
            logger.info(f"  Stop-loss set at ${stop_price}")

    except Exception as e:
        logger.error(f"Order failed for {signal.ticker}: {e}")