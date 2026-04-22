# executor.py — risk-managed order placement via Alpaca
# ─────────────────────────────────────────────────────────────
import alpaca_trade_api as tradeapi
from loguru import logger
from datetime import date
import config
from signal_generator import Signal
from config import is_crypto
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoLatestBarRequest
from config import is_crypto, to_order_symbol

_crypto_client = CryptoHistoricalDataClient(
    api_key=config.ALPACA_API_KEY,
    secret_key=config.ALPACA_SECRET_KEY
)

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
        pos = api.get_position(to_order_symbol(ticker))
        return float(pos.qty) != 0
    except:
        return False

def is_market_open(ticker: str) -> bool:
    if is_crypto(ticker):
        return True   # crypto never closes
    clock = api.get_clock()
    return clock.is_open

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
        logger.debug(f"Already holding {signal.ticker} — cached BUY signal skipped")
        return

    risk = config.CRYPTO_RISK_PER_TRADE if is_crypto(signal.ticker) else config.RISK_PER_TRADE
    stop = config.CRYPTO_STOP_LOSS_PCT if is_crypto(signal.ticker) else config.STOP_LOSS_PCT

    # ── Position sizing ───────────────────────────────────────
    portfolio = get_portfolio_value()

    if is_crypto(signal.ticker):
        request = CryptoLatestBarRequest(symbol_or_symbols=signal.ticker)
        bar = _crypto_client.get_crypto_latest_bar(request)
        price = float(bar[signal.ticker].close)
    else:
        price = float(api.get_latest_trade(signal.ticker).price)

    dollars    = portfolio * risk * signal.confidence

    if is_crypto(signal.ticker):
        # Crypto: use fractional qty, rounded to 6 decimal places
        qty = round(dollars / price, 6)
        qty_str = str(qty)
        time_in_force = 'gtc'  # crypto orders must be GTC, not 'day'
    else:
        # Stocks: whole shares only
        qty = int(dollars / price)
        if qty < 1:
            logger.warning(f"Position size too small for {signal.ticker} — skipping")
            return
        qty_str = str(qty)
        time_in_force = 'day'

    side = 'buy' if signal.action == 'BUY' else 'sell'

    order_symbol = to_order_symbol(signal.ticker)
    # ── Place the market order ────────────────────────────────
    try:
        order = api.submit_order(
            symbol=order_symbol,  # ← changed
            qty=qty_str,
            side=side,
            type='market',
            time_in_force=time_in_force
        )
        _daily_trades += 1
        logger.success(
            f"✅ {signal.action} {qty}x {signal.ticker} @ ~${price:.2f} | "
            f"conf={signal.confidence:.0%} | {signal.reason}"
        )

        # ── Place stop-loss order ────────────────────────────────
        if signal.action == 'BUY':
            stop_price = round(price * (1 - stop), 2)
            api.submit_order(
                symbol=order_symbol,  # ← changed
                qty=qty_str,
                side='sell',
                type='stop',
                stop_price=stop_price,
                time_in_force='gtc'
            )
            logger.info(f"  Stop-loss set at ${stop_price}")



    except Exception as e:
        logger.error(f"Order failed for {signal.ticker}: {e}")