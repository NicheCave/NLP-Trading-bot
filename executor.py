# executor.py — risk-managed order placement via Alpaca
# ─────────────────────────────────────────────────────────────
import alpaca_trade_api as tradeapi
from loguru import logger
from datetime import date, datetime, timezone
import config
from signal_generator import Signal
from config import is_crypto
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoLatestBarRequest
from config import is_crypto, to_order_symbol

_pending_buys: set[str] = set()

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

def _sync_daily_trades_from_api():
    """
    Count today's filled orders directly from Alpaca so the in-memory
    counter survives process restarts and never drifts from reality.
    """
    global _daily_trades
    try:
        today_str = date.today().isoformat()
        orders = api.list_orders(
            status='filled',
            after=f"{today_str}T00:00:00Z",
            limit=500
        )
        _daily_trades = len(orders)
        logger.debug(f"Daily trade counter synced from Alpaca: {_daily_trades} filled orders today")
    except Exception as e:
        logger.warning(f"Could not sync daily trade counter from Alpaca: {e}")

def get_portfolio_value() -> float:
    return float(api.get_account().portfolio_value)

def get_open_positions() -> list:
    return api.list_positions()

def already_in_position(ticker: str) -> bool:
    """
    Check Alpaca for an open position. URL-encodes the slash in crypto
    symbols (BTC/USD -> BTC%2FUSD) so Alpaca does not parse it as a path.
    """
    try:
        pos = api.get_position(ticker)
        has_position = float(pos.qty) != 0
        if has_position:
            _pending_buys.discard(ticker)
        return has_position
    except tradeapi.rest.APIError as e:
        if e.status_code == 404:
            return False
        logger.warning(f"API error checking position for {ticker}: {e}")
        return False
    except Exception as e:
        if '404' in str(e):
            return False
        logger.warning(f"Unexpected error checking position for {ticker}: {e}")
        return False

def has_open_or_pending_buy_order(ticker: str) -> bool:
    """
    Query Alpaca directly for any open/pending buy orders for this ticker.
    This is the hard guard against duplicate entries — it survives process
    restarts and catches orders placed in previous sessions that have not
    filled or been cancelled yet.
    """
    order_symbol = to_order_symbol(ticker)
    try:
        open_orders = api.list_orders(status='open', limit=100)
        for o in open_orders:
            if o.symbol == order_symbol and o.side == 'buy':
                logger.debug(f"Found open buy order for {ticker} (id={o.id}) — skipping")
                return True
        return False
    except Exception as e:
        logger.warning(f"Could not check open orders for {ticker}: {e} — allowing trade")
        return False  # fail open rather than freeze trading entirely

def is_market_open(ticker: str) -> bool:
    if is_crypto(ticker):
        return True   # crypto never closes
    clock = api.get_clock()
    return clock.is_open

def check_exits():
    """Manually enforce take-profit and stop-loss for crypto positions."""
    try:
        positions = api.list_positions()
    except Exception as e:
        logger.warning(f"Could not fetch positions for exit check: {e}")
        return

    for pos in positions:
        symbol = pos.symbol  # e.g. 'BTC/USD'
        unrealized_pct = float(pos.unrealized_plpc)  # e.g. 0.05 = +5%

        # Map back to internal ticker for config lookup
        internal = next(
            (k for k, v in config.ALPACA_ORDER_SYMBOL.items() if v == symbol),
            symbol
        )
        take_profit = config.CRYPTO_TAKE_PROFIT_PCT if is_crypto(internal) else config.TAKE_PROFIT_PCT
        stop        = config.CRYPTO_STOP_LOSS_PCT   if is_crypto(internal) else config.STOP_LOSS_PCT

        if unrealized_pct >= take_profit:
            logger.info(f"Take-profit triggered for {symbol} at +{unrealized_pct:.1%}")
            try:
                qty_clean = str(float(pos.qty))
                api.submit_order(symbol=symbol, qty=qty_clean, side='sell',
                                 type='market', time_in_force='gtc')
                _pending_buys.discard(internal)
            except Exception as e:
                logger.error(f"Take-profit order failed for {symbol}: {e}")

        elif unrealized_pct <= -stop:
            logger.info(f"Stop-loss triggered for {symbol} at {unrealized_pct:.1%}")
            try:
                qty_clean = str(float(pos.qty))
                api.submit_order(symbol=symbol, qty=qty_clean, side='sell',
                                 type='market', time_in_force='gtc')
                _pending_buys.discard(internal)
            except Exception as e:
                logger.error(f"Stop-loss order failed for {symbol}: {e}")

def execute_signal(signal: Signal):
    """Execute a trade signal with full risk management checks."""
    global _daily_trades

    # ── Pre-flight checks ─────────────────────────────────────
    _reset_daily_counter_if_needed()
    _sync_daily_trades_from_api()   # always use the live count, not stale RAM

    if signal.action == 'HOLD':
        logger.info(f"HOLD {signal.ticker} — {signal.reason}")
        return

    if signal.action == 'BUY' and _daily_trades >= config.MAX_DAILY_TRADES:
        logger.warning("Daily trade limit reached — skipping BUY")
        return

    open_positions = get_open_positions()
    if signal.action == 'BUY' and len(open_positions) >= config.MAX_OPEN_POSITIONS:
        logger.warning("Max open positions reached — skipping BUY")
        return

    if signal.action == 'BUY':
        # Layer 1: in-memory lock (fast)
        if signal.ticker in _pending_buys:
            logger.debug(f"{signal.ticker} buy already pending (in-memory) — skipping")
            return
        # Layer 2: live position check via Alpaca (fixes BTC/ETH URL slash bug)
        if already_in_position(signal.ticker):
            logger.debug(f"Already holding {signal.ticker} — skipping")
            return
        # Layer 3: live open-order check via Alpaca (survives restarts)
        if has_open_or_pending_buy_order(signal.ticker):
            logger.debug(f"Open buy order already exists for {signal.ticker} — skipping")
            _pending_buys.add(signal.ticker)  # sync in-memory lock
            return
        # Acquire in-memory lock before any further async work
        _pending_buys.add(signal.ticker)

    if signal.action == 'SELL' and not already_in_position(signal.ticker):
        logger.debug(f"No position in {signal.ticker} — cached SELL signal skipped")
        return

    risk = config.CRYPTO_RISK_PER_TRADE if is_crypto(signal.ticker) else config.RISK_PER_TRADE
    stop = config.CRYPTO_STOP_LOSS_PCT if is_crypto(signal.ticker) else config.STOP_LOSS_PCT

    # ── Position sizing ───────────────────────────────────────
    portfolio    = get_portfolio_value()
    order_symbol = to_order_symbol(signal.ticker)

    if is_crypto(signal.ticker):
        request = CryptoLatestBarRequest(symbol_or_symbols=order_symbol)
        bar     = _crypto_client.get_crypto_latest_bar(request)
        price   = float(bar[order_symbol].close)
    else:
        price = float(api.get_latest_trade(signal.ticker).price)

    dollars = portfolio * risk * signal.confidence

    if is_crypto(signal.ticker):
        qty           = round(dollars / price, 6)
        qty_str       = str(qty)
        time_in_force = 'gtc'
    else:
        qty = int(dollars / price)
        if qty < 1:
            logger.warning(f"Position size too small for {signal.ticker} — skipping")
            _pending_buys.discard(signal.ticker)
            return
        qty_str       = str(qty)
        time_in_force = 'day'

    side = 'buy' if signal.action == 'BUY' else 'sell'

    # ── ForSELL: always close the entire position, not a calculated qty ──
    if signal.action == 'SELL':
        try:
            pos = api.get_position(signal.ticker)
            qty_clean = str(float(pos.qty))
            logger.info(f"Closing full position: {qty_str} {order_symbol}")
        except Exception as e:
            logger.error(f"Could not fetch position qty for SELL {signal.ticker}: {e}")
            return

    # ── Place the market order ────────────────────────────────
    try:
        order = api.submit_order(
            symbol=order_symbol,
            qty=qty_str,
            side=side,
            type='market',
            time_in_force=time_in_force
        )
        _daily_trades += 1
        logger.success(
            f"✅ {signal.action} {qty_str} {signal.ticker} @ ~${price:.2f} | "
            f"conf={signal.confidence:.0%} | {signal.reason}"
        )
        if signal.action == 'SELL':
            _pending_buys.discard(signal.ticker)

        # ── Place stop-loss order (stocks only) ──────────────
        if signal.action == 'BUY':
            stop_price = round(price * (1 - stop), 2)
            if is_crypto(signal.ticker):
                logger.warning(
                    f"  Stop-loss skipped for {signal.ticker} "
                    f"(crypto stop orders unsupported) — check_exits() will enforce ${stop_price}"
                )
            else:
                api.submit_order(
                    symbol=order_symbol,
                    qty=qty_str,
                    side='sell',
                    type='stop',
                    stop_price=stop_price,
                    time_in_force='gtc'
                )
                logger.info(f"  Stop-loss set at ${stop_price}")

    except Exception as e:
        logger.error(f"Order failed for {signal.ticker}: {e}")
        _pending_buys.discard(signal.ticker)