# executor.py — risk-managed order placement via Alpaca (v3)
# ─────────────────────────────────────────────────────────────
# Changes from v2 (Bellafiore pyramiding + Elliott Wave caution):
#
#   - scale_in_winner(): NEW — adds 50% more size to an already-profitable
#     position when signal.add_to_winner is True AND unrealized P&L >= 1 ATR.
#     Implements Bellafiore's "One Good Trade" load-the-boat logic:
#     enter at 50% on the initial signal, add the other 50% only when the
#     trade proves itself.
#
#   - execute_signal(): now checks signal.wave5_caution and halves
#     CRYPTO_RISK_PER_TRADE for that trade if the Elliott Wave 5 warning
#     is active. This reduces position size automatically — no manual override.
#
#   - execute_signal(): now checks signal.add_to_winner and calls
#     scale_in_winner() instead of placing a duplicate full-size buy.
#
#   - All other logic preserved from v2 (3-layer duplicate guard, daily
#     sync on DAILY_SYNC_INTERVAL, crypto stop-loss via check_exits).
# ─────────────────────────────────────────────────────────────
import alpaca_trade_api as tradeapi
from loguru import logger
from datetime import date, datetime, timezone
import config
from signal_generator import Signal
from config import is_crypto, to_order_symbol
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoLatestBarRequest
import pandas as pd

_pending_buys: set[str] = set()

_crypto_client = CryptoHistoricalDataClient(
    api_key=config.ALPACA_API_KEY,
    secret_key=config.ALPACA_SECRET_KEY,
)

api = tradeapi.REST(
    config.ALPACA_API_KEY,
    config.ALPACA_SECRET_KEY,
    config.ALPACA_BASE_URL,
    api_version='v2',
)

# ── Daily trade counter ───────────────────────────────────────
_today        = date.today()
_daily_trades = 0


def _reset_daily_counter_if_needed():
    global _today, _daily_trades
    today = date.today()
    if today != _today:
        _today        = today
        _daily_trades = 0
        logger.info("New day — daily trade counter reset.")


def sync_daily_counter():
    """
    Sync today's filled order count directly from Alpaca.
    Call on DAILY_SYNC_INTERVAL (main.py). Survives restarts.
    """
    global _daily_trades
    _reset_daily_counter_if_needed()
    try:
        today_str = date.today().isoformat()
        orders    = api.list_orders(
            status='filled',
            after=f"{today_str}T00:00:00Z",
            limit=500,
        )
        _daily_trades = len(orders)
        logger.debug(f"Daily trade counter synced: {_daily_trades} filled orders today")
    except Exception as e:
        logger.warning(f"Could not sync daily trade counter: {e}")


def get_portfolio_value() -> float:
    return float(api.get_account().portfolio_value)


def get_open_positions() -> list:
    return api.list_positions()


def already_in_position(ticker: str) -> bool:
    try:
        pos          = api.get_position(ticker)
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
        return False


def is_market_open(ticker: str) -> bool:
    if is_crypto(ticker):
        return True
    clock = api.get_clock()
    return clock.is_open


def _get_current_price(ticker: str, order_symbol: str) -> float | None:
    """Fetch latest price for a ticker."""
    if is_crypto(ticker):
        try:
            request = CryptoLatestBarRequest(symbol_or_symbols=order_symbol)
            bar     = _crypto_client.get_crypto_latest_bar(request)
            return float(bar[order_symbol].close)
        except Exception as e:
            logger.warning(f"Price fetch error ({ticker}): {e}")
            return None
    else:
        try:
            return float(api.get_latest_trade(ticker).price)
        except Exception as e:
            logger.warning(f"Price fetch error ({ticker}): {e}")
            return None


def _get_atr_dollars(ticker: str) -> float:
    """
    Fetch the 14-bar ATR (in dollars) from the latest hourly bars.
    Used by scale_in_winner() to confirm >= 1 ATR of profit before pyramiding.

    Delegates to indicators.atr() — the same formula used by market_analyzer —
    so both modules always agree on volatility. Returns 0.0 on failure so the
    caller can skip the scale-in gracefully.
    """
    try:
        from market_data import fetch_bars
        from alpaca.data.timeframe import TimeFrame
        from indicators import atr as calc_atr

        df = fetch_bars(ticker, TimeFrame.Hour, limit=20)
        if df.empty or len(df) < 15:
            return 0.0

        result = calc_atr(df, period=config.ATR_PERIOD)
        return result.value if result is not None else 0.0
    except Exception as e:
        logger.warning(f"ATR fetch failed for {ticker}: {e}")
        return 0.0


def scale_in_winner(ticker: str, signal: Signal):
    """
    Bellafiore Pyramiding: add 50% of the original risk to a winning position.

    Called when:
    - signal.add_to_winner is True (Alligator eating, AO bullish, AVWAP confirmed, RVOL in-play)
    - The position is already open and showing unrealized gain >= 1 ATR

    This implements "load the boat" — only adding fuel to a fire that's already burning.
    The scale-in uses 50% of the normal risk allocation to keep total risk controlled.
    """
    global _daily_trades
    _reset_daily_counter_if_needed()

    limit = config.MAX_DAILY_TRADES_CRYPTO if is_crypto(ticker) else config.MAX_DAILY_TRADES_STOCKS
    if _daily_trades >= limit:
        logger.info(f"Scale-in skipped for {ticker} — daily limit reached")
        return

    order_symbol = to_order_symbol(ticker)

    try:
        pos = api.get_position(ticker)
    except Exception:
        logger.debug(f"Scale-in skipped for {ticker} — no open position found")
        return

    unrealized_pct = float(pos.unrealized_plpc)
    current_price  = _get_current_price(ticker, order_symbol)
    if current_price is None:
        return

    atr_dollars = _get_atr_dollars(ticker)

    # Bellafiore rule: only pyramid if trade is AT LEAST 1 ATR in profit
    # This prevents adding to a losing position disguised as a winner.
    if atr_dollars > 0:
        atr_pct = atr_dollars / current_price
        if unrealized_pct < atr_pct:
            logger.info(
                f"Scale-in NOT triggered for {ticker} — "
                f"unrealized={unrealized_pct:.1%} < 1 ATR={atr_pct:.1%}. "
                f"Waiting for position to prove itself."
            )
            return
    elif unrealized_pct < 0.01:
        # Fallback: require at least 1% profit before adding
        logger.info(f"Scale-in skipped for {ticker} — position not profitable enough ({unrealized_pct:.1%})")
        return

    # Position size: 50% of normal risk (don't double the full position)
    portfolio   = get_portfolio_value()
    risk        = config.CRYPTO_RISK_PER_TRADE * 0.50   # half-size add-on
    dollars     = portfolio * risk * signal.confidence

    if is_crypto(ticker):
        qty     = round(dollars / current_price, 6)
        qty_str = str(qty)
        tif     = 'gtc'
    else:
        qty = int(dollars / current_price)
        if qty < 1:
            logger.info(f"Scale-in too small for {ticker} — skipping")
            return
        qty_str = str(qty)
        tif     = 'day'

    try:
        api.submit_order(
            symbol=order_symbol,
            qty=qty_str,
            side='buy',
            type='market',
            time_in_force=tif,
        )
        _daily_trades += 1
        logger.success(
            f"📈 SCALE-IN {qty_str} {ticker} @ ~${current_price:.2f} | "
            f"unrealized={unrealized_pct:.1%} | "
            f"conf={signal.confidence:.0%} | Bellafiore pyramiding"
        )
    except Exception as e:
        logger.error(f"Scale-in order failed for {ticker}: {e}")


# Tracks symbols for which a TP/SL sell order has already been submitted
# this cycle, preventing duplicate orders when check_exits() fires again
# before the first order has filled.
_pending_exits: set[str] = set()


def check_exits():
    """Manually enforce take-profit and stop-loss for crypto positions."""
    global _pending_exits

    try:
        positions = api.list_positions()
    except Exception as e:
        logger.warning(f"Could not fetch positions for exit check: {e}")
        return

    # Clear stale pending-exit locks for symbols no longer held
    current_symbols = {p.symbol for p in positions}
    _pending_exits  = {s for s in _pending_exits if s in current_symbols}

    for pos in positions:
        symbol         = pos.symbol
        unrealized_pct = float(pos.unrealized_plpc)

        internal = next(
            (k for k, v in config.ALPACA_ORDER_SYMBOL.items() if v == symbol),
            symbol,
        )
        take_profit = config.CRYPTO_TAKE_PROFIT_PCT if is_crypto(internal) else config.TAKE_PROFIT_PCT
        stop        = config.CRYPTO_STOP_LOSS_PCT   if is_crypto(internal) else config.STOP_LOSS_PCT

        if unrealized_pct >= take_profit:
            if symbol in _pending_exits:
                logger.debug(f"TP sell already pending for {symbol} — skipping duplicate")
                continue
            logger.info(f"Take-profit triggered for {symbol} at +{unrealized_pct:.1%}")
            try:
                qty_clean = str(float(pos.qty))
                api.submit_order(symbol=symbol, qty=qty_clean, side='sell',
                                 type='market', time_in_force='gtc')
                _pending_exits.add(symbol)
                _pending_buys.discard(internal)
            except Exception as e:
                logger.error(f"Take-profit order failed for {symbol}: {e}")

        elif unrealized_pct <= -stop:
            if symbol in _pending_exits:
                logger.debug(f"SL sell already pending for {symbol} — skipping duplicate")
                continue
            logger.info(f"Stop-loss triggered for {symbol} at {unrealized_pct:.1%}")
            try:
                qty_clean = str(float(pos.qty))
                api.submit_order(symbol=symbol, qty=qty_clean, side='sell',
                                 type='market', time_in_force='gtc')
                _pending_exits.add(symbol)
                _pending_buys.discard(internal)
            except Exception as e:
                logger.error(f"Stop-loss order failed for {symbol}: {e}")


def execute_signal(signal: Signal):
    """Execute a trade signal with full risk management checks."""
    global _daily_trades

    _reset_daily_counter_if_needed()

    if signal.action == 'HOLD':
        logger.info(f"HOLD {signal.ticker} — {signal.reason}")
        return

    limit = config.MAX_DAILY_TRADES_CRYPTO if is_crypto(signal.ticker) else config.MAX_DAILY_TRADES_STOCKS
    if signal.action == 'BUY' and _daily_trades >= limit:
        logger.warning(f"Daily trade limit ({limit}) reached — skipping BUY for {signal.ticker}")
        return

    open_positions = get_open_positions()
    order_symbol   = to_order_symbol(signal.ticker)

    # ── Pyramiding check ──────────────────────────────────────
    # If the signal says to add to winner AND we already have a position,
    # route to scale_in_winner() instead of a normal entry order.
    if signal.action == 'BUY' and signal.add_to_winner and already_in_position(signal.ticker):
        logger.info(f"Pyramiding check for {signal.ticker} — delegating to scale_in_winner()")
        scale_in_winner(signal.ticker, signal)
        return

    if signal.action == 'BUY' and len(open_positions) >= config.MAX_OPEN_POSITIONS:
        logger.warning("Max open positions reached — skipping BUY")
        return

    if signal.action == 'BUY':
        # Layer 1: in-memory lock
        if signal.ticker in _pending_buys:
            logger.debug(f"{signal.ticker} buy already pending (in-memory) — skipping")
            return
        # Layer 2: live position check
        if already_in_position(signal.ticker):
            logger.debug(f"Already holding {signal.ticker} — skipping")
            return
        # Layer 3: live open-order check
        if has_open_or_pending_buy_order(signal.ticker):
            logger.debug(f"Open buy order already exists for {signal.ticker} — skipping")
            _pending_buys.add(signal.ticker)
            return
        _pending_buys.add(signal.ticker)

    if signal.action == 'SELL' and not already_in_position(signal.ticker):
        logger.debug(f"No position in {signal.ticker} — cached SELL signal skipped")
        return

    # ── Position sizing ───────────────────────────────────────
    risk = config.CRYPTO_RISK_PER_TRADE if is_crypto(signal.ticker) else config.RISK_PER_TRADE

    # Elliott Wave 5 caution: halve the risk per trade
    if signal.wave5_caution:
        risk *= 0.50
        logger.warning(
            f"⚠ Wave 5 caution active for {signal.ticker} — "
            f"risk halved to {risk:.2%} of portfolio"
        )

    stop         = config.CRYPTO_STOP_LOSS_PCT if is_crypto(signal.ticker) else config.STOP_LOSS_PCT
    portfolio    = get_portfolio_value()
    price        = _get_current_price(signal.ticker, order_symbol)

    if price is None:
        logger.error(f"Cannot get price for {signal.ticker} — skipping order")
        _pending_buys.discard(signal.ticker)
        return

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

    if signal.action == 'SELL':
        try:
            pos           = api.get_position(signal.ticker)
            qty_str       = str(float(pos.qty))
            time_in_force = 'gtc'
            logger.info(f"Closing full position: {qty_str} {order_symbol}")
        except Exception as e:
            logger.error(f"Could not fetch position qty for SELL {signal.ticker}: {e}")
            return

    try:
        order = api.submit_order(
            symbol=order_symbol,
            qty=qty_str,
            side=side,
            type='market',
            time_in_force=time_in_force,
        )
        _daily_trades += 1

        trap_tag = ' [TRAP FLIP]' if signal.trap_flip else ''
        w5_tag   = ' [W5 CAUTION]' if signal.wave5_caution else ''
        logger.success(
            f"✅ {signal.action}{trap_tag}{w5_tag} {qty_str} {signal.ticker} "
            f"@ ~${price:.2f} | conf={signal.confidence:.0%} | {signal.reason}"
        )

        if signal.action == 'SELL':
            _pending_buys.discard(signal.ticker)

        if signal.action == 'BUY':
            stop_price = round(price * (1 - stop), 2)
            if is_crypto(signal.ticker):
                logger.warning(
                    f"  Stop-loss skipped for {signal.ticker} "
                    f"(crypto stop orders unsupported) — check_exits() enforces ${stop_price}"
                )
            else:
                api.submit_order(
                    symbol=order_symbol,
                    qty=qty_str,
                    side='sell',
                    type='stop',
                    stop_price=stop_price,
                    time_in_force='gtc',
                )
                logger.info(f"  Stop-loss set at ${stop_price}")

    except Exception as e:
        logger.error(f"Order failed for {signal.ticker}: {e}")
        _pending_buys.discard(signal.ticker)