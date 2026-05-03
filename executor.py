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
import json
import os

STATE_FILE = "logs/executor_state.json"
import pandas as pd

_pending_buys: set[str] = set()

# v4: tracks entry timestamp per ticker for time-based stops
_entry_times: dict[str, datetime] = {}

# v4: trailing stop state — highest price seen since entry per ticker
_trailing_highs: dict[str, float] = {}

# v5: tracks symbols for which a TP/SL/time/trail sell has already been
# submitted but not yet confirmed filled. Persisted across restarts so a
# crash mid-exit doesn't cause a duplicate exit on the next boot.
_pending_exits: set[str] = set()

# v5: pyramid storm guard — last time we scaled into each ticker.
# Prevents scale_in_winner from firing every 20s while the cached
# add_to_winner=True signal is still alive.
_last_pyramid: dict[str, datetime] = {}

# v5: loss-streak cooldown state.
# _recent_outcomes: ticker → most-recent-first list of 'win'/'loss'.
# _cooldown_until:  ticker → datetime (UTC) before which BUYs are blocked.
_recent_outcomes: dict[str, list[str]] = {}
_cooldown_until:  dict[str, datetime]   = {}

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

def _save_executor_state():
    """Persists entry times, trailing highs, pending exits, pyramid timing,
    recent outcomes, and cooldowns to disk. (v5: pending_exits + cooldown
    state added so a crash mid-exit doesn't double-fire on restart.)"""
    try:
        # Convert datetime objects to ISO strings for JSON compatibility
        serializable_entries  = {k: v.isoformat() for k, v in _entry_times.items()}
        serializable_pyramid  = {k: v.isoformat() for k, v in _last_pyramid.items()}
        serializable_cooldown = {k: v.isoformat() for k, v in _cooldown_until.items()}
        with open(STATE_FILE, 'w') as f:
            json.dump({
                "entry_times":      serializable_entries,
                "trailing_highs":   _trailing_highs,
                "pending_exits":    list(_pending_exits),       # v5
                "last_pyramid":     serializable_pyramid,        # v5
                "recent_outcomes":  _recent_outcomes,            # v5
                "cooldown_until":   serializable_cooldown,       # v5
            }, f)
    except Exception as e:
        logger.warning(f"Failed to save executor state: {e}")

def _load_executor_state():
    """Restores state from disk on startup. (v5: also restores pending_exits,
    pyramid timing, outcomes, cooldowns.)"""
    global _entry_times, _trailing_highs, _pending_exits
    global _last_pyramid, _recent_outcomes, _cooldown_until
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)
                _entry_times = {
                    k: datetime.fromisoformat(v)
                    for k, v in data.get("entry_times", {}).items()
                }
                _trailing_highs = data.get("trailing_highs", {})
                _pending_exits  = set(data.get("pending_exits", []))
                _last_pyramid = {
                    k: datetime.fromisoformat(v)
                    for k, v in data.get("last_pyramid", {}).items()
                }
                _recent_outcomes = data.get("recent_outcomes", {})
                _cooldown_until = {
                    k: datetime.fromisoformat(v)
                    for k, v in data.get("cooldown_until", {}).items()
                }
            logger.info(
                f"Restored state: {len(_entry_times)} active trades, "
                f"{len(_pending_exits)} pending exits, "
                f"{len(_cooldown_until)} cooldowns."
            )
        except Exception as e:
            logger.warning(f"Could not load executor state: {e}")

_load_executor_state()

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
    if is_crypto(ticker) and '/' not in order_symbol:
        order_symbol = to_order_symbol(ticker)

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


_atr_cache: dict[str, tuple[float, float]] = {}  # ticker -> (timestamp, value)


def _get_atr_dollars(ticker: str) -> float:
    """Fetch ATR with a 10-minute cache to reduce API latency."""
    import time
    now = time.time()
    if ticker in _atr_cache:
        ts, val = _atr_cache[ticker]
        if (now - ts) < 600:  # 10 minute cache
            return val

    try:
        from market_data import fetch_bars
        from alpaca.data.timeframe import TimeFrame
        from indicators import atr as calc_atr

        df = fetch_bars(ticker, TimeFrame.Hour, limit=20)
        if df.empty or len(df) < 15:
            return 0.0

        result = calc_atr(df, period=config.ATR_PERIOD)
        val = result.value if result is not None else 0.0
        _atr_cache[ticker] = (now, val)
        return val
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

    v5 FIX: now respects PYRAMID_MIN_INTERVAL_SEC. Previously this could fire
    every 20 s while the cached add_to_winner=True signal was alive, turning
    a 50% add into a 600% add inside one analysis cycle. Now it refuses to
    re-pyramid the same ticker more than once per analysis cycle.
    """
    global _daily_trades
    _reset_daily_counter_if_needed()

    # ── v5: pyramid storm guard ───────────────────────────────
    last = _last_pyramid.get(ticker)
    if last is not None:
        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
        if elapsed < config.PYRAMID_MIN_INTERVAL_SEC:
            logger.debug(
                f"Scale-in skipped for {ticker} — last pyramid was {elapsed:.0f}s "
                f"ago (< {config.PYRAMID_MIN_INTERVAL_SEC}s minimum interval)"
            )
            return

    limit = config.MAX_DAILY_TRADES_CRYPTO
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
        _last_pyramid[ticker] = datetime.now(timezone.utc)   # v5: storm guard
        _save_executor_state()
        logger.success(
            f"📈 SCALE-IN {qty_str} {ticker} @ ~${current_price:.2f} | "
            f"unrealized={unrealized_pct:.1%} | "
            f"conf={signal.confidence:.0%} | Bellafiore pyramiding"
        )
    except Exception as e:
        logger.error(f"Scale-in order failed for {ticker}: {e}")


# _pending_exits is declared at the top of this module (see v5 state block).
# It's persisted across restarts via _save_executor_state / _load_executor_state.


def check_exits():
    """Manually enforce take-profit, stop-loss, time stop, and trailing stop for crypto positions."""
    global _pending_exits

    try:
        positions = api.list_positions()
    except Exception as e:
        logger.warning(f"Could not fetch positions for exit check: {e}")
        return

    # Clear stale pending-exit locks for symbols no longer held
    current_symbols = {p.symbol for p in positions}
    stale_count     = len(_pending_exits) - len(_pending_exits & current_symbols)
    _pending_exits  = {s for s in _pending_exits if s in current_symbols}
    if stale_count > 0:
        # A pending exit symbol disappeared from positions → it was filled.
        # Persist the reduced set so a restart doesn't try to re-exit a closed position.
        _save_executor_state()

    # v5: sweep orphaned tracking state. When a SELL signal fills (handled via
    # check_exits no longer seeing the position) we need to release the
    # entry_times / trailing_highs / last_pyramid records for that ticker.
    # Build the set of internal tickers that currently have positions.
    open_internal = set()
    for p in positions:
        internal = next(
            (k for k, v in config.ALPACA_ORDER_SYMBOL.items() if v == p.symbol),
            p.symbol,
        )
        open_internal.add(internal)

    orphans = (set(_entry_times.keys()) - open_internal) | \
              (set(_trailing_highs.keys()) - open_internal) | \
              (set(_last_pyramid.keys()) - open_internal)
    if orphans:
        for tk in orphans:
            logger.debug(f"check_exits: cleaning orphaned tracking state for {tk}")
            _entry_times.pop(tk, None)
            _trailing_highs.pop(tk, None)
            _last_pyramid.pop(tk, None)
            _pending_buys.discard(tk)
        _save_executor_state()

    for pos in positions:
        symbol         = pos.symbol
        unrealized_pct = float(pos.unrealized_plpc)

        internal = next(
            (k for k, v in config.ALPACA_ORDER_SYMBOL.items() if v == symbol),
            symbol,
        )

        take_profit = config.CRYPTO_TAKE_PROFIT_PCT
        stop = config.CRYPTO_STOP_LOSS_PCT

        # ── Fetch current price (needed for trailing stop) ────
        price_symbol = to_order_symbol(internal)
        current_price = _get_current_price(internal, symbol)

        # ── Update trailing high watermark ────────────────────
        # Every time check_exits() fires, record the highest price
        # seen since entry. This is what the trailing stop tracks.
        if current_price is not None:
            if internal not in _trailing_highs or current_price > _trailing_highs[internal]:
                _trailing_highs[internal] = current_price
                _save_executor_state()

        # ── Time-based stop ───────────────────────────────────
        if internal in _entry_times:
            hours_held = (datetime.now(timezone.utc) - _entry_times[internal]).total_seconds() / 3600
            if hours_held >= config.TIME_STOP_HOURS:
                if symbol not in _pending_exits:
                    logger.warning(
                        f"⏱ TIME STOP triggered for {symbol} — "
                        f"held {hours_held:.1f}h >= {config.TIME_STOP_HOURS}h limit | "
                        f"P&L={unrealized_pct:.1%}"
                    )
                    try:
                        qty_clean = str(float(pos.qty))
                        api.submit_order(symbol=symbol, qty=qty_clean, side='sell',
                                         type='market', time_in_force='gtc')
                        _pending_exits.add(symbol)
                        _pending_buys.discard(internal)
                        _entry_times.pop(internal, None)      # clean up entry time
                        _trailing_highs.pop(internal, None)   # clean up trailing high
                        _save_executor_state()                # v5: persist pending exit
                    except Exception as e:
                        logger.error(f"Time stop order failed for {symbol}: {e}")
                continue  # skip the other checks for this position this cycle

        # ── Trailing stop ─────────────────────────────────────
        if current_price is not None and internal in _trailing_highs:
            atr_dollars = _get_atr_dollars(internal)
            if atr_dollars > 0:
                # Trail distance = 2.0 × ATR below the highest price seen
                trail_distance = config.TRAILING_ATR_MULTIPLIER * atr_dollars
                trail_stop_price = _trailing_highs[internal] - trail_distance

                if current_price <= trail_stop_price:
                    if symbol not in _pending_exits:
                        logger.warning(
                            f"📉 TRAILING STOP triggered for {symbol} — "
                            f"price ${current_price:.2f} crossed below trail stop "
                            f"${trail_stop_price:.2f} (high=${_trailing_highs[internal]:.2f}, "
                            f"ATR=${atr_dollars:.2f})"
                        )
                        try:
                            qty_clean = str(float(pos.qty))
                            api.submit_order(symbol=symbol, qty=qty_clean, side='sell',
                                             type='market', time_in_force='gtc')
                            _pending_exits.add(symbol)
                            _pending_buys.discard(internal)
                            _entry_times.pop(internal, None)
                            _trailing_highs.pop(internal, None)
                            _save_executor_state()            # v5: persist pending exit
                        except Exception as e:
                            logger.error(f"Trailing stop order failed for {symbol}: {e}")
                    continue  # skip fixed TP/SL checks this cycle

        # ── Fixed Take-profit ─────────────────────────────────
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
                _entry_times.pop(internal, None)
                _trailing_highs.pop(internal, None)
                _save_executor_state()                        # v5: persist pending exit
            except Exception as e:
                logger.error(f"Take-profit order failed for {symbol}: {e}")

        # ── Fixed Stop-loss ───────────────────────────────────
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
                _entry_times.pop(internal, None)
                _trailing_highs.pop(internal, None)
                _save_executor_state()                        # v5: persist pending exit
            except Exception as e:
                logger.error(f"Stop-loss order failed for {symbol}: {e}")


def execute_signal(signal: Signal):
    """Execute a trade signal with full risk management checks."""
    global _daily_trades
    _reset_daily_counter_if_needed()

    if signal.action == 'HOLD':
        logger.info(f"HOLD {signal.ticker} — {signal.reason}")
        return

    order_symbol = to_order_symbol(signal.ticker)

    # ── SELL PATH ─────────────────────────────────────────────
    if signal.action == 'SELL':
        if not already_in_position(signal.ticker):
            return

        # v5 FIX: don't clear tracking state until the order is confirmed
        # filled (handled by check_exits when the position disappears from
        # Alpaca's position list). Clearing it here was a race: a partial
        # fill or rejection would leave us tracking nothing while still
        # holding inventory.
        try:
            pos = api.get_position(signal.ticker)
            qty_str = str(float(pos.qty))

            api.submit_order(
                symbol=order_symbol, qty=qty_str, side='sell',
                type='market', time_in_force='gtc'
            )

            # Mark this symbol as having a pending exit so check_exits won't
            # double-fire. The actual state cleanup (entry_times, trailing_highs)
            # happens when check_exits sees the position is gone.
            _pending_exits.add(order_symbol)
            _save_executor_state()

            logger.success(
                f"✅ SELL {qty_str} {signal.ticker} — technical exit: {signal.reason}"
            )
            return
        except Exception as e:
            logger.error(f"SELL order failed: {e}")
            return

    # ── BUY PATH ──────────────────────────────────────────────
    limit = config.MAX_DAILY_TRADES_CRYPTO
    if _daily_trades >= limit:
        logger.warning(f"Daily limit reached — skipping BUY for {signal.ticker}")
        return

    # v5: loss-streak cooldown — block fresh BUYs when this ticker is on ice.
    # Pyramid signals (add_to_winner=True) bypass the cooldown because they
    # only act on already-open positions; the streak is about new entries.
    if not signal.add_to_winner and is_in_cooldown(signal.ticker):
        until = _cooldown_until.get(signal.ticker)
        logger.warning(
            f"🧊 COOLDOWN: BUY for {signal.ticker} blocked — "
            f"{config.LOSS_STREAK_THRESHOLD} consecutive losses, "
            f"frozen until {until.isoformat() if until else 'unknown'}"
        )
        return

    if already_in_position(signal.ticker):
        if signal.add_to_winner:
            scale_in_winner(signal.ticker, signal)
        return

    if len(get_open_positions()) >= config.MAX_OPEN_POSITIONS:
        logger.warning("Max positions reached — skipping BUY")
        return

    if signal.ticker in _pending_buys or has_open_or_pending_buy_order(signal.ticker):
        _pending_buys.add(signal.ticker)
        return

    # Sizing
    risk = config.CRYPTO_RISK_PER_TRADE
    if signal.wave5_caution:
        risk *= 0.50
        logger.warning(f"⚠ Wave 5 caution — risk reduced to {risk:.2%}")

    price = _get_current_price(signal.ticker, order_symbol)
    if price is None:
        return

    portfolio = get_portfolio_value()
    dollars = portfolio * risk * signal.confidence
    qty = round(dollars / price, 6) if is_crypto(signal.ticker) else int(dollars / price)

    if qty <= 0:
        logger.info(f"BUY skipped for {signal.ticker} — qty rounds to 0 (very low confidence)")
        return

    try:
        _pending_buys.add(signal.ticker)
        api.submit_order(symbol=order_symbol, qty=str(qty), side='buy', type='market', time_in_force='gtc')

        # Track for v4 stops
        _entry_times[signal.ticker] = datetime.now(timezone.utc)
        _trailing_highs[signal.ticker] = price
        _daily_trades += 1
        _save_executor_state()

        logger.success(f"✅ BUY {qty} {signal.ticker} @ ~${price:.2f} | {signal.reason}")
    except Exception as e:
        logger.error(f"Order failed: {e}")
        _pending_buys.discard(signal.ticker)


# ══════════════════════════════════════════════════════════════
# v5: LOSS-STREAK COOLDOWN — public API used by EdgeTracker + signal_generator
# ══════════════════════════════════════════════════════════════

def register_trade_outcome(ticker: str, outcome: str):
    """
    Called from EdgeTracker.record_exit when a trade closes.
    Tracks per-ticker recent outcomes; if N consecutive losses are seen,
    sets a cooldown timer that blocks new BUYs on this ticker.

    Args:
        ticker:  internal ticker e.g. 'BTCUSD'
        outcome: 'win' or 'loss'
    """
    history = _recent_outcomes.setdefault(ticker, [])
    history.insert(0, outcome)                   # most recent first
    del history[10:]                             # keep last 10 only

    if outcome == 'loss':
        # Count leading losses
        streak = 0
        for o in history:
            if o == 'loss':
                streak += 1
            else:
                break

        if streak >= config.LOSS_STREAK_THRESHOLD:
            until = datetime.now(timezone.utc) + \
                    pd.Timedelta(hours=config.LOSS_STREAK_COOLDOWN_HRS)
            _cooldown_until[ticker] = until
            logger.warning(
                f"🧊 COOLDOWN ENGAGED for {ticker}: {streak} consecutive losses "
                f"→ no new BUYs until {until.strftime('%Y-%m-%d %H:%M UTC')}"
            )

    elif outcome == 'win':
        # A win clears the cooldown on this ticker
        if ticker in _cooldown_until:
            logger.info(f"❄ COOLDOWN CLEARED for {ticker} — winning trade resets the streak")
            _cooldown_until.pop(ticker, None)

    _save_executor_state()


def is_in_cooldown(ticker: str) -> bool:
    """Return True if this ticker is currently frozen by a loss streak."""
    until = _cooldown_until.get(ticker)
    if until is None:
        return False
    if datetime.now(timezone.utc) >= until:
        # Cooldown expired — clean up
        _cooldown_until.pop(ticker, None)
        _save_executor_state()
        return False
    return True


def clear_tracking_state(ticker: str):
    """Resets v4 + v5 trade-state. Call when a trade is fully closed (position
    no longer in Alpaca's position list)."""
    _entry_times.pop(ticker, None)
    _trailing_highs.pop(ticker, None)
    _last_pyramid.pop(ticker, None)         # v5
    _pending_buys.discard(ticker)
    _save_executor_state()