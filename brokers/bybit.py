# brokers/bybit.py — Bybit v5 USDT-perpetual adapter.
# ─────────────────────────────────────────────────────────────
# Targets Demo Trading by default (api-demo.bybit.com): real mainnet
# prices, spreads and liquidity, but fake money. Keys are minted from the
# Demo Trading section of a MAINNET account — testnet keys will not work.
#
# Account assumptions, enforced in prepare():
#   - one-way mode (positionIdx=0), so a symbol holds at most one position
#   - isolated margin, so one blown trade can't drain the rest of the book
#   - leverage from config, which caps notional but is NOT the risk lever
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

import config
from data import (Instrument, get_clock_offset_ms, get_instrument,
                  get_last_price, get_mark_price)

from . import clock_sync
from .base import Broker, BrokerError, BrokerPosition, OrderResult

try:
    from pybit.unified_trading import HTTP
except ImportError:  # pragma: no cover
    HTTP = None


# Bybit reports "you asked for the setting it already has" as an error.
# These codes mean the account is already in the desired state.
_ALREADY_SET_CODES = {
    110043,   # leverage not modified
    110026,   # margin mode not modified
    34036,    # leverage not modified (unified account)
    34040,    # trading stop not modified (SL/TP already at this value)
    10001,    # position mode not modified (context-dependent)
}


class _PybitToLoguru(logging.Handler):
    """Send pybit's internal chatter to the debug log, not the console."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            return
        # pybit logs every retry ATTEMPT at ERROR, including ones it then
        # recovers from — routine against api-demo. Only surface the ones
        # that aren't followed by a retry.
        level = 'DEBUG' if 'Retrying' in message else 'WARNING'
        logger.log(level, f"[pybit] {message}")


def _route_pybit_logging() -> None:
    """
    Re-point every pybit logger at loguru.

    Must run AFTER an HTTP session exists: pybit calls
    `logging.getLogger(__name__)` — i.e. `pybit._http_manager`, a CHILD of
    `pybit` — and attaches a StreamHandler to it during construction.

    Configuring the parent beforehand therefore does nothing, which is why
    the raw `... - pybit._http_manager - ERROR - ...` lines kept appearing.
    """
    for name, existing in list(logging.Logger.manager.loggerDict.items()):
        if name == 'pybit' or name.startswith('pybit.'):
            if isinstance(existing, logging.Logger):
                existing.handlers = [_PybitToLoguru()]
                existing.propagate = False


def check_clock(max_ahead_ms: float = 500.0) -> tuple[bool, str]:
    """
    Preflight the local clock against Bybit's, before any signed request.

    Bybit accepts a request only when

        server_time − recv_window  ≤  req_timestamp  ≤  server_time + 1000

    The window is asymmetric. Running SLOW is forgiven up to recv_window;
    running FAST is rejected past ~1 second no matter what recv_window says.
    pybit responds to a 10002 by enlarging recv_window and retrying, which
    cannot possibly help in the fast direction — so the failure presents as
    a confusing retry loop ending in a traceback.

    Catching it here turns that into one actionable sentence.
    """
    offset = get_clock_offset_ms()
    if offset is None:
        return True, "could not reach Bybit to check the clock — continuing"

    if offset > max_ahead_ms:
        return False, (
            f"system clock is {offset / 1000:+.1f}s AHEAD of Bybit. "
            f"Signed requests will be rejected (retCode 10002); raising "
            f"recv_window cannot fix a fast clock. Resync the clock."
        )

    if offset < -config.RECV_WINDOW * 0.5:
        return False, (
            f"system clock is {offset / 1000:+.1f}s BEHIND Bybit, close to the "
            f"{config.RECV_WINDOW / 1000:.0f}s recv_window. Resync the clock."
        )

    return True, f"clock offset {offset:+.0f} ms — OK"


class BybitFuturesBroker(Broker):
    """Live/demo execution against Bybit v5 linear perpetuals."""

    def __init__(self,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 demo: Optional[bool] = None,
                 testnet: Optional[bool] = None):
        if HTTP is None:
            raise ImportError("pybit is required for live trading: pip install pybit")

        api_key = api_key or config.BYBIT_API_KEY
        api_secret = api_secret or config.BYBIT_API_SECRET
        if not api_key or not api_secret:
            raise ValueError(
                "BYBIT_API_KEY / BYBIT_API_SECRET missing. Create them in the "
                "Demo Trading section of your Bybit account and add them to .env"
            )

        # Timestamp signed requests against Bybit's clock rather than the
        # local one. Must happen BEFORE the session is built so the very
        # first call is already corrected.
        clock_sync.install()

        self.demo = config.BYBIT_DEMO if demo is None else demo
        self.testnet = config.BYBIT_TESTNET if testnet is None else testnet

        self.session = HTTP(
            testnet=self.testnet,
            demo=self.demo,
            api_key=api_key,
            api_secret=api_secret,
            # pybit defaults to a 10s timeout and no retries. The demo
            # endpoint is noticeably slower than mainnet and read timeouts
            # are routine, so give it room and let transient failures retry
            # before they surface as an error.
            timeout=config.HTTP_TIMEOUT,
            max_retries=config.HTTP_MAX_RETRIES,
            retry_delay=config.HTTP_RETRY_DELAY,
            force_retry=True,
            # Tolerate a few seconds of local clock drift (retCode 10002).
            recv_window=config.RECV_WINDOW,
        )
        # pybit attaches its own stderr handler to `pybit._http_manager` when
        # the session is built, so this has to happen AFTER construction.
        _route_pybit_logging()

        env = 'DEMO' if self.demo else ('TESTNET' if self.testnet else 'LIVE')
        logger.info(f"Bybit broker connected — {env} ({config.CATEGORY})")

    # ── internals ─────────────────────────────────────────────
    @staticmethod
    def _ok(response: dict) -> bool:
        return response.get('retCode') == 0

    @staticmethod
    def _benign(response: dict) -> bool:
        """True when an 'error' just means the setting was already correct."""
        return response.get('retCode') in _ALREADY_SET_CODES

    def _call(self, fn, **kwargs) -> tuple[bool, dict]:
        """Invoke a pybit method, normalising exceptions into (ok, payload)."""
        try:
            response = fn(**kwargs)
            if self._ok(response):
                return True, response.get('result', {}) or {}
            logger.warning(f"Bybit {fn.__name__} → {response.get('retCode')} "
                           f"{response.get('retMsg')}")
            return False, response
        except Exception as e:
            message = str(e)
            # pybit RAISES on a non-zero retCode rather than returning it, so
            # "already in the desired state" responses land here and never
            # reach _benign(). Without this they are logged as errors.
            if any(str(code) in message for code in _ALREADY_SET_CODES):
                logger.debug(f"Bybit {fn.__name__}: {message}")
                return True, {}
            logger.error(f"Bybit {fn.__name__} raised: {message}")
            return False, {'retMsg': message}

    # ── account ───────────────────────────────────────────────
    def get_equity(self) -> float:
        ok, result = self._call(self.session.get_wallet_balance,
                                accountType='UNIFIED',
                                coin=config.QUOTE_CURRENCY)
        if not ok:
            return 0.0
        accounts = result.get('list') or []
        if not accounts:
            return 0.0

        account = accounts[0]
        # totalEquity covers the whole unified account; fall back to the
        # per-coin figure on accounts where it isn't populated.
        total = account.get('totalEquity')
        if total not in (None, ''):
            return float(total)
        for coin in account.get('coin', []):
            if coin.get('coin') == config.QUOTE_CURRENCY:
                return float(coin.get('equity') or 0.0)
        return 0.0

    def get_available_margin(self) -> float:
        ok, result = self._call(self.session.get_wallet_balance,
                                accountType='UNIFIED',
                                coin=config.QUOTE_CURRENCY)
        if not ok:
            return 0.0
        accounts = result.get('list') or []
        if not accounts:
            return 0.0
        account = accounts[0]
        for key in ('totalAvailableBalance', 'totalMarginBalance'):
            value = account.get(key)
            if value not in (None, ''):
                return float(value)
        return 0.0

    # ── positions ─────────────────────────────────────────────
    def get_positions(self) -> dict[str, BrokerPosition]:
        ok, result = self._call(self.session.get_positions,
                                category=config.CATEGORY,
                                settleCoin=config.QUOTE_CURRENCY)
        if not ok:
            # Deliberately NOT an empty dict. Returning {} here would tell
            # reconciliation that every open position had been closed, and a
            # single read timeout would then wipe tracking state and allow a
            # duplicate entry into a position still held at the exchange.
            raise BrokerError(f"could not read positions: {result.get('retMsg', result)}")

        positions: dict[str, BrokerPosition] = {}
        for entry in result.get('list') or []:
            size = float(entry.get('size') or 0)
            if size == 0:
                continue    # Bybit lists flat symbols with size 0

            symbol = entry['symbol']
            created = entry.get('createdTime')
            positions[symbol] = BrokerPosition(
                symbol=symbol,
                side='long' if entry.get('side') == 'Buy' else 'short',
                qty=size,
                entry_price=float(entry.get('avgPrice') or 0),
                mark_price=float(entry.get('markPrice') or 0),
                unrealised_pnl=float(entry.get('unrealisedPnl') or 0),
                leverage=float(entry.get('leverage') or 1),
                stop_loss=self._opt_float(entry.get('stopLoss')),
                take_profit=self._opt_float(entry.get('takeProfit')),
                created_at=(datetime.fromtimestamp(int(created) / 1000, tz=timezone.utc)
                            if created else None),
            )
        return positions

    @staticmethod
    def _opt_float(value) -> Optional[float]:
        """Bybit sends '' or '0' for 'no stop set'."""
        if value in (None, '', '0', '0.0'):
            return None
        try:
            parsed = float(value)
            return parsed if parsed != 0 else None
        except (TypeError, ValueError):
            return None

    # ── trading ───────────────────────────────────────────────
    def open_position(self,
                      symbol: str,
                      side: str,
                      qty: float,
                      stop_loss: Optional[float] = None,
                      take_profit: Optional[float] = None) -> OrderResult:
        instrument = self.get_instrument(symbol)
        qty = instrument.round_qty(qty)
        if qty <= 0:
            return OrderResult(False, error=f"qty below {symbol} min lot {instrument.min_qty}")

        price = self.get_price(symbol)
        if price and not instrument.is_tradeable(qty, price):
            return OrderResult(
                False,
                error=f"notional {qty * price:.2f} below {symbol} min {instrument.min_notional}")

        params: dict = {
            'category':    config.CATEGORY,
            'symbol':      symbol,
            'side':        'Buy' if side == 'long' else 'Sell',
            'orderType':   'Market',
            'qty':         str(qty),
            'timeInForce': 'IOC',       # market orders are always IOC
            'positionIdx': 0,           # one-way mode
        }

        # Attach protection to the entry itself. Triggering off mark price
        # rather than last price stops a thin-book wick from taking us out.
        if stop_loss is not None:
            params['stopLoss'] = str(instrument.round_price(stop_loss))
            params['slTriggerBy'] = config.TRIGGER_PRICE_TYPE
        if take_profit is not None:
            params['takeProfit'] = str(instrument.round_price(take_profit))
            params['tpTriggerBy'] = config.TRIGGER_PRICE_TYPE
        if stop_loss is not None or take_profit is not None:
            params['tpslMode'] = 'Full'

        ok, result = self._call(self.session.place_order, **params)
        if not ok:
            return OrderResult(False, error=str(result.get('retMsg', result)))

        logger.success(
            f"✅ OPEN {side.upper()} {qty} {symbol} @ ~{price} | "
            f"SL={params.get('stopLoss', '—')} TP={params.get('takeProfit', '—')}"
        )
        return OrderResult(True, order_id=result.get('orderId'), filled_price=price)

    def close_position(self, symbol: str, qty: Optional[float] = None) -> OrderResult:
        position = self.get_position(symbol)
        if position is None:
            return OrderResult(False, error=f"no open position for {symbol}")

        instrument = self.get_instrument(symbol)
        close_qty = instrument.round_qty(qty if qty is not None else position.qty)
        if close_qty <= 0:
            return OrderResult(False, error="close qty rounds to zero")

        # reduceOnly guarantees this can only shrink the position — it can
        # never accidentally flip us short. Bybit rejects TP/SL alongside it.
        ok, result = self._call(
            self.session.place_order,
            category=config.CATEGORY,
            symbol=symbol,
            side='Sell' if position.side == 'long' else 'Buy',
            orderType='Market',
            qty=str(close_qty),
            timeInForce='IOC',
            positionIdx=0,
            reduceOnly=True,
        )
        if not ok:
            return OrderResult(False, error=str(result.get('retMsg', result)))

        logger.success(f"✅ CLOSE {close_qty} {symbol} ({position.side})")
        return OrderResult(True, order_id=result.get('orderId'))

    def set_stops(self,
                  symbol: str,
                  stop_loss: Optional[float] = None,
                  take_profit: Optional[float] = None) -> OrderResult:
        if stop_loss is None and take_profit is None:
            return OrderResult(True)

        instrument = self.get_instrument(symbol)
        params: dict = {
            'category':    config.CATEGORY,
            'symbol':      symbol,
            'positionIdx': 0,
            'tpslMode':    'Full',
        }
        if stop_loss is not None:
            params['stopLoss'] = str(instrument.round_price(stop_loss))
            params['slTriggerBy'] = config.TRIGGER_PRICE_TYPE
        if take_profit is not None:
            params['takeProfit'] = str(instrument.round_price(take_profit))
            params['tpTriggerBy'] = config.TRIGGER_PRICE_TYPE

        ok, result = self._call(self.session.set_trading_stop, **params)
        if not ok and not self._benign(result):
            return OrderResult(False, error=str(result.get('retMsg', result)))
        return OrderResult(True)

    # ── venue metadata ────────────────────────────────────────
    def set_leverage(self, symbol: str, leverage: float) -> bool:
        try:
            response = self.session.set_leverage(
                category=config.CATEGORY,
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage),
            )
            return self._ok(response) or self._benign(response)
        except Exception as e:
            # pybit raises on non-zero retCode; "not modified" is success.
            if any(str(code) in str(e) for code in _ALREADY_SET_CODES):
                return True
            logger.warning(f"set_leverage({symbol}) failed: {e}")
            return False

    def get_instrument(self, symbol: str) -> Instrument:
        return get_instrument(symbol)

    def get_price(self, symbol: str) -> Optional[float]:
        return get_last_price(symbol)

    def get_mark_price(self, symbol: str) -> Optional[float]:
        return get_mark_price(symbol)

    def get_last_exit(self, symbol: str,
                      since: datetime) -> Optional[tuple[float, datetime]]:
        # Demo only retains ~7 days of closed PnL; outside that we fall back.
        ok, result = self._call(self.session.get_closed_pnl,
                                category=config.CATEGORY, symbol=symbol, limit=10)
        if not ok:
            return None
        since_ms = since.timestamp() * 1000
        rows = [r for r in (result.get('list') or [])
                if int(r.get('updatedTime') or 0) >= since_ms]
        if not rows:
            return None
        latest = max(rows, key=lambda r: int(r['updatedTime']))
        return (float(latest['avgExitPrice']),
                datetime.fromtimestamp(int(latest['updatedTime']) / 1000, tz=timezone.utc))

    # ── lifecycle ─────────────────────────────────────────────
    def prepare(self, symbols: list[str]) -> None:
        """Put the account into the state the risk model assumes."""
        for symbol in symbols:
            try:
                self.session.switch_margin_mode(
                    category=config.CATEGORY,
                    symbol=symbol,
                    tradeMode=1 if config.MARGIN_MODE == 'ISOLATED' else 0,
                    buyLeverage=str(config.LEVERAGE),
                    sellLeverage=str(config.LEVERAGE),
                )
            except Exception as e:
                # Already-set is the common case and is not a failure.
                if not any(str(code) in str(e) for code in _ALREADY_SET_CODES):
                    logger.debug(f"switch_margin_mode({symbol}): {e}")

            if self.set_leverage(symbol, config.LEVERAGE):
                logger.info(f"{symbol}: leverage {config.LEVERAGE}× "
                            f"({config.MARGIN_MODE.lower()} margin)")
            else:
                logger.warning(f"{symbol}: could not set leverage — using account default")

    def request_demo_funds(self, amount: float = 100_000) -> bool:
        """
        Top up the Demo account (USDT). Rate-limited to 1 request/minute.
        Only meaningful on the demo endpoint.
        """
        if not self.demo:
            logger.warning("request_demo_funds only applies to Demo Trading")
            return False
        try:
            response = self.session._submit_request(
                method="POST",
                path=f"{self.session.endpoint}/v5/account/demo-apply-money",
                query={"adjustType": 0,
                       "utaDemoApplyMoney": [{"coin": "USDT", "amountStr": str(amount)}]},
                auth=True,
            )
            ok = self._ok(response)
            logger.info(f"Demo funds request: {'OK' if ok else response.get('retMsg')}")
            return ok
        except Exception as e:
            logger.warning(f"Demo funds request failed: {e}")
            return False
