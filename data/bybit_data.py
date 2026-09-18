# data/bybit_data.py — Bybit v5 public market data.
# ─────────────────────────────────────────────────────────────
# Replaces market_data.py (Alpaca spot) as the source of truth.
#
# Everything here uses PUBLIC endpoints — no API key, no signing. The same
# module therefore serves the backtester, the local simulator and the live
# Demo bot, so all three see byte-identical candles.
#
# Two Bybit quirks this module normalises away, both of which silently
# corrupt a backtest if you get them wrong:
#
#   1. /v5/market/kline returns bars NEWEST-FIRST. We reverse to ascending.
#   2. The newest bar is the one still forming. Indicators computed on a
#      partial bar flicker between cycles and, in a backtest, leak the
#      future. We drop it unless the caller explicitly asks for it.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from typing import Optional

import pandas as pd
import requests
from loguru import logger

from . import cache

BASE_URL = "https://api.bybit.com"
CATEGORY = "linear"          # USDT perpetuals
MAX_LIMIT = 1000             # Bybit's per-request cap for kline/funding

# Bybit interval code → bar duration. Used for pagination and for deciding
# whether the most recent bar has closed yet.
INTERVAL_MS: dict[str, int] = {
    '1':   60_000,
    '3':   3 * 60_000,
    '5':   5 * 60_000,
    '15':  15 * 60_000,
    '30':  30 * 60_000,
    '60':  60 * 60_000,
    '120': 2 * 60 * 60_000,
    '240': 4 * 60 * 60_000,
    '360': 6 * 60 * 60_000,
    '720': 12 * 60 * 60_000,
    'D':   24 * 60 * 60_000,
    'W':   7 * 24 * 60 * 60_000,
}


# ══════════════════════════════════════════════════════════════
# HTTP
# ══════════════════════════════════════════════════════════════

_session = requests.Session()


def _get(endpoint: str, params: dict, retries: int = 4) -> dict:
    """
    GET a public v5 endpoint with retry + exponential backoff.

    Bybit signals application errors through a non-zero retCode in an
    otherwise-200 response, so both transport and protocol failures are
    checked here. Raises RuntimeError once retries are exhausted.
    """
    delay = 0.5
    last_err: Optional[str] = None

    for attempt in range(retries):
        try:
            resp = _session.get(f"{BASE_URL}{endpoint}", params=params, timeout=20)
            resp.raise_for_status()
            payload = resp.json()

            if payload.get('retCode') == 0:
                return payload['result']

            last_err = f"retCode={payload.get('retCode')} {payload.get('retMsg')}"
            # 10006/10018 are rate limits — always worth another go after a wait.
        except Exception as e:
            last_err = str(e)

        if attempt < retries - 1:
            time.sleep(delay)
            delay *= 2

    raise RuntimeError(f"Bybit {endpoint} failed after {retries} attempts: {last_err}")


def _advance(cursor: int, oldest: int) -> Optional[int]:
    """
    Step a backwards-paging cursor, refusing to stand still.

    Bybit's endpoints disagree about time-window parameter names (kline takes
    start/end, funding and open-interest take startTime/endTime) and silently
    ignore the ones they don't recognise. When that happens every page comes
    back identical, the cursor stops moving and the walk spins forever. This
    returns None instead of a non-decreasing cursor so the caller can stop.
    """
    nxt = oldest - 1
    if nxt >= cursor:
        logger.warning(
            f"Pagination stalled at {cursor} (oldest={oldest}) — stopping. "
            f"This usually means a time-window parameter was ignored."
        )
        return None
    return nxt


def _now_ms() -> int:
    return int(time.time() * 1000)


def _days_ago_ms(days: float) -> int:
    return int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)


# ══════════════════════════════════════════════════════════════
# INSTRUMENT METADATA
# ══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Instrument:
    """
    Trading rules for one perpetual contract.

    Bybit rejects orders whose quantity or price isn't an exact multiple of
    the symbol's step size, so every order the bot sends must be quantised
    through here first.
    """
    symbol:        str
    qty_step:      Decimal
    min_qty:       Decimal
    max_qty:       Decimal
    tick_size:     Decimal
    min_notional:  Decimal
    max_leverage:  float

    def round_qty(self, qty: float) -> float:
        """
        Quantise a quantity DOWN to the nearest valid step.

        Always down, never up: rounding up would size the position slightly
        above the risk budget, and repeated across many trades that bias
        compounds against you.
        """
        q = Decimal(str(qty)).quantize(self.qty_step, rounding=ROUND_DOWN)
        if q < self.min_qty:
            return 0.0
        return float(min(q, self.max_qty))

    def round_price(self, price: float) -> float:
        """Quantise a price to the nearest valid tick."""
        p = Decimal(str(price)).quantize(self.tick_size, rounding=ROUND_HALF_UP)
        return float(p)

    def is_tradeable(self, qty: float, price: float) -> bool:
        """True when qty clears both the min-lot and min-notional floors."""
        if qty <= 0:
            return False
        return Decimal(str(qty * price)) >= self.min_notional


_instrument_cache: dict[str, Instrument] = {}


def get_instrument(symbol: str) -> Instrument:
    """Fetch (and memoise) the trading rules for a symbol."""
    if symbol in _instrument_cache:
        return _instrument_cache[symbol]

    result = _get('/v5/market/instruments-info',
                  {'category': CATEGORY, 'symbol': symbol})
    entries = result.get('list') or []
    if not entries:
        raise ValueError(f"Unknown Bybit {CATEGORY} symbol: {symbol}")

    info = entries[0]
    lot = info['lotSizeFilter']
    prc = info['priceFilter']

    instrument = Instrument(
        symbol=symbol,
        qty_step=Decimal(lot['qtyStep']),
        min_qty=Decimal(lot['minOrderQty']),
        max_qty=Decimal(lot['maxOrderQty']),
        tick_size=Decimal(prc['tickSize']),
        min_notional=Decimal(lot.get('minNotionalValue', '0')),
        max_leverage=float(info['leverageFilter']['maxLeverage']),
    )
    _instrument_cache[symbol] = instrument
    logger.debug(
        f"{symbol}: qtyStep={instrument.qty_step} tick={instrument.tick_size} "
        f"minNotional={instrument.min_notional} maxLev={instrument.max_leverage}"
    )
    return instrument


# ══════════════════════════════════════════════════════════════
# KLINES
# ══════════════════════════════════════════════════════════════

def _fetch_klines_raw(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Page backwards through /v5/market/kline until start_ms is reached.

    Bybit caps each response at 1000 bars and returns them newest-first, so
    we walk backwards by moving `end` to just before the oldest bar we got.
    """
    bar_ms = INTERVAL_MS[interval]
    rows: list[list[str]] = []
    cursor = end_ms

    while cursor > start_ms:
        result = _get('/v5/market/kline', {
            'category': CATEGORY,
            'symbol':   symbol,
            'interval': interval,
            'end':      cursor,
            'limit':    MAX_LIMIT,
        })
        batch = result.get('list') or []
        if not batch:
            break

        rows.extend(batch)

        # batch is newest-first, so the last element is the oldest bar.
        oldest = int(batch[-1][0])
        if oldest <= start_ms or len(batch) < MAX_LIMIT:
            break
        cursor = _advance(cursor, oldest)
        if cursor is None:
            break
        time.sleep(0.12)          # stay well inside the public rate limit

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['ts'] = pd.to_datetime(df['ts'].astype('int64'), unit='ms', utc=True)
    for col in ('open', 'high', 'low', 'close', 'volume', 'turnover'):
        df[col] = df[col].astype(float)

    df = df.set_index('ts').sort_index()
    df = df[~df.index.duplicated(keep='last')]
    return df[df.index >= pd.Timestamp(start_ms, unit='ms', tz='UTC')]


def get_klines(symbol: str,
               interval: str = '60',
               days: float = 730,
               use_cache: bool = True,
               include_partial: bool = False) -> pd.DataFrame:
    """
    OHLCV candles for a perpetual, ascending by time, indexed UTC.

    Args:
        symbol:          e.g. 'BTCUSDT'
        interval:        Bybit interval code — '60' is 1H (see INTERVAL_MS)
        days:            how much history to cover
        use_cache:       read/extend the on-disk parquet cache
        include_partial: keep the still-forming final bar. Defaults to False
                         because that bar has not closed — using it leaks the
                         future in a backtest and makes live indicators
                         flicker between cycles.

    Returns:
        DataFrame[open, high, low, close, volume, turnover]
    """
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval {interval!r}; expected one of {sorted(INTERVAL_MS)}")

    bar_ms = INTERVAL_MS[interval]
    want_start = _days_ago_ms(days)
    now = _now_ms()

    path = cache.cache_path('klines', symbol, interval)
    cached = cache.load(path) if use_cache else None

    if cached is not None and not cached.empty:
        cached_start = int(cached.index[0].timestamp() * 1000)
        cached_end = int(cached.index[-1].timestamp() * 1000)

        fresh_parts = []
        # Extend forward from the last cached bar (which may have been partial).
        if now - cached_end > bar_ms:
            fresh_parts.append(_fetch_klines_raw(symbol, interval, cached_end, now))
        # Extend backward if the caller now wants more history than we hold.
        if cached_start - want_start > bar_ms:
            fresh_parts.append(_fetch_klines_raw(symbol, interval, want_start, cached_start))

        df = cached
        for part in fresh_parts:
            df = cache.merge(df, part)
    else:
        df = _fetch_klines_raw(symbol, interval, want_start, now)

    if df.empty:
        logger.warning(f"No klines returned for {symbol} {interval}")
        return df

    if use_cache:
        cache.save(df, path)

    # Drop the bar that hasn't closed yet.
    if not include_partial:
        last_open_ms = int(df.index[-1].timestamp() * 1000)
        if last_open_ms + bar_ms > now:
            df = df.iloc[:-1]

    df = df[df.index >= pd.Timestamp(want_start, unit='ms', tz='UTC')]
    logger.debug(f"{symbol} {interval}: {len(df)} bars {df.index[0]} → {df.index[-1]}")
    return df


# ══════════════════════════════════════════════════════════════
# FUNDING
# ══════════════════════════════════════════════════════════════

def get_funding_history(symbol: str, days: float = 730, use_cache: bool = True) -> pd.DataFrame:
    """
    Historical funding rates, ascending by settlement time.

    Funding is the defining cost of holding a perpetual — it settles every
    8h and a 48h hold pays it up to six times. It is also a genuine signal:
    persistently positive funding means longs are crowded and paying to stay
    that way, which is exactly the condition that precedes long squeezes.

    Returns:
        DataFrame[funding_rate] indexed by settlement timestamp (UTC).
    """
    path = cache.cache_path('funding', symbol)
    cached = cache.load(path) if use_cache else None

    start_ms = _days_ago_ms(days)
    now = _now_ms()
    fetch_from = start_ms
    if cached is not None and not cached.empty:
        cached_end = int(cached.index[-1].timestamp() * 1000)
        # Only re-fetch from the last settlement we already hold.
        fetch_from = max(start_ms, cached_end)

    rows: list[dict] = []
    cursor = now
    while cursor > fetch_from:
        result = _get('/v5/market/funding/history', {
            'category': CATEGORY,
            'symbol':   symbol,
            # NOTE: this endpoint takes startTime/endTime, NOT the start/end
            # that /v5/market/kline uses. Passing `end` here is silently
            # ignored — every page comes back identical and the walk never
            # terminates.
            'endTime':  cursor,
            'limit':    200,          # funding history caps at 200/request
        })
        batch = result.get('list') or []
        if not batch:
            break
        rows.extend(batch)

        oldest = int(batch[-1]['fundingRateTimestamp'])
        if oldest <= fetch_from or len(batch) < 200:
            break
        cursor = _advance(cursor, oldest)
        if cursor is None:
            break
        time.sleep(0.12)

    fresh = pd.DataFrame()
    if rows:
        fresh = pd.DataFrame(rows)
        fresh['ts'] = pd.to_datetime(fresh['fundingRateTimestamp'].astype('int64'), unit='ms', utc=True)
        fresh['funding_rate'] = fresh['fundingRate'].astype(float)
        fresh = fresh[['ts', 'funding_rate']].set_index('ts').sort_index()
        fresh = fresh[~fresh.index.duplicated(keep='last')]

    df = cache.merge(cached, fresh)
    if df.empty:
        logger.warning(f"No funding history for {symbol}")
        return df

    if use_cache:
        cache.save(df, path)

    return df[df.index >= pd.Timestamp(start_ms, unit='ms', tz='UTC')]


# ══════════════════════════════════════════════════════════════
# OPEN INTEREST
# ══════════════════════════════════════════════════════════════

def get_open_interest(symbol: str,
                      interval_time: str = '1h',
                      days: float = 180,
                      use_cache: bool = True) -> pd.DataFrame:
    """
    Open-interest series, ascending by time.

    OI is the number of contracts outstanding. Read alongside price it
    separates real conviction from noise: price up on rising OI is new money
    entering, price up on falling OI is shorts covering — the second is far
    more likely to fade.

    NOTE: Bybit only retains roughly the last 6 months of OI, so this cannot
    cover a 2-year backtest. Callers must treat a missing/short series as
    "no information" rather than assuming zero.

    Returns:
        DataFrame[open_interest] indexed UTC.
    """
    path = cache.cache_path('oi', symbol, interval_time)
    cached = cache.load(path) if use_cache else None

    start_ms = _days_ago_ms(days)
    now = _now_ms()

    rows: list[dict] = []
    cursor = now
    while cursor > start_ms:
        try:
            result = _get('/v5/market/open-interest', {
                'category':     CATEGORY,
                'symbol':       symbol,
                'intervalTime': interval_time,
                'endTime':      cursor,     # startTime/endTime, not start/end
                'limit':        200,
            })
        except RuntimeError as e:
            logger.warning(f"OI fetch stopped for {symbol}: {e}")
            break

        batch = result.get('list') or []
        if not batch:
            break
        rows.extend(batch)

        oldest = int(batch[-1]['timestamp'])
        if oldest <= start_ms or len(batch) < 200:
            break
        cursor = _advance(cursor, oldest)
        if cursor is None:
            break
        time.sleep(0.12)

    fresh = pd.DataFrame()
    if rows:
        fresh = pd.DataFrame(rows)
        fresh['ts'] = pd.to_datetime(fresh['timestamp'].astype('int64'), unit='ms', utc=True)
        fresh['open_interest'] = fresh['openInterest'].astype(float)
        fresh = fresh[['ts', 'open_interest']].set_index('ts').sort_index()
        fresh = fresh[~fresh.index.duplicated(keep='last')]

    df = cache.merge(cached, fresh)
    if df.empty:
        logger.warning(f"No open-interest data for {symbol}")
        return df

    if use_cache:
        cache.save(df, path)

    return df[df.index >= pd.Timestamp(start_ms, unit='ms', tz='UTC')]


# ══════════════════════════════════════════════════════════════
# SPOT PRICE
# ══════════════════════════════════════════════════════════════

def get_clock_offset_ms() -> Optional[float]:
    """
    Local clock minus Bybit's server clock, in milliseconds.

    Positive means the local machine is AHEAD. That direction matters:
    Bybit accepts a request only when

        server_time − recv_window  ≤  req_timestamp  ≤  server_time + 1000

    so recv_window widens the window BACKWARDS only. A clock running more
    than ~1s fast is rejected (retCode 10002) regardless of how large
    recv_window is set — which makes raising it useless in that direction.

    Returns None if the server time can't be read.
    """
    try:
        before = time.time()
        result = _get('/v5/market/time', {})
        after = time.time()

        server_ms = int(result['timeNano']) / 1e6
        # Midpoint of the request brackets out most of the round-trip.
        local_ms = (before + after) / 2 * 1000
        return local_ms - server_ms
    except Exception as e:
        logger.debug(f"Could not read server time: {e}")
        return None


def get_last_price(symbol: str) -> Optional[float]:
    """Latest traded price for a perpetual, or None if the call fails."""
    try:
        result = _get('/v5/market/tickers', {'category': CATEGORY, 'symbol': symbol})
        entries = result.get('list') or []
        return float(entries[0]['lastPrice']) if entries else None
    except Exception as e:
        logger.warning(f"Price fetch error ({symbol}): {e}")
        return None


def get_mark_price(symbol: str) -> Optional[float]:
    """
    Latest MARK price — the one Bybit uses for liquidation and, in our
    config, for stop triggers. It can differ from last price during thin
    or manipulated moves, which is precisely why stops trigger off it.
    """
    try:
        result = _get('/v5/market/tickers', {'category': CATEGORY, 'symbol': symbol})
        entries = result.get('list') or []
        return float(entries[0]['markPrice']) if entries else None
    except Exception as e:
        logger.warning(f"Mark price fetch error ({symbol}): {e}")
        return None
