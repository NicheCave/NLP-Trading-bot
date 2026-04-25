# market_data.py — OHLCV bar fetcher via Alpaca (replaces data_fetcher.py)
# ─────────────────────────────────────────────────────────────
# Fetches candlestick data at multiple timeframes for TA.
# Crypto only. No NLP, no news APIs.
# ─────────────────────────────────────────────────────────────
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestBarRequest
from alpaca.data.timeframe import TimeFrame
from loguru import logger
from datetime import datetime, timedelta, timezone
import pandas as pd
import config

_client = CryptoHistoricalDataClient(
    api_key=config.ALPACA_API_KEY,
    secret_key=config.ALPACA_SECRET_KEY,
)


def _to_alpaca_symbol(ticker: str) -> str:
    """Convert internal ticker (BTCUSD) to Alpaca bar format (BTC/USD)."""
    return config.ALPACA_ORDER_SYMBOL.get(ticker, ticker)


def fetch_bars(ticker: str, timeframe: TimeFrame, limit: int = 200) -> pd.DataFrame:
    """
    Fetch OHLCV candlesticks for a crypto ticker.

    Args:
        ticker:    Internal ticker e.g. 'BTCUSD'
        timeframe: alpaca TimeFrame — e.g. TimeFrame.Minute, TimeFrame.Hour, TimeFrame.Day
        limit:     Number of bars to fetch (max 1000)

    Returns:
        DataFrame with columns: open, high, low, close, volume, vwap
        Indexed by timestamp (UTC). Returns empty DataFrame on error.
    """
    symbol = _to_alpaca_symbol(ticker)
    # Alpaca requires a start time — fetch enough history to cover `limit` bars
    # Use a generous window so we always get enough data even on weekends/gaps
    days_back = max(limit // 24 + 5, 7)  # rough heuristic — overestimate then slice
    start = datetime.now(timezone.utc) - timedelta(days=days_back)

    try:
        req = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
        )
        bars = _client.get_crypto_bars(req)
        df = bars.df

        if df.empty:
            logger.warning(f"No bars returned for {ticker} ({timeframe})")
            return pd.DataFrame()

        # Multi-symbol response wraps in MultiIndex — flatten
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level='symbol')

        df = df.sort_index()
        df = df.tail(limit)  # keep only `limit` most recent bars

        df.columns = [c.lower() for c in df.columns]
        logger.debug(f"Fetched {len(df)} {timeframe} bars for {ticker}")
        return df

    except Exception as e:
        logger.warning(f"Bar fetch error ({ticker}, {timeframe}): {e}")
        return pd.DataFrame()


def fetch_multi_timeframe(ticker: str) -> dict[str, pd.DataFrame]:
    """
    Fetch bars at three timeframes simultaneously — the backbone of multi-TF analysis.

    Returns:
        {
          '1m':  last 100 1-minute bars,
          '1h':  last 200 1-hour bars,
          '1d':  last 200 daily bars,
        }
    All may be empty DataFrames if the request fails.
    """
    return {
        '1m':  fetch_bars(ticker, TimeFrame.Minute, limit=100),
        '1h':  fetch_bars(ticker, TimeFrame.Hour,   limit=200),
        '1d':  fetch_bars(ticker, TimeFrame.Day,    limit=200),
    }


def get_current_price(ticker: str) -> float | None:
    """Fetch the latest trade price for a crypto ticker."""
    symbol = _to_alpaca_symbol(ticker)
    try:
        req = CryptoLatestBarRequest(symbol_or_symbols=symbol)
        bar = _client.get_crypto_latest_bar(req)
        return float(bar[symbol].close)
    except Exception as e:
        logger.warning(f"Price fetch error ({ticker}): {e}")
        return None
