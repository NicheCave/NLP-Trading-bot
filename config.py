# config.py — central configuration file
# ─────────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv

load_dotenv()  # reads your .env file into environment variables

# ── API Credentials ──────────────────────────────────────────
ALPACA_API_KEY    = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL   = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
NEWS_API_KEY      = os.getenv('NEWS_API_KEY')
FINNHUB_ARTICLES = 20
ALPACA_ORDER_SYMBOL = {
    'BTCUSD': 'BTC/USD',
    'ETHUSD': 'ETH/USD',
    'SOLUSD': 'SOL/USD',
    'DOGEUSD': 'DOGE/USD',
}

# ── Watchlist ─────────────────────────────────────────────────
_raw = os.getenv('WATCHLIST', 'TSLA,NVDA,AAPL')
WATCHLIST = [t.strip() for t in _raw.split(',')]

# ── Strategy Parameters ───────────────────────────────────────
BUY_THRESHOLD    = 0.65   # 65% of articles must be bullish to trigger BUY
SELL_THRESHOLD   = 0.60   # 60% of articles must be bearish to trigger SELL
MIN_ARTICLES     = 8      # minimum articles required before making a decision
MIN_AVG_SCORE    = 0.18   # minimum average sentiment strength to confirm signal
MIN_CONFIDENCE   = 0.70   # FinBERT confidence threshold — ignore weak predictions
NEWS_ARTICLES    = 15     # number of news headlines to fetch per ticker

# ── Risk Management ───────────────────────────────────────────
RISK_PER_TRADE   = 0.02   # max 2% of portfolio per trade
STOP_LOSS_PCT    = 0.03   # stop-loss at -3% below entry
TAKE_PROFIT_PCT  = 0.06   # take-profit at +6% above entry
MAX_DAILY_TRADES_STOCKS = 4   # stay under PDT threshold
MAX_DAILY_TRADES_CRYPTO = 40  # crypto has no PDT rule, be more generous
MAX_OPEN_POSITIONS = 5    # hold max 3 positions simultaneously

# ── Schedule ──────────────────────────────────────────────────
CHECK_INTERVAL   = 35     # sentiment refresh — every 60 minutes
EXECUTION_INTERVAL = 45   # order execution check — every 5 minutes
CRYPTO_TICKERS = {'BTCUSD', 'ETHUSD', 'SOLUSD', 'DOGEUSD'}

# Crypto-specific overrides
CRYPTO_RISK_PER_TRADE  = 0.01   # half the stock allocation — crypto is riskier
CRYPTO_STOP_LOSS_PCT   = 0.05   # wider stop — crypto swings hard
CRYPTO_TAKE_PROFIT_PCT = 0.10
CRYPTO_BUY_THRESHOLD  = 0.25   # 35% bulls enough (vs 65% for stocks)
CRYPTO_MIN_AVG_SCORE  = 0.06   # lower bar (vs 0.18 for stocks)
CRYPTO_SELL_THRESHOLD = 0.22

def is_crypto(ticker: str) -> bool:
    return ticker in CRYPTO_TICKERS

def to_order_symbol(ticker: str) -> str:
    """Convert internal ticker to Alpaca order format."""
    return ALPACA_ORDER_SYMBOL.get(ticker, ticker)

# ── GPU: auto-detect — FinBERT uses GPU if available ──────────
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'