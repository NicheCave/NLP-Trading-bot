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
REDDIT_CLIENT_ID  = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'SentimentBot/1.0')

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
REDDIT_POSTS     = 20     # number of Reddit posts to fetch per ticker

# ── Risk Management ───────────────────────────────────────────
RISK_PER_TRADE   = 0.02   # max 2% of portfolio per trade
STOP_LOSS_PCT    = 0.03   # stop-loss at -3% below entry
TAKE_PROFIT_PCT  = 0.06   # take-profit at +6% above entry
MAX_DAILY_TRADES = 5      # never trade more than 5x per day
MAX_OPEN_POSITIONS = 3    # hold max 3 positions simultaneously

# ── Schedule ──────────────────────────────────────────────────
CHECK_INTERVAL   = 60     # run bot every 60 minutes

# ── GPU: auto-detect — FinBERT uses GPU if available ──────────
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'