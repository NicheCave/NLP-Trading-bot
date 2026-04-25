# config.py — crypto-only configuration (TA + SMC edition, v2)
# ─────────────────────────────────────────────────────────────
# Changes from v1:
#   - CHECK_INTERVAL: 2 → 5 min (2 min was adding API noise with no benefit
#     since signals are driven by 1H bars, not 1M bars)
#   - DAILY_SYNC_INTERVAL: new constant — how often executor syncs the trade
#     counter from Alpaca (was called every 20s = 3×/min, wasteful)
#   - SMC thresholds added for order block / FVG proximity
#   - CRYPTO_BUY_THRESHOLD / SELL_THRESHOLD: kept at 0.35 — the RSI fix alone
#     will raise effective scores, so threshold does not need lowering
# ─────────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv

load_dotenv()

# ── API Credentials ───────────────────────────────────────────
ALPACA_API_KEY    = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL   = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

# ── Symbol mappings ───────────────────────────────────────────
ALPACA_ORDER_SYMBOL = {
    'BTCUSD':  'BTC/USD',
    'ETHUSD':  'ETH/USD',
    'SOLUSD':  'SOL/USD',
    'DOGEUSD': 'DOGE/USD',
}

CRYPTO_TICKERS = set(ALPACA_ORDER_SYMBOL.keys())

# ── Watchlist ─────────────────────────────────────────────────
_raw = os.getenv('WATCHLIST', 'BTCUSD,ETHUSD,SOLUSD')
WATCHLIST = [t.strip() for t in _raw.split(',') if t.strip() in CRYPTO_TICKERS]
if not WATCHLIST:
    WATCHLIST = ['BTCUSD', 'ETHUSD', 'SOLUSD']

# ── Signal Thresholds ─────────────────────────────────────────
CRYPTO_BUY_THRESHOLD  = 0.35   # ≥35% weighted bull votes → BUY candidate
CRYPTO_SELL_THRESHOLD = 0.35   # ≥35% weighted bear votes → SELL candidate
CRYPTO_MIN_AVG_SCORE  = 0.15   # net_score must exceed this to confirm direction

# ── Risk Management ───────────────────────────────────────────
CRYPTO_RISK_PER_TRADE  = 0.01   # 1% of portfolio per trade
CRYPTO_STOP_LOSS_PCT   = 0.05   # 5% stop-loss
CRYPTO_TAKE_PROFIT_PCT = 0.10   # 10% take-profit

USE_ATR_STOP        = True
ATR_STOP_MULTIPLIER = 1.5

MAX_DAILY_TRADES_CRYPTO = 40
MAX_OPEN_POSITIONS      = 4

# ── Schedule ──────────────────────────────────────────────────
# FIX: CHECK_INTERVAL raised from 2 → 5 minutes.
# The primary signal timeframe is 1H bars. Checking every 2 minutes
# adds 3× the API calls with no meaningful benefit — the 1H candle
# hasn't changed. 5 minutes is still responsive for intraday moves.
CHECK_INTERVAL     = 4    # TA + SMC analysis cycle (minutes)
EXECUTION_INTERVAL = 20   # order execution check (seconds)
EXIT_INTERVAL      = 30   # stop/TP check (seconds)

# How often to sync the daily trade counter from Alpaca (seconds).
# FIX: was called every EXECUTION_INTERVAL (every 20s = 3 API calls/min).
# Now synced once per minute — sufficient accuracy, much lower API load.
DAILY_SYNC_INTERVAL = 60  # seconds

# ── Indicator parameters ──────────────────────────────────────
EMA_FAST    = 9
EMA_SLOW    = 21
RSI_PERIOD  = 14
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9
BB_PERIOD   = 20
BB_STD      = 2.0
ATR_PERIOD  = 14

# ── SMC parameters ────────────────────────────────────────────
SMC_OB_IMPULSE_PCT   = 1.2   # min % move to qualify a candle as an order block origin
SMC_OB_PROXIMITY_PCT = 2.0   # max % from OB mid to cast a vote
SMC_FVG_PROXIMITY_PCT= 1.5   # max % from FVG to cast a vote
SMC_BOS_MAX_BARS     = 3     # BOS/CHoCH older than this many bars is stale
SMC_SWEEP_MAX_BARS   = 2     # liquidity sweep older than this is stale

# ── Helper functions ──────────────────────────────────────────
def is_crypto(ticker: str) -> bool:
    return ticker in CRYPTO_TICKERS

def to_order_symbol(ticker: str) -> str:
    return ALPACA_ORDER_SYMBOL.get(ticker, ticker)

# ── Device (kept for future ML expansion) ─────────────────────
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'