# config.py — crypto-only configuration (TA + SMC + Advanced TA, v3)
# ─────────────────────────────────────────────────────────────
# Changes from v2:
#   - Added RVOL_MIN_THRESHOLD      (Bellafiore "Stocks in Play" filter)
#   - Added TRAP_MAX_BARS           (Schwager trap detector staleness window)
#   - Added EDGE_LOG_INTERVAL       (Steenbarger edge report frequency)
#   - Added PYRAMID_MIN_ATR_FACTOR  (Bellafiore: min ATR profit before scaling)
#   - Added AVWAP_LOOKBACK          (Shannon: bar window for AVWAP pivot detection)
#   - Added ALLIGATOR_SPREAD_PCT    (Williams: threshold for sleeping detection)
#   - Added ELLIOTT_LOOKBACK        (wave divergence lookback window)
#   - All v2 settings preserved unchanged
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
CRYPTO_BUY_THRESHOLD  = 0.35
CRYPTO_SELL_THRESHOLD = 0.35
CRYPTO_MIN_AVG_SCORE  = 0.15

# ── Risk Management ───────────────────────────────────────────
CRYPTO_RISK_PER_TRADE  = 0.01   # 1% of portfolio per trade
CRYPTO_STOP_LOSS_PCT   = 0.05   # 5% stop-loss
CRYPTO_TAKE_PROFIT_PCT = 0.10   # 10% take-profit

USE_ATR_STOP        = True
ATR_STOP_MULTIPLIER = 1.5

MAX_DAILY_TRADES_CRYPTO = 40
MAX_OPEN_POSITIONS      = 4

# ── Schedule ──────────────────────────────────────────────────
CHECK_INTERVAL      = 4    # TA + SMC + AdvTA analysis cycle (minutes)
EXECUTION_INTERVAL  = 20   # order execution check (seconds)
EXIT_INTERVAL       = 30   # stop/TP check (seconds)
DAILY_SYNC_INTERVAL = 60   # daily trade counter sync (seconds)

# Steenbarger edge report frequency (minutes)
# Every 4 hours is a reasonable default — enough trades to be meaningful
# but not so frequent that it floods the log.
EDGE_LOG_INTERVAL = 240    # minutes (4 hours)

# ── Indicator parameters (existing) ──────────────────────────
EMA_FAST    = 9
EMA_SLOW    = 21
RSI_PERIOD  = 14
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9
BB_PERIOD   = 20
BB_STD      = 2.0
ATR_PERIOD  = 14

# ── SMC parameters (existing) ────────────────────────────────
SMC_OB_IMPULSE_PCT    = 1.2
SMC_OB_PROXIMITY_PCT  = 2.0
SMC_FVG_PROXIMITY_PCT = 1.5
SMC_BOS_MAX_BARS      = 3
SMC_SWEEP_MAX_BARS    = 2

# ── v3: Brian Shannon — Anchored VWAP ────────────────────────
# Number of bars to scan for the anchor pivots (swing high/low).
# 60 hourly bars = ~2.5 days — captures recent institutional events.
AVWAP_LOOKBACK = 60

# ── v3: Bill Williams — Alligator ────────────────────────────
# Below this spread % between jaw/teeth/lips, the market is "sleeping".
# 0.3% is Williams' original guidance adapted for crypto volatility.
# Raise to 0.5% for lower-volatility assets.
ALLIGATOR_SPREAD_PCT = 0.3

# ── v3: Bellafiore — Relative Volume ("Stocks in Play") ──────
# Minimum RVOL multiple to consider an asset "in play" for entry.
# 1.5x = at least 50% more volume than the rolling 20-bar average.
# Only assets above this threshold have confirmed institutional presence.
RVOL_MIN_THRESHOLD = 1.5

# ── v3: Bellafiore — Pyramiding ───────────────────────────────
# Minimum unrealized P&L as a multiple of ATR before scaling in.
# 1.0 = position must be 1 ATR profitable before we add.
# Higher = more conservative pyramiding; lower = more aggressive.
PYRAMID_MIN_ATR_FACTOR = 1.0

# ── v3: Jack Schwager — Trap Detector ────────────────────────
# Maximum age (in bars) of a trap pattern for it to still be actionable.
# A bull trap detected 10 bars ago is already played out.
# 5 bars = ~5 hours on 1H chart — still relevant for a momentum flip.
TRAP_MAX_BARS = 5

# ── v3: Elliott Wave — Divergence Lookback ───────────────────
# Number of 1H bars to scan for swing highs/lows when detecting
# RSI divergence (Wave 3 / Wave 5 identification).
ELLIOTT_LOOKBACK = 50

# ── Helper functions ──────────────────────────────────────────
def is_crypto(ticker: str) -> bool:
    return ticker in CRYPTO_TICKERS

def to_order_symbol(ticker: str) -> str:
    return ALPACA_ORDER_SYMBOL.get(ticker, ticker)

# ── Device ────────────────────────────────────────────────────
# Torch/CUDA removed: all computation is pure NumPy/Pandas.
# Keeping this constant so any code that reads config.DEVICE still works.
DEVICE = 'cpu'