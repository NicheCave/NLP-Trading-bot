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
CRYPTO_MIN_AVG_SCORE  = 0.22   # raised from 0.15 — kills marginal-edge signals (v5)

# ── Risk Management ───────────────────────────────────────────
CRYPTO_RISK_PER_TRADE  = 0.01   # 1% of portfolio per trade
CRYPTO_STOP_LOSS_PCT   = 0.05   # 5% stop-loss
CRYPTO_TAKE_PROFIT_PCT = 0.07   # 7% take-profit — reachable within time stop window (was 0.15) (v6)

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
RVOL_MIN_THRESHOLD = 2.0  # (v6: raised from 1.5 — 2× volume confirms stronger institutional presence)

# ── v3: Bellafiore — Pyramiding ───────────────────────────────
# Minimum unrealized P&L as a multiple of ATR before scaling in.
# 1.0 = position must be 1 ATR profitable before we add.
# Higher = more conservative pyramiding; lower = more aggressive.
PYRAMID_MIN_ATR_FACTOR = 2.0  # (v6: raised from 1.0 — position must be 2 ATR profitable before scaling in)

# ── v3: Jack Schwager — Trap Detector ────────────────────────
# Maximum age (in bars) of a trap pattern for it to still be actionable.
# A bull trap detected 10 bars ago is already played out.
# 5 bars = ~5 hours on 1H chart — still relevant for a momentum flip.
TRAP_MAX_BARS = 5

# ── v3: Elliott Wave — Divergence Lookback ───────────────────
# Number of 1H bars to scan for swing highs/lows when detecting
# RSI divergence (Wave 3 / Wave 5 identification).
ELLIOTT_LOOKBACK = 50

# ── v4: Institutional "Kill Zones" ────────────────────────────
# Smart money (institutional) volume concentrates at session overlaps.
# Signals outside these windows are more likely retail noise.
# All times are UTC.  Set KILL_ZONES_ENABLED=False to disable.
KILL_ZONES_ENABLED = True

KILL_ZONES = [
    # (label,        start_hour_utc, end_hour_utc)
    # IMPORTANT: Overlap MUST come before NY because (13–15) ⊂ (13–16).
    # The matcher returns the first hit, so Overlap has to be tested first
    # or it'd be shadowed by NY and never reported. (v5 fix)
    ('Overlap',     13,  15),   # London–NY overlap — THE highest-probability window
    ('London',       8,  11),   # London open — strong EUR/GBP/BTC correlation
    ('NY',          13,  16),   # New York open — highest USD-denominated volume
    ('Asia',         0,   3),   # Tokyo/Singapore open — BTC/ETH deepest Asian vol
]

# Confidence penalty (multiplier) applied to signals generated OUTSIDE kill zones.
# 0.80 = 20% confidence haircut for "off-hours" entries.
KILL_ZONE_OFF_HOURS_PENALTY = 0.80

# ── v4: MTF Daily Alignment ───────────────────────────────────
# If 1D trend opposes the 1H signal direction, apply this confidence penalty.
# 0.70 = 30% haircut for counter-trend trades (still allows them but shrinks size).
MTF_COUNTER_TREND_PENALTY  = 0.70

# Minimum daily_trend requirement for a BUY to get full size.
# 'uptrend'  → only buy when 1D is also bullish
# 'sideways' → allow buys in sideways + uptrend (less strict)
MTF_BUY_MIN_DAILY_TREND = 'sideways'   # set to 'uptrend' for strictest mode

# ── v4: Time-Based Stops ──────────────────────────────────────
# If a position hasn't hit TP or SL after this many hours, exit at market.
# Steenbarger/Bellafiore: time is risk. Zombie trades occupy capital.
TIME_STOP_HOURS = 48  # (v6: was 12 — 12h was too short for a 7% TP target; 48h gives price room to breathe)

# ── v4: ATR Trailing Stop ─────────────────────────────────────
# Multiplier for the ATR-based trailing stop.
# 2.0 × ATR  keeps stop wide enough to breathe but still locks in profits.
TRAILING_ATR_MULTIPLIER = 2.0

# ─────────────────────────────────────────────────────────────
# v5: NEW EDGE CONSTANTS + BUG-FIX KNOBS
# ─────────────────────────────────────────────────────────────

# ── v5: Confidence floor (signal_generator) ───────────────────
# After all stacking penalties (counter-trend × kill-zone × wave5 × bear),
# confidence can compound down to single digits. Below this threshold the
# signal is downgraded to HOLD instead of placing a tiny token order.
# 0.25 = 25% — anything weaker is just clutter and burns daily-trade quota.
CONFIDENCE_FLOOR = 0.25

# ── v5: Loss-streak cooldown ──────────────────────────────────
# After this many consecutive losses on the same ticker, freeze new BUYs
# on that ticker for COOLDOWN_HOURS. Steenbarger "tilt insurance":
# losing streaks correlate with regime changes the model hasn't adapted to.
LOSS_STREAK_THRESHOLD     = 2
LOSS_STREAK_COOLDOWN_HRS  = 6

# ── v5: Volume-confirmed BOS / CHoCH ──────────────────────────
# A structural break on average volume is much weaker than one on RVOL ≥ 1.5.
# When RVOL meets this threshold during a BOS/CHoCH, the smc_bos_choch vote
# weight is multiplied by BOOST. When RVOL is anaemic, the vote is REDUCED.
BOS_VOLUME_RVOL_MIN  = 1.5
BOS_VOLUME_BOOST     = 1.30   # weight × 1.30 when volume confirms
BOS_VOLUME_PENALTY   = 0.60   # weight × 0.60 when volume is weak

# ── v5: Pyramiding storm guard ────────────────────────────────
# Minimum interval (in seconds) between two scale-ins on the same ticker.
# Without this, scale_in_winner would fire every EXECUTION_INTERVAL while
# the cached signal still says add_to_winner=True — turning a 50% pyramid
# into a 600% one inside a single 4-minute analysis cycle.
# Default = one full analysis cycle worth of seconds.
PYRAMID_MIN_INTERVAL_SEC = CHECK_INTERVAL * 60   # 240 s by default

# ── Helper functions ──────────────────────────────────────────
def is_crypto(ticker: str) -> bool:
    return ticker in CRYPTO_TICKERS

def to_order_symbol(ticker: str) -> str:
    return ALPACA_ORDER_SYMBOL.get(ticker, ticker)

# ── Device ────────────────────────────────────────────────────
# Torch/CUDA removed: all computation is pure NumPy/Pandas.
# Keeping this constant so any code that reads config.DEVICE still works.
DEVICE = 'cpu'