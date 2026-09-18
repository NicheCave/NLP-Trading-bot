# config.py — Bybit USDT-perpetual futures configuration (v7)
# ─────────────────────────────────────────────────────────────
# Migrated from Alpaca crypto SPOT (long-only) to Bybit linear perpetuals
# traded in BOTH directions on the Demo Trading environment.
#
# What changed from v6, and why:
#   - Exchange is Bybit v5; symbols are perps (BTCUSDT, not BTC/USD).
#   - Risk is now expressed as "% of equity lost when the stop is hit",
#     which is what RISK_PER_TRADE always claimed to mean but never did.
#   - Stops and targets are ATR multiples, not flat percentages, so they
#     scale with each symbol's volatility instead of treating BTC and SOL
#     as if they move the same amount.
#   - Funding is a first-class cost and a signal input.
#
# Every legacy indicator threshold is preserved verbatim so the backtest
# baseline measures the CURRENT strategy, not a re-tuned one. Changes get
# made after the baseline exists, never before.
# ─────────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════
# EXCHANGE / CREDENTIALS
# ══════════════════════════════════════════════════════════════

BYBIT_API_KEY    = os.getenv('BYBIT_API_KEY')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET')

# Demo Trading runs inside the real platform: genuine mainnet prices,
# spreads and liquidity, but fake money. Bybit's own guidance is to use
# Demo rather than Testnet when evaluating a strategy, because Testnet is
# separate infrastructure whose prices don't resemble the real market.
#
# Keys must be minted from the Demo Trading section of a MAINNET account.
BYBIT_DEMO    = os.getenv('BYBIT_DEMO', 'true').lower() == 'true'
BYBIT_TESTNET = os.getenv('BYBIT_TESTNET', 'false').lower() == 'true'

CATEGORY = 'linear'          # USDT-margined perpetuals

# ══════════════════════════════════════════════════════════════
# UNIVERSE
# ══════════════════════════════════════════════════════════════

# Screened from Bybit's full linear-perp list on three criteria, because
# widening the universe is the only way to raise trade frequency WITHOUT
# lowering the conviction bar — and lowering the bar was measured to be
# actively destructive (gross expectancy goes negative below |net| ~0.35).
#
#   ≥730 days of 1H history   the walk-forward needs two years; new listings
#                             cannot be validated and would be pure guesswork
#   spread ≤3.5 bps           the spread is a cost paid on every round trip;
#                             the backtest assumes 2 bps of slippage per side
#   ≥$20M daily turnover      thin books make that slippage assumption a lie
#
# Deliberately EXCLUDED despite high turnover: tokenised equities and
# commodities (SOXLUSDT, SPCXUSDT, SNDKUSDT, SKHYNIXUSDT, XAUUSDT). They
# have market-hours gaps and are driven by things this stack never models,
# and the strategy was validated exclusively on 24/7 crypto.
_raw_symbols = os.getenv(
    'SYMBOLS',
    'BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,BNBUSDT,LINKUSDT,NEARUSDT,SUIUSDT,WLDUSDT'
)
SYMBOLS = [s.strip().upper() for s in _raw_symbols.split(',') if s.strip()]

# BTC is the market's tide — used as the intermarket anchor for alts.
ANCHOR_SYMBOL = 'BTCUSDT'

QUOTE_CURRENCY = 'USDT'

# ══════════════════════════════════════════════════════════════
# FUTURES MECHANICS
# ══════════════════════════════════════════════════════════════

# Leverage caps notional; it is NOT the risk lever. Risk is set by the
# stop distance (see RISK_PER_TRADE). Leverage only decides how much
# margin the exchange locks up and where liquidation sits.
LEVERAGE     = float(os.getenv('LEVERAGE', '3'))
MARGIN_MODE  = 'ISOLATED'    # per-position margin — one bad trade can't
                             # cascade into the rest of the book
POSITION_MODE = 'one_way'    # positionIdx=0; no simultaneous long+short

# Stops trigger off MARK price, not last price. Last price can be wicked
# by a thin book; mark price is index-anchored and much harder to push.
TRIGGER_PRICE_TYPE = 'MarkPrice'

# Signature validity window, in milliseconds. Bybit rejects a request whose
# timestamp is further than this from ITS clock (retCode 10002).
#
# The default of 5000 assumes a well-synchronised machine. Windows clocks
# routinely drift several seconds, which produces intermittent rejections —
# including, potentially, on an order. This is a tolerance, NOT a fix: if
# you see 10002, resync the system clock (`w32tm /resync /force`).
RECV_WINDOW = 20_000

# api-demo.bybit.com drops idle keep-alive connections far more often than
# mainnet. These retries are handled inside pybit and usually succeed.
HTTP_TIMEOUT     = 30
HTTP_MAX_RETRIES = 3
HTTP_RETRY_DELAY = 2

# How often to poll the exchange for positions while holding nothing. The
# management loop runs every EXIT_INTERVAL seconds, but with nothing open
# there is nothing to manage — polling that often just multiplies the
# chance of hitting a connection reset.
FLAT_POLL_INTERVAL = 300   # seconds

# ══════════════════════════════════════════════════════════════
# RISK
# ══════════════════════════════════════════════════════════════

# Fraction of equity lost if the stop is hit. This is real risk now:
#   qty = (equity × RISK_PER_TRADE) / |entry − stop|
RISK_PER_TRADE = 0.01        # 1%

MAX_OPEN_POSITIONS = 4

# Total simultaneous risk across all open positions. With 1% per trade and
# 4 positions the theoretical max is 4%, but BTC/ETH/SOL are ~0.8
# correlated — in a hard sell-off all four stops fill together, so this is
# effectively one 4% bet, not four independent 1% ones.
MAX_PORTFOLIO_HEAT = 0.04

# Correlated exposure cap: how many positions may face the same direction
# at once given how tightly majors move together.
MAX_CORRELATED_POSITIONS = 2
CORRELATION_THRESHOLD    = 0.75
CORRELATION_LOOKBACK     = 168   # 1 week of 1H bars

# ── Stops and targets (ATR-based) ─────────────────────────────
# Replaces the flat 5% / 7% of v6, which applied the same distance to BTC
# and SOL despite very different volatility.
ATR_PERIOD           = 14
ATR_STOP_MULTIPLIER  = 1.8   # stop distance = 1.8 × ATR
ATR_TARGET_MULTIPLIER = 3.6  # target distance = 3.6 × ATR → 2.0 R
MIN_STOP_PCT         = 0.004  # floor: never place a stop inside noise
MAX_STOP_PCT         = 0.08   # ceiling: refuse absurdly wide stops

# ATR trailing stop, applied once a trade is meaningfully in profit.
TRAILING_ATR_MULTIPLIER = 2.0
TRAILING_ACTIVATE_R     = 1.0   # only start trailing after +1R

# ── Time stop ─────────────────────────────────────────────────
# Perps charge funding every 8h, so a stale position bleeds on a schedule.
TIME_STOP_HOURS = 48

# Bail out early if funding turns strongly against the position: holding a
# long while paying elevated funding is a negative-carry bet that needs the
# price move to outrun the cost.
FUNDING_EXIT_THRESHOLD = 0.0005   # 0.05% per 8h ≈ 55% annualised

# ── Costs (must match live Bybit fees for the backtest to mean anything) ──
TAKER_FEE    = 0.00055    # 0.055%
MAKER_FEE    = 0.0002     # 0.02%
SLIPPAGE_BPS = 2.0

# ══════════════════════════════════════════════════════════════
# SCHEDULE
# ══════════════════════════════════════════════════════════════

CHECK_INTERVAL      = 4    # analysis cycle (minutes)
EXECUTION_INTERVAL  = 20   # order execution check (seconds)
EXIT_INTERVAL       = 30   # local stop/TP safety net (seconds)
DAILY_SYNC_INTERVAL = 60   # state reconciliation (seconds)
EDGE_LOG_INTERVAL   = 240  # edge report (minutes)

MAX_DAILY_TRADES = 40

# ══════════════════════════════════════════════════════════════
# SIGNAL THRESHOLDS  (preserved from v6 for baseline fidelity)
# ══════════════════════════════════════════════════════════════

BUY_THRESHOLD  = 0.35
SELL_THRESHOLD = 0.35
MIN_AVG_SCORE  = 0.22
CONFIDENCE_FLOOR = 0.25

# Legacy aliases — older modules still reference the CRYPTO_* names.
CRYPTO_BUY_THRESHOLD  = BUY_THRESHOLD
CRYPTO_SELL_THRESHOLD = SELL_THRESHOLD
CRYPTO_MIN_AVG_SCORE  = MIN_AVG_SCORE
CRYPTO_RISK_PER_TRADE = RISK_PER_TRADE

# ══════════════════════════════════════════════════════════════
# INDICATOR PARAMETERS  (preserved verbatim from v6)
# ══════════════════════════════════════════════════════════════

EMA_FAST    = 9
EMA_SLOW    = 21
RSI_PERIOD  = 14
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9
BB_PERIOD   = 20
BB_STD      = 2.0

# ── SMC ───────────────────────────────────────────────────────
SMC_OB_IMPULSE_PCT    = 1.2
SMC_OB_PROXIMITY_PCT  = 2.0
SMC_FVG_PROXIMITY_PCT = 1.5
SMC_BOS_MAX_BARS      = 3
SMC_SWEEP_MAX_BARS    = 2

# ── Advanced TA ───────────────────────────────────────────────
AVWAP_LOOKBACK       = 60
ALLIGATOR_SPREAD_PCT = 0.3
RVOL_MIN_THRESHOLD   = 2.0
PYRAMID_MIN_ATR_FACTOR = 2.0
TRAP_MAX_BARS        = 5
ELLIOTT_LOOKBACK     = 50

BOS_VOLUME_RVOL_MIN = 1.5
BOS_VOLUME_BOOST    = 1.30
BOS_VOLUME_PENALTY  = 0.60

PYRAMID_MIN_INTERVAL_SEC = CHECK_INTERVAL * 60

# ── Kill zones (UTC) ──────────────────────────────────────────
# Retained for the baseline, but flagged for measurement: this is FX and
# equities session logic applied to a market that never closes. Phase 5
# tests it against measured hour-of-week performance rather than assuming.
KILL_ZONES_ENABLED = True
KILL_ZONES = [
    ('Overlap', 13, 15),   # must precede NY — (13,15) ⊂ (13,16)
    ('London',   8, 11),
    ('NY',      13, 16),
    ('Asia',     0,  3),
]
KILL_ZONE_OFF_HOURS_PENALTY = 0.80

# ── MTF ───────────────────────────────────────────────────────
MTF_COUNTER_TREND_PENALTY = 0.70
MTF_BUY_MIN_DAILY_TREND   = 'sideways'

# ── Loss-streak cooldown ──────────────────────────────────────
LOSS_STREAK_THRESHOLD    = 2
LOSS_STREAK_COOLDOWN_HRS = 6

# ══════════════════════════════════════════════════════════════
# BACKTEST
# ══════════════════════════════════════════════════════════════

BACKTEST_DAYS           = 730
BACKTEST_INITIAL_EQUITY = 10_000.0
BACKTEST_WARMUP_BARS    = 300

# Bars of 1H history handed to the indicator stack per decision.
# The live path and the backtest MUST use the same number — otherwise the
# backtest scores a different strategy than the one that trades. 400 bars
# comfortably covers the longest lookback in the stack (AVWAP 60, Elliott
# 50, fractals 60) with room for the recursive EMAs/SMMAs to converge.
ANALYSIS_WINDOW_BARS = 400

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def is_anchor(symbol: str) -> bool:
    return symbol == ANCHOR_SYMBOL


def credentials_present() -> bool:
    """True when live/demo trading is possible. Backtests need no keys."""
    return bool(BYBIT_API_KEY and BYBIT_API_SECRET)


DEVICE = 'cpu'
