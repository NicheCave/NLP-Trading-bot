# risk.py — position sizing and portfolio risk control.
# ─────────────────────────────────────────────────────────────
# This module exists because the previous sizing rule did not do what its
# own config claimed. It computed:
#
#     dollars = portfolio × RISK_PER_TRADE × confidence
#     qty     = dollars / price
#
# which allocates 1% of equity as NOTIONAL. With a 5% stop the actual loss
# at the stop was 1% × 5% = 0.05% of equity — 20× smaller than intended,
# and, worse, unrelated to where the stop actually sat. Four "1% risk"
# positions risked 0.2% in total.
#
# The correct formulation makes risk independent of price and of stop width:
#
#     qty = (equity × RISK_PER_TRADE) / |entry − stop|
#
# A wider stop now buys a smaller position, so every trade loses the same
# fraction of equity when it's wrong. That is what makes R-multiples
# comparable across symbols and regimes — and therefore what makes the
# backtest's expectancy figure mean anything.
#
# Leverage plays no part in sizing. It only caps notional and sets where
# liquidation sits; it is a margin constraint, not a risk lever.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

import config
from brokers.base import BrokerPosition
from data import Instrument


# ══════════════════════════════════════════════════════════════
# STOPS AND TARGETS
# ══════════════════════════════════════════════════════════════

def atr_stop_target(entry: float,
                    side: str,
                    atr: float,
                    stop_mult: Optional[float] = None,
                    target_mult: Optional[float] = None) -> tuple[float, float]:
    """
    Volatility-scaled stop and target.

    Replaces the flat 5% / 7% that applied the same distance to BTC and SOL
    despite very different daily ranges — on a quiet BTC day a 5% stop sits
    miles beyond any meaningful invalidation, while on SOL it can be inside
    normal noise.

    The distance is clamped to [MIN_STOP_PCT, MAX_STOP_PCT] so a volatility
    collapse can't place the stop inside the spread, and a volatility spike
    can't demand an absurdly wide one.

    Returns (stop_price, take_price).
    """
    stop_mult = config.ATR_STOP_MULTIPLIER if stop_mult is None else stop_mult
    target_mult = config.ATR_TARGET_MULTIPLIER if target_mult is None else target_mult

    distance = atr * stop_mult if atr and atr > 0 else entry * config.MIN_STOP_PCT
    distance = float(np.clip(distance,
                             entry * config.MIN_STOP_PCT,
                             entry * config.MAX_STOP_PCT))

    # Keep the target at the intended R multiple even after the stop was
    # clamped — otherwise clamping silently changes the payoff ratio.
    reward = distance * (target_mult / stop_mult)

    if side == 'long':
        return entry - distance, entry + reward
    return entry + distance, entry - reward


# ══════════════════════════════════════════════════════════════
# POSITION SIZING
# ══════════════════════════════════════════════════════════════

@dataclass
class SizingResult:
    qty:             float = 0.0
    risk_amount:     float = 0.0
    notional:        float = 0.0
    margin_required: float = 0.0
    ok:              bool = False
    reason:          str = ''

    def __bool__(self) -> bool:
        return self.ok


def size_position(equity: float,
                  entry: float,
                  stop: float,
                  instrument: Instrument,
                  risk_pct: Optional[float] = None,
                  leverage: Optional[float] = None,
                  available_margin: Optional[float] = None,
                  confidence: float = 1.0,
                  scale_by_confidence: bool = False) -> SizingResult:
    """
    Size a position so that hitting the stop costs exactly risk_pct of equity.

    Args:
        scale_by_confidence: multiply risk by the signal's confidence. Off by
            default — it is only justified once confidence is CALIBRATED
            against realised win rate. Scaling by an uncalibrated score just
            adds variance to position size for no expected-value gain.
    """
    risk_pct = config.RISK_PER_TRADE if risk_pct is None else risk_pct
    leverage = config.LEVERAGE if leverage is None else leverage

    stop_distance = abs(entry - stop)
    if stop_distance <= 0:
        return SizingResult(reason='stop equals entry')
    if entry <= 0 or equity <= 0:
        return SizingResult(reason='invalid equity or price')

    risk_amount = equity * risk_pct
    if scale_by_confidence:
        risk_amount *= max(0.0, min(1.0, confidence))

    qty = risk_amount / stop_distance
    notional = qty * entry

    # Leverage cap: notional can never exceed equity × leverage.
    max_notional = equity * leverage
    if notional > max_notional:
        qty = max_notional / entry
        notional = qty * entry
        risk_amount = qty * stop_distance
        logger.debug(f"size capped by {leverage}× leverage — risk reduced to "
                     f"{risk_amount / equity:.2%} of equity")

    # Bybit rejects a market order above the symbol's maximum size. Past it the
    # position is capped and simply risks less than the budget. Only large
    # accounts reach it: SUIUSDT caps out around $170k of equity.
    max_qty = float(instrument.max_order_qty)
    if qty > max_qty:
        qty = max_qty
        logger.info(f"{instrument.symbol}: size capped at the exchange maximum of "
                    f"{max_qty:g} per order — risk reduced to "
                    f"{qty * stop_distance / equity:.2%} of equity")

    qty = instrument.round_qty(qty)
    if qty <= 0:
        return SizingResult(reason=f'qty below min lot {instrument.min_qty}')

    notional = qty * entry
    if not instrument.is_tradeable(qty, entry):
        return SizingResult(reason=f'notional {notional:.2f} below exchange minimum '
                                   f'{instrument.min_notional}')

    margin_required = notional / leverage
    if available_margin is not None and margin_required > available_margin:
        return SizingResult(reason=f'margin {margin_required:.2f} exceeds available '
                                   f'{available_margin:.2f}')

    return SizingResult(
        qty=qty,
        risk_amount=qty * stop_distance,
        notional=notional,
        margin_required=margin_required,
        ok=True,
        reason='ok',
    )


# ══════════════════════════════════════════════════════════════
# PORTFOLIO-LEVEL RISK
# ══════════════════════════════════════════════════════════════

def position_risk(position: BrokerPosition) -> float:
    """
    Quote-currency loss if this position's stop is hit right now.

    A position with no stop is treated as risking its whole notional: an
    unprotected leveraged position genuinely can lose that much, and
    pretending otherwise would understate portfolio heat.
    """
    if position.stop_loss is None:
        return position.qty * position.mark_price
    return abs(position.entry_price - position.stop_loss) * position.qty


def portfolio_heat(positions: dict[str, BrokerPosition], equity: float) -> float:
    """Total open risk as a fraction of equity."""
    if equity <= 0:
        return 0.0
    return sum(position_risk(p) for p in positions.values()) / equity


# ── correlation ───────────────────────────────────────────────

_corr_cache: dict = {}


def correlation_matrix(symbols: list[str],
                       lookback: Optional[int] = None,
                       ttl_seconds: int = 3600) -> pd.DataFrame:
    """
    Rolling correlation of hourly returns, cached for an hour.

    Needed because BTC/ETH/SOL routinely run above 0.8: four "independent"
    1% positions in majors is really one 4% bet that all resolves together.
    """
    lookback = lookback or config.CORRELATION_LOOKBACK
    key = (tuple(sorted(symbols)), lookback)
    entry = _corr_cache.get(key)
    if entry and (time.time() - entry[0]) < ttl_seconds:
        return entry[1]

    from data import get_klines

    closes = {}
    for symbol in symbols:
        try:
            df = get_klines(symbol, '60', days=max(10.0, lookback / 24 + 2))
            if not df.empty:
                closes[symbol] = df['close'].tail(lookback)
        except Exception as e:
            logger.debug(f"correlation fetch failed for {symbol}: {e}")

    if len(closes) < 2:
        return pd.DataFrame()

    matrix = pd.DataFrame(closes).pct_change().dropna().corr()
    _corr_cache[key] = (time.time(), matrix)
    return matrix


def correlated_exposure_ok(symbol: str,
                           side: str,
                           positions: dict[str, BrokerPosition],
                           threshold: Optional[float] = None,
                           max_correlated: Optional[int] = None) -> tuple[bool, str]:
    """
    Refuse a trade that merely duplicates exposure we already hold.

    Counts existing positions that are (a) in the same direction and
    (b) correlated above the threshold. Anti-correlated positions in
    opposite directions are also duplicates, and are counted too.
    """
    threshold = threshold or config.CORRELATION_THRESHOLD
    max_correlated = max_correlated or config.MAX_CORRELATED_POSITIONS
    if not positions:
        return True, 'no open positions'

    matrix = correlation_matrix(list({symbol} | set(positions)))
    if matrix.empty or symbol not in matrix.columns:
        return True, 'correlation unavailable'

    duplicates = 0
    for other, position in positions.items():
        if other == symbol or other not in matrix.columns:
            continue
        rho = float(matrix.at[symbol, other])
        same_direction = position.side == side
        if (rho >= threshold and same_direction) or (rho <= -threshold and not same_direction):
            duplicates += 1

    if duplicates >= max_correlated:
        return False, (f'{duplicates} correlated positions already open '
                       f'(ρ ≥ {threshold}) — this would concentrate risk')
    return True, f'{duplicates} correlated positions'


# ══════════════════════════════════════════════════════════════
# ENTRY GATE
# ══════════════════════════════════════════════════════════════

@dataclass
class RiskDecision:
    allowed: bool
    reason:  str
    sizing:  Optional[SizingResult] = None

    def __bool__(self) -> bool:
        return self.allowed


def evaluate_entry(symbol: str,
                   side: str,
                   entry: float,
                   stop: float,
                   equity: float,
                   positions: dict[str, BrokerPosition],
                   instrument: Instrument,
                   available_margin: Optional[float] = None,
                   confidence: float = 1.0) -> RiskDecision:
    """
    Single gate every new position passes through.

    Checks run cheapest-first so an obviously-blocked trade never triggers a
    correlation fetch.
    """
    if symbol in positions:
        return RiskDecision(False, 'already in position')

    if len(positions) >= config.MAX_OPEN_POSITIONS:
        return RiskDecision(False, f'max positions ({config.MAX_OPEN_POSITIONS}) reached')

    heat = portfolio_heat(positions, equity)
    incoming = config.RISK_PER_TRADE
    if heat + incoming > config.MAX_PORTFOLIO_HEAT:
        return RiskDecision(
            False,
            f'portfolio heat {heat:.1%} + {incoming:.1%} exceeds '
            f'{config.MAX_PORTFOLIO_HEAT:.1%} cap')

    ok, why = correlated_exposure_ok(symbol, side, positions)
    if not ok:
        return RiskDecision(False, why)

    sizing = size_position(
        equity=equity, entry=entry, stop=stop, instrument=instrument,
        available_margin=available_margin, confidence=confidence,
    )
    if not sizing.ok:
        return RiskDecision(False, sizing.reason, sizing)

    return RiskDecision(True, 'ok', sizing)


# ══════════════════════════════════════════════════════════════
# EXITS
# ══════════════════════════════════════════════════════════════

def trailing_stop_price(position: BrokerPosition,
                        extreme_price: float,
                        atr: float,
                        activate_r: Optional[float] = None,
                        multiplier: Optional[float] = None) -> Optional[float]:
    """
    Trailing stop level, or None while the trail is not yet armed.

    Armed only after the position has run activate_r in our favour.
    Trailing from bar one converts ordinary noise into stop-outs and is one
    of the fastest ways to turn a positive-edge system negative.

    `extreme_price` is the best price seen since entry (high for a long,
    low for a short).
    """
    activate_r = config.TRAILING_ACTIVATE_R if activate_r is None else activate_r
    multiplier = config.TRAILING_ATR_MULTIPLIER if multiplier is None else multiplier

    if not atr or atr <= 0 or position.stop_loss is None:
        return None

    risk_per_unit = abs(position.entry_price - position.stop_loss)
    if risk_per_unit <= 0:
        return None

    if position.side == 'long':
        excursion_r = (extreme_price - position.entry_price) / risk_per_unit
        if excursion_r < activate_r:
            return None
        level = extreme_price - multiplier * atr
        return level if level > position.stop_loss else None

    excursion_r = (position.entry_price - extreme_price) / risk_per_unit
    if excursion_r < activate_r:
        return None
    level = extreme_price + multiplier * atr
    return level if level < position.stop_loss else None


def funding_argues_for_exit(side: str, funding_rate: float) -> bool:
    """
    True when funding has turned meaningfully against the position.

    A long paying elevated funding is a negative-carry bet: the price move
    has to outrun the cost just to break even, and 48h of it is six
    settlements. This is a futures-specific exit the spot bot could not have.
    """
    threshold = config.FUNDING_EXIT_THRESHOLD
    if side == 'long':
        return funding_rate > threshold
    return funding_rate < -threshold
