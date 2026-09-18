# strategy_v7.py — regime-routed, block-aggregated futures strategy.
# ─────────────────────────────────────────────────────────────
# Reuses the entire existing indicator stack. What changes is how its
# output is COMBINED and GATED:
#
#   1. Factor blocks       correlated indicators can no longer vote five
#                          times for the same idea (features/blocks.py)
#   2. Regime routing      trend logic is trusted in trends, mean-reversion
#                          in ranges, and nothing in volatile chop
#   3. Symmetric direction shorts are first-class, not "don't be long"
#   4. Derivatives context funding and open interest — futures-only
#   5. ATR stops           risk scales with each symbol's volatility
#   6. One confidence calc a single bounded logistic instead of a chain of
#                          multiplicative boosts and penalties
#
# Everything tunable lives in V7Params so Phase 5 can fit it out-of-sample
# rather than by hand.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd

import config
import risk
from backtest.engine import Intent
from features.blocks import aggregate_blocks, agreement
from features.derivatives import DerivativesState, analyse_derivatives
from features.regime import Regime, RegimeState, classify_regime
from indicators import atr as calc_atr
from market_analyzer import analyze_frames


@dataclass
class V7Params:
    """Every knob the strategy exposes. Fitted in Phase 5, not by hand."""
    # ── entry gates ───────────────────────────────────────────
    min_net:              float = 0.18   # |net block score| to act at all
    min_agreement:        float = 0.55   # share of block weight that must agree
    counter_trend_min_net: float = 0.32  # steeper bar for fading a trend
    min_active_blocks:    int   = 3      # breadth requirement

    # ── confidence (logistic) ─────────────────────────────────
    conf_edge_weight:  float = 6.0
    conf_agree_weight: float = 3.0
    conf_bias:         float = 1.9
    confidence_floor:  float = 0.25

    # ── stops ─────────────────────────────────────────────────
    atr_stop_mult:   float = 1.8
    atr_target_mult: float = 3.6

    # ── context gates ─────────────────────────────────────────
    use_btc_anchor:  bool = True
    use_funding_gate: bool = True
    funding_extreme_z: float = 1.8
    trade_squeeze:   bool = False   # sit out compression, act on release

    # Restrict entries to config.KILL_ZONES (session windows, UTC).
    #
    # v7 originally dropped these as unvalidated FX logic applied to a 24/7
    # market. Measuring them on the baseline's 815 trades contradicted that:
    # in-zone entries returned +0.039R (52.4% win) against −0.036R (48.7%)
    # outside, a +0.075R gap. That is suggestive but roughly one standard
    # error, and the per-hour detail is plainly noise (09:00 London −0.251R,
    # 10:00 London +0.237R, ~30 trades each).
    #
    # So it is a parameter, not a decision: let walk-forward settle it.
    # Default off, matching what has actually been validated.
    use_kill_zones:  bool = False


DEFAULT_PARAMS = V7Params()


@dataclass
class V7Decision:
    """Full reasoning trace — used by the edge log and for debugging."""
    direction:  str = 'flat'
    net_score:  float = 0.0
    agreement:  float = 0.0
    confidence: float = 0.0
    regime:     Optional[RegimeState] = None
    derivatives: Optional[DerivativesState] = None
    blocks:     dict = None
    reason:     str = ''
    tags:       list[str] = None
    rejected:   str = ''


def _derivatives_block(state: DerivativesState, params: V7Params) -> Optional[float]:
    """
    Collapse funding and open interest into one directional score in -1..+1.

    Kept deliberately weak and CONTEXTUAL. Extreme funding can persist for
    weeks inside a strong trend, so shorting purely because funding is high
    is a well-documented way to lose money slowly. It nudges; it never
    decides.
    """
    if not state.available:
        return None

    score = 0.0
    if params.use_funding_gate and state.funding_extreme:
        score += 0.6 * state.funding_bias
    if state.oi_bias:
        score += 0.4 * state.oi_bias
    return max(-1.0, min(1.0, score)) if score else None


def _confidence(net_abs: float, block_agreement: float, params: V7Params) -> float:
    """
    One bounded logistic, replacing the old multiplicative chain.

    The previous version started at net×2.5 then applied up to seven boosts
    (×1.20, ×1.15, ×1.10, ×1.08, ×1.10, ×1.07, ×1.05) and four penalties
    (×0.70, ×0.80, ×0.60, ×0.65), clipping at 1.0 throughout. Because it
    clipped, strong signals all collapsed to the same 1.0 and the ordering
    of the multiplications changed the answer. This is monotonic in both
    inputs, bounded by construction, and has parameters that can actually
    be fitted against realised win rate.
    """
    z = (params.conf_edge_weight * net_abs
         + params.conf_agree_weight * (block_agreement - 0.5)
         - params.conf_bias)
    return 1.0 / (1.0 + math.exp(-z))


def _in_kill_zone(ts) -> bool:
    """True when `ts` falls inside one of config.KILL_ZONES (UTC hours)."""
    hour = ts.hour
    return any(start <= hour < end for _, start, end in config.KILL_ZONES)


def _regime_allows(direction: str, regime: RegimeState, net_abs: float,
                   params: V7Params) -> tuple[bool, str]:
    """Is this direction sensible given the regime?"""
    if not regime.tradeable:
        return False, f'regime not tradeable: {regime.reason}'

    if regime.regime is Regime.SQUEEZE and not params.trade_squeeze:
        return False, 'volatility squeeze — waiting for the release'

    # Fading an established trend needs materially stronger evidence.
    if regime.regime is Regime.TREND_UP and direction == 'short':
        if net_abs < params.counter_trend_min_net:
            return False, (f'short into an uptrend needs |net| ≥ '
                           f'{params.counter_trend_min_net:.2f}, got {net_abs:.2f}')
    if regime.regime is Regime.TREND_DOWN and direction == 'long':
        if net_abs < params.counter_trend_min_net:
            return False, (f'long into a downtrend needs |net| ≥ '
                           f'{params.counter_trend_min_net:.2f}, got {net_abs:.2f}')

    return True, ''


def _anchor_allows(symbol: str, direction: str, btc_trend: Optional[str],
                   params: V7Params) -> tuple[bool, str]:
    """
    Intermarket anchor, now symmetric.

    The old rule only blocked altcoin BUYs when BTC was falling and had no
    equivalent for the other side, because spot had no other side. On
    futures a falling BTC is not merely a reason to avoid longs — it is the
    single best backdrop for altcoin SHORTS, since alts have higher beta to
    BTC on the way down than up.
    """
    if not params.use_btc_anchor or config.is_anchor(symbol) or btc_trend is None:
        return True, ''

    if direction == 'long' and btc_trend == 'downtrend':
        return False, 'BTC downtrend — altcoin longs suppressed'
    if direction == 'short' and btc_trend == 'uptrend':
        return False, 'BTC uptrend — altcoin shorts suppressed'
    return True, ''


def decide(symbol: str,
           hist_1h: pd.DataFrame,
           hist_1d: Optional[pd.DataFrame] = None,
           ts: Optional[pd.Timestamp] = None,
           btc_trend: Optional[str] = None,
           funding: Optional[pd.Series] = None,
           open_interest: Optional[pd.Series] = None,
           params: Optional[V7Params] = None) -> V7Decision:
    """
    Full decision for one bar, with the reasoning retained.

    `hist_1h` / `hist_1d` must contain only bars that had CLOSED at `ts`.
    """
    params = params or DEFAULT_PARAMS
    decision = V7Decision(blocks={}, tags=[])

    # ── 1. Regime first — it decides what evidence to trust ───
    regime = classify_regime(hist_1h)
    decision.regime = regime
    if not regime.tradeable:
        decision.rejected = f'regime: {regime.reason}'
        return decision

    # ── 2. Indicator votes (the existing stack, unchanged) ────
    analysis = analyze_frames(symbol, hist_1h, hist_1d, btc_trend)
    if analysis is None:
        decision.rejected = 'insufficient data'
        return decision

    # ── 3. Derivatives context ────────────────────────────────
    derivatives = analyse_derivatives(
        funding, open_interest, hist_1h['close'],
        extreme_z=params.funding_extreme_z,
    )
    decision.derivatives = derivatives
    extra = {}
    deriv_score = _derivatives_block(derivatives, params)
    if deriv_score is not None:
        extra['derivatives'] = deriv_score

    # ── 4. Two-stage aggregation ──────────────────────────────
    aggregate = aggregate_blocks(analysis.votes, regime.block_weights, extra)
    decision.net_score = aggregate.net_score
    decision.blocks = {name: block.score for name, block in aggregate.blocks.items()}

    net_abs = abs(aggregate.net_score)
    if net_abs < params.min_net:
        decision.rejected = f'|net| {net_abs:.3f} < {params.min_net:.3f}'
        return decision

    direction = 'long' if aggregate.net_score > 0 else 'short'
    decision.agreement = agreement(aggregate, 1 if direction == 'long' else -1)

    # ── 5. Breadth: how many families actually spoke? ─────────
    active = sum(1 for b in aggregate.blocks.values() if b.n_votes)
    if active < params.min_active_blocks:
        decision.rejected = f'only {active} active blocks'
        return decision

    if decision.agreement < params.min_agreement:
        decision.rejected = (f'block agreement {decision.agreement:.0%} < '
                             f'{params.min_agreement:.0%} — evidence is split')
        return decision

    # ── 6. Context gates ──────────────────────────────────────
    ok, why = _regime_allows(direction, regime, net_abs, params)
    if not ok:
        decision.rejected = why
        return decision

    ok, why = _anchor_allows(symbol, direction, btc_trend, params)
    if not ok:
        decision.rejected = why
        return decision

    if params.use_kill_zones and ts is not None and not _in_kill_zone(ts):
        decision.rejected = f'{ts.hour:02d}:00 UTC is outside the session windows'
        return decision

    # A long paying extreme funding is negative carry on top of everything
    # else; over a 48h hold that is six settlements against the position.
    if params.use_funding_gate and derivatives.funding_extreme:
        if ((direction == 'long' and derivatives.funding_z > 0)
                or (direction == 'short' and derivatives.funding_z < 0)):
            if regime.regime not in (Regime.TREND_UP, Regime.TREND_DOWN):
                decision.rejected = (
                    f'funding z={derivatives.funding_z:+.1f} is crowded against a '
                    f'{direction} outside a trend')
                return decision

    # ── 7. Confidence ─────────────────────────────────────────
    confidence = _confidence(net_abs, decision.agreement, params)
    if confidence < params.confidence_floor:
        decision.rejected = (f'confidence {confidence:.0%} < floor '
                             f'{params.confidence_floor:.0%}')
        return decision

    decision.direction = direction
    decision.confidence = confidence
    decision.tags = _build_tags(aggregate, regime, derivatives, direction)
    decision.reason = (
        f'{direction.upper()} | {regime.regime.value} | net={aggregate.net_score:+.3f} '
        f'agree={decision.agreement:.0%} | {aggregate.summary()}'
    )
    return decision


def v7_intent(symbol: str,
              hist_1h: pd.DataFrame,
              hist_1d: Optional[pd.DataFrame] = None,
              ts: Optional[pd.Timestamp] = None,
              btc_trend: Optional[str] = None,
              funding: Optional[pd.Series] = None,
              open_interest: Optional[pd.Series] = None,
              params: Optional[V7Params] = None) -> Optional[Intent]:
    """Run `decide` and translate an actionable result into a sized Intent."""
    params = params or DEFAULT_PARAMS
    decision = decide(symbol, hist_1h, hist_1d, ts, btc_trend,
                      funding, open_interest, params)
    if decision.direction == 'flat':
        return None

    close = float(hist_1h['close'].iloc[-1])
    atr_result = calc_atr(hist_1h, period=config.ATR_PERIOD)
    atr_value = atr_result.value if atr_result is not None else None
    if not atr_value or atr_value <= 0:
        return None

    stop, take = risk.atr_stop_target(
        close, decision.direction, atr_value,
        stop_mult=params.atr_stop_mult,
        target_mult=params.atr_target_mult,
    )

    return Intent(
        direction=decision.direction,
        confidence=decision.confidence,
        stop_price=stop,
        take_price=take,
        atr=atr_value,
        reason=decision.reason[:200],
        tags=decision.tags,
    )


def _build_tags(aggregate, regime: RegimeState,
                derivatives: DerivativesState, direction: str) -> list[str]:
    """Attribution tags, so the edge log can say WHICH ideas actually pay."""
    tags = [f'regime:{regime.regime.value}']

    for name, block in aggregate.blocks.items():
        if block.n_votes and abs(block.score) >= 0.5:
            tags.append(f'{name}:{"bull" if block.score > 0 else "bear"}')

    if derivatives.available:
        if derivatives.funding_extreme:
            tags.append('funding:extreme')
        if derivatives.oi_signal not in ('unknown', 'flat'):
            tags.append(f'oi:{derivatives.oi_signal}')

    tags.append(f'dir:{direction}')
    return tags
