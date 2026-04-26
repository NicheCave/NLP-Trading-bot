# signal_generator.py — TA + SMC + Advanced TA signal generation (v3)
# ─────────────────────────────────────────────────────────────
# Changes from v2:
#   - Alligator sleeping VETO: if market_analyzer reports alligator_sleeping,
#     force HOLD regardless of other votes. Choppy market = no edge.
#   - Elliott Wave 5 WARNING: if wave5_warning flag is set on a BUY signal,
#     confidence is reduced by 40% and CRYPTO_RISK_PER_TRADE is signalled
#     to be halved (executor reads signal.wave5_caution flag).
#   - Trap signal override: if trap_active and trap_signal contradicts the
#     pending action, the signal is FLIPPED (Schwager's failed breakout rule).
#   - AVWAP slope filter: BUY signals where AVWAP slope_filter is False get
#     a 20% confidence reduction (checked via vote reason string).
#   - Pyramiding flag added to Signal dataclass: signal.add_to_winner
#     is True when strong trend conditions are met on an existing position
#     → executor.py uses this to scale in 50% more (Bellafiore pyramiding).
# ─────────────────────────────────────────────────────────────
from dataclasses import dataclass, field
from typing import Literal, Optional
from loguru import logger
import config
from market_analyzer import analyze, AnalysisResult


@dataclass
class Signal:
    ticker: str
    action: Literal['BUY', 'SELL', 'HOLD']
    confidence: float       # 0.0–1.0 — scales position size in executor.py
    avg_score: float        # net_score (-1.0 to +1.0)
    bull_pct: float
    bear_pct: float
    article_count: int      # number of indicator votes cast
    reason: str
    # ── v3 additions ──────────────────────────────────────────
    wave5_caution: bool = False    # True → executor halves CRYPTO_RISK_PER_TRADE
    add_to_winner: bool = False    # True → executor scales in (pyramiding)
    trap_flip: bool = False        # True → signal was flipped from opposite direction


_EPSILON = 0.001


def _has_vote(analysis: AnalysisResult, name: str) -> bool:
    return any(v.name == name and v.vote != 0 for v in analysis.votes)


def _get_vote(analysis: AnalysisResult, name: str) -> int:
    for v in analysis.votes:
        if v.name == name:
            return v.vote
    return 0


def _get_reason(analysis: AnalysisResult, name: str) -> str:
    for v in analysis.votes:
        if v.name == name:
            return v.reason
    return ''


def _alligator_is_sleeping(analysis: AnalysisResult) -> bool:
    """Check if the Alligator is sleeping (market too choppy to trade)."""
    # Check via the analysis flag (v3 AnalysisResult extension)
    if hasattr(analysis, 'alligator_sleeping') and analysis.alligator_sleeping:
        return True
    # Fallback: check vote reason string
    for v in analysis.votes:
        if v.name == 'williams_alligator' and 'SLEEPING' in v.reason.upper():
            return True
    return False


def _wave5_warning_active(analysis: AnalysisResult) -> bool:
    """Check if Elliott Wave 5 divergence warning is active."""
    if hasattr(analysis, 'wave5_warning') and analysis.wave5_warning:
        return True
    for v in analysis.votes:
        if v.name == 'elliott_divergence' and 'Wave 5' in v.reason:
            return True
    return False


def _trap_contradicts(analysis: AnalysisResult, action: str) -> bool:
    """
    Check if the trap detector is flipping the signal.
    A bull trap contradicts a BUY. A bear trap contradicts a SELL.
    """
    if hasattr(analysis, 'trap_active') and analysis.trap_active:
        trap_sig = getattr(analysis, 'trap_signal', 'none')
        if action == 'BUY'  and trap_sig == 'sell':
            return True
        if action == 'SELL' and trap_sig == 'buy':
            return True
    # Fallback via vote
    trap_vote = _get_vote(analysis, 'trap_detector')
    if action == 'BUY'  and trap_vote == -1:
        return True
    if action == 'SELL' and trap_vote == +1:
        return True
    return False


def _avwap_slope_filter_failed(analysis: AnalysisResult) -> bool:
    """True if the AVWAP breakout occurred but the slope filter was NOT met."""
    reason = _get_reason(analysis, 'avwap_pinch')
    return 'slope filter ✗' in reason


def _should_pyramid(analysis: AnalysisResult) -> bool:
    """
    Bellafiore pyramiding: return True when conditions are strong enough
    to add to an existing winning position.

    Conditions required (ALL must be true):
    1. Alligator is eating (not sleeping or sated)
    2. AO is bullish (above zero)
    3. AVWAP breakout confirmed (slope filter met)
    4. RVOL is in-play (smart money present)
    5. No Wave 5 warning
    6. No trap detected
    """
    gator_eating = _get_vote(analysis, 'williams_alligator') == +1
    ao_bullish   = _get_vote(analysis, 'williams_ao') in (+1,)
    avwap_ok     = _get_vote(analysis, 'avwap_pinch') == +1 and not _avwap_slope_filter_failed(analysis)
    rvol_ok      = _get_vote(analysis, 'rvol_filter') != 0   # any non-zero = in play
    no_wave5     = not _wave5_warning_active(analysis)
    no_trap      = _get_vote(analysis, 'trap_detector') == 0

    return gator_eating and ao_bullish and avwap_ok and rvol_ok and no_wave5 and no_trap


def generate_signal(ticker: str) -> Signal:
    """
    Run multi-timeframe TA + SMC + Advanced TA analysis and produce a
    BUY / SELL / HOLD signal with confidence score.
    """
    analysis = analyze(ticker)

    if analysis is None:
        return Signal(
            ticker, 'HOLD', 0.0, 0.0, 0.0, 0.0, 0,
            'Analysis failed — insufficient market data'
        )

    n_votes = len(analysis.votes)
    if n_votes < 4:
        return Signal(
            ticker, 'HOLD', 0.0, analysis.net_score,
            analysis.bull_score, analysis.bear_score, n_votes,
            f'Only {n_votes} indicator votes — need at least 4'
        )

    net    = analysis.net_score
    bull   = analysis.bull_score
    bear   = analysis.bear_score
    reason = analysis.summary

    # ── ALLIGATOR SLEEPING VETO ───────────────────────────────
    # Bill Williams: if the Alligator is sleeping, the market has no trend
    # and all signals are noise. Hard HOLD — no exceptions.
    if _alligator_is_sleeping(analysis):
        hold_reason = f'[ALLIGATOR SLEEPING] Market choppy — all signals suppressed. {reason}'
        logger.info(f"HOLD {ticker}: Alligator veto — {hold_reason}")
        return Signal(
            ticker, 'HOLD', 0.0, net, bull, bear, n_votes, hold_reason
        )

    # ── Bear market override ───────────────────────────────────
    in_bear_market = analysis.trend == 'downtrend'

    buy_threshold  = config.CRYPTO_BUY_THRESHOLD
    sell_threshold = config.CRYPTO_SELL_THRESHOLD
    min_net        = config.CRYPTO_MIN_AVG_SCORE

    if in_bear_market:
        buy_threshold = min(0.55, buy_threshold + 0.20)
        min_net       = min_net + 0.10
        reason        = f'[BEAR MKT] {reason}'
        logger.debug(f"{ticker}: bear market — raising BUY threshold to {buy_threshold:.0%}")

    # ── SMC confluence flags ───────────────────────────────────
    has_ob_vote    = _has_vote(analysis, 'smc_order_block')
    has_fvg_vote   = _has_vote(analysis, 'smc_fvg')
    has_bos_vote   = _has_vote(analysis, 'smc_bos_choch')
    has_sweep_vote = _has_vote(analysis, 'smc_liquidity')
    choch_bullish  = _get_vote(analysis, 'smc_bos_choch') == +1 and \
                     any('CHoCH' in v.reason for v in analysis.votes if v.name == 'smc_bos_choch')
    choch_bearish  = _get_vote(analysis, 'smc_bos_choch') == -1 and \
                     any('CHoCH' in v.reason for v in analysis.votes if v.name == 'smc_bos_choch')
    sweep_bullish  = _get_vote(analysis, 'smc_liquidity') == +1
    sweep_bearish  = _get_vote(analysis, 'smc_liquidity') == -1

    # ── BUY signal ────────────────────────────────────────────
    if bull >= buy_threshold - _EPSILON and net >= min_net - _EPSILON:
        confidence = min(1.0, max(0.3, net * 2.5))
        trap_flip  = False

        # Schwager Trap Override: if a bull trap is detected against this
        # BUY signal, flip to SELL immediately. Failed breakout = fastest move.
        if _trap_contradicts(analysis, 'BUY'):
            trap_reason = _get_reason(analysis, 'trap_detector')
            logger.warning(
                f"TRAP FLIP {ticker}: BUY signal overridden by bull trap → SELL | {trap_reason}"
            )
            confidence = min(1.0, max(0.4, abs(net) * 2.5 * 1.2))  # traps = high confidence
            return Signal(
                ticker, 'SELL', confidence, net, bull, bear, n_votes,
                f'[TRAP FLIP] Bull trap detected — {trap_reason} | was BUY',
                trap_flip=True,
            )

        # CHoCH bullish reversal boost
        if choch_bullish:
            confidence = min(1.0, confidence * 1.20)
            reason += ' | CHoCH bullish reversal'

        # Liquidity sweep confirmation
        if sweep_bullish:
            confidence = min(1.0, confidence * 1.15)
            reason += ' | liquidity sweep confirms entry'

        # SMC count bonus
        smc_bull_count = sum([
            _get_vote(analysis, 'smc_order_block') == +1,
            _get_vote(analysis, 'smc_fvg') == +1,
            has_bos_vote and _get_vote(analysis, 'smc_bos_choch') == +1,
        ])
        if smc_bull_count >= 2:
            confidence = min(1.0, confidence * 1.10)
            reason += f' | {smc_bull_count} SMC aligned'

        # v3: Alligator eating confirmation → small boost
        if _get_vote(analysis, 'williams_alligator') == +1:
            confidence = min(1.0, confidence * 1.08)
            reason += ' | Alligator eating bull'

        # v3: AVWAP breakout with slope filter → boost
        if _get_vote(analysis, 'avwap_pinch') == +1 and not _avwap_slope_filter_failed(analysis):
            confidence = min(1.0, confidence * 1.10)
            reason += ' | AVWAP pinch breakout (slope ✓)'
        elif _avwap_slope_filter_failed(analysis):
            confidence *= 0.80
            reason += ' | AVWAP slope filter ✗ (reduced)'

        # v3: AO bullish cross → additional momentum confirmation
        ao_vote = _get_vote(analysis, 'williams_ao')
        if ao_vote == +1 and 'zero-line cross' in _get_reason(analysis, 'williams_ao'):
            confidence = min(1.0, confidence * 1.07)
            reason += ' | AO zero-line cross'

        # v3: Fractal breakout confirmed
        if _get_vote(analysis, 'williams_fractals') == +1:
            confidence = min(1.0, confidence * 1.05)
            reason += ' | fractal structure confirmed'

        # v3: Wave 5 warning — REDUCE RISK (not a veto, but a caution)
        wave5_caution = _wave5_warning_active(analysis)
        if wave5_caution:
            confidence *= 0.60   # cut conviction significantly
            reason += ' | ⚠ Elliott Wave 5 WARNING — reduced size'
            logger.warning(f"{ticker}: Wave 5 divergence detected — reducing confidence to {confidence:.0%}")

        if in_bear_market:
            confidence *= 0.65
            reason += ' | confidence reduced — bear market context'

        # v3: Pyramiding flag — should we add to an existing winner?
        pyramid = _should_pyramid(analysis) and not wave5_caution and not in_bear_market

        logger.success(
            f"BUY signal: {ticker} | bull={bull:.0%} net={net:+.3f} "
            f"conf={confidence:.0%} pyramid={pyramid} | {reason}"
        )
        return Signal(
            ticker, 'BUY', confidence, net, bull, bear, n_votes, reason,
            wave5_caution=wave5_caution,
            add_to_winner=pyramid,
        )

    # ── SELL signal ───────────────────────────────────────────
    if bear >= sell_threshold - _EPSILON and net <= -(min_net - _EPSILON):
        confidence = min(1.0, max(0.3, abs(net) * 2.5))
        trap_flip  = False

        # Schwager Trap Override: bear trap detected → flip to BUY
        if _trap_contradicts(analysis, 'SELL'):
            trap_reason = _get_reason(analysis, 'trap_detector')
            logger.warning(
                f"TRAP FLIP {ticker}: SELL signal overridden by bear trap → BUY | {trap_reason}"
            )
            confidence = min(1.0, max(0.4, net * 2.5 * 1.2))
            return Signal(
                ticker, 'BUY', confidence, net, bull, bear, n_votes,
                f'[TRAP FLIP] Bear trap detected — {trap_reason} | was SELL',
                trap_flip=True,
            )

        if choch_bearish:
            confidence = min(1.0, confidence * 1.20)
            reason += ' | CHoCH bearish reversal'

        if sweep_bearish:
            confidence = min(1.0, confidence * 1.15)
            reason += ' | liquidity sweep confirms sell'

        smc_bear_count = sum([
            _get_vote(analysis, 'smc_order_block') == -1,
            _get_vote(analysis, 'smc_fvg') == -1,
            has_bos_vote and _get_vote(analysis, 'smc_bos_choch') == -1,
        ])
        if smc_bear_count >= 2:
            confidence = min(1.0, confidence * 1.10)
            reason += f' | {smc_bear_count} SMC aligned'

        # v3: Alligator eating bear → boost SELL confidence
        if _get_vote(analysis, 'williams_alligator') == -1:
            confidence = min(1.0, confidence * 1.08)
            reason += ' | Alligator eating bear'

        # v3: AVWAP bearish breakout
        if _get_vote(analysis, 'avwap_pinch') == -1:
            confidence = min(1.0, confidence * 1.08)
            reason += ' | AVWAP pinch bearish breakout'

        logger.success(
            f"SELL signal: {ticker} | bear={bear:.0%} net={net:+.3f} "
            f"conf={confidence:.0%} | {reason}"
        )
        return Signal(ticker, 'SELL', confidence, net, bull, bear, n_votes, reason)

    # ── HOLD ──────────────────────────────────────────────────
    hold_reason = (
        f"bull={bull:.0%} bear={bear:.0%} net={net:+.3f} — "
        f"thresholds not met (need bull≥{buy_threshold:.0%} or bear≥{sell_threshold:.0%})"
    )
    logger.info(f"HOLD {ticker}: {hold_reason}")
    return Signal(
        ticker, 'HOLD', 0.0, net, bull, bear, n_votes, hold_reason
    )