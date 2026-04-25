# signal_generator.py — TA + SMC signal generation (v2)
# ─────────────────────────────────────────────────────────────
# Changes from v1:
#   - HOLD confidence fixed to 0.0 (was hardcoded 0.5 — misleading in logs)
#   - Float comparison epsilon added to prevent 0.3499... missing threshold
#   - CHoCH events (Change of Character) upgrade confidence by 20% — they
#     are reversal signals and deserve higher weight than plain threshold pass
#   - Liquidity sweep confluence bonus: if a sweep vote is present in the
#     analysis, confidence is boosted by 15% (sweeps are the highest-quality
#     SMC entry signal)
#   - Bear market confidence reduction tuned: 0.5× → 0.65× (was too punishing)
#   - Signal reason now includes top SMC context when present
# ─────────────────────────────────────────────────────────────
from dataclasses import dataclass
from typing import Literal
from loguru import logger
import config
from market_analyzer import analyze, AnalysisResult


@dataclass
class Signal:
    ticker: str
    action: Literal['BUY', 'SELL', 'HOLD']
    confidence: float    # 0.0–1.0 — scales position size in executor.py
    avg_score: float     # net_score (-1.0 to +1.0)
    bull_pct: float      # bull_score (0.0–1.0)
    bear_pct: float      # bear_score (0.0–1.0)
    article_count: int   # repurposed: number of indicator votes cast
    reason: str


_EPSILON = 0.001  # float comparison tolerance — prevents 0.3499... missing 0.35 threshold


def _has_vote(analysis: AnalysisResult, name: str) -> bool:
    """Check if a named indicator cast a non-zero vote."""
    return any(v.name == name and v.vote != 0 for v in analysis.votes)


def _get_vote(analysis: AnalysisResult, name: str) -> int:
    """Return the vote value for a named indicator, or 0 if absent."""
    for v in analysis.votes:
        if v.name == name:
            return v.vote
    return 0


def generate_signal(ticker: str) -> Signal:
    """
    Run multi-timeframe TA + SMC analysis and produce a BUY / SELL / HOLD
    signal with confidence score.
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
    # These modify confidence AFTER a signal threshold is passed.
    has_ob_vote      = _has_vote(analysis, 'smc_order_block')
    has_fvg_vote     = _has_vote(analysis, 'smc_fvg')
    has_bos_vote     = _has_vote(analysis, 'smc_bos_choch')
    has_sweep_vote   = _has_vote(analysis, 'smc_liquidity')
    choch_bullish    = _get_vote(analysis, 'smc_bos_choch') == +1 and \
                       any('CHoCH' in v.reason for v in analysis.votes if v.name == 'smc_bos_choch')
    choch_bearish    = _get_vote(analysis, 'smc_bos_choch') == -1 and \
                       any('CHoCH' in v.reason for v in analysis.votes if v.name == 'smc_bos_choch')
    sweep_bullish    = _get_vote(analysis, 'smc_liquidity') == +1
    sweep_bearish    = _get_vote(analysis, 'smc_liquidity') == -1

    # ── BUY signal ────────────────────────────────────────────
    if bull >= buy_threshold - _EPSILON and net >= min_net - _EPSILON:
        confidence = min(1.0, max(0.3, net * 2.5))

        # CHoCH bullish = change of character from bearish to bullish
        # This is a reversal signal — boost confidence
        if choch_bullish:
            confidence = min(1.0, confidence * 1.20)
            reason += ' | CHoCH bullish reversal'

        # Liquidity sweep confirming the bull move
        if sweep_bullish:
            confidence = min(1.0, confidence * 1.15)
            reason += ' | liquidity sweep confirms entry'

        # SMC count bonus: each additional SMC signal aligning adds a small boost
        smc_bull_count = sum([
            _get_vote(analysis, 'smc_order_block') == +1,
            _get_vote(analysis, 'smc_fvg') == +1,
            has_bos_vote and _get_vote(analysis, 'smc_bos_choch') == +1,
        ])
        if smc_bull_count >= 2:
            confidence = min(1.0, confidence * 1.10)
            reason += f' | {smc_bull_count} SMC signals aligned'

        if in_bear_market:
            confidence *= 0.65   # was 0.6 — slightly less punishing
            reason += ' | confidence reduced — bear market context'

        logger.success(
            f"BUY signal: {ticker} | bull={bull:.0%} net={net:+.3f} "
            f"conf={confidence:.0%} | {reason}"
        )
        return Signal(ticker, 'BUY', confidence, net, bull, bear, n_votes, reason)

    # ── SELL signal ───────────────────────────────────────────
    if bear >= sell_threshold - _EPSILON and net <= -(min_net - _EPSILON):
        confidence = min(1.0, max(0.3, abs(net) * 2.5))

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
            reason += f' | {smc_bear_count} SMC signals aligned'

        logger.success(
            f"SELL signal: {ticker} | bear={bear:.0%} net={net:+.3f} "
            f"conf={confidence:.0%} | {reason}"
        )
        return Signal(ticker, 'SELL', confidence, net, bull, bear, n_votes, reason)

    # ── HOLD ──────────────────────────────────────────────────
    # FIX: confidence set to 0.0 (was 0.5, which was meaningless and confusing in logs)
    hold_reason = (
        f"bull={bull:.0%} bear={bear:.0%} net={net:+.3f} — "
        f"thresholds not met (need bull≥{buy_threshold:.0%} or bear≥{sell_threshold:.0%})"
    )
    logger.info(f"HOLD {ticker}: {hold_reason}")
    return Signal(
        ticker, 'HOLD', 0.0, net, bull, bear, n_votes, hold_reason
    )