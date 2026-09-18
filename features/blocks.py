# features/blocks.py — decorrelated factor aggregation.
# ─────────────────────────────────────────────────────────────
# The previous aggregation summed all 24 indicator votes as though they
# were independent evidence:
#
#     net = (Σ bull weights − Σ bear weights) / Σ all weights
#
# They are not independent. ema_cross, macd, williams_ao, williams_alligator
# and mtf_daily_align are five different ways of measuring ONE thing —
# whether price has been going up lately. In a strong trend all five fire
# together and contribute 8.5 of ~30 total weight, so bull_score clears the
# 0.35 threshold on momentum alone.
#
# That is worst exactly when it matters most: the five agree most strongly
# when a move is already extended, so the stack produces its most confident
# entries at the point of maximum risk. It is a structural late-entry bias,
# not a tuning problem, and no threshold change fixes it.
#
# The fix: aggregate WITHIN a factor family first (so five momentum
# indicators produce at most one momentum opinion), then across families.
# Correlated indicators now buy conviction inside their own block rather
# than extra weight in the total.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

# Which factor family each indicator belongs to.
FACTOR_BLOCKS: dict[str, str] = {
    # ── trend: is there a prevailing direction? ───────────────
    'ema_cross':          'trend',
    'mtf_daily_align':    'trend',
    'williams_alligator': 'trend',

    # ── momentum: how fast, and is it accelerating? ───────────
    'macd':               'momentum',
    'williams_ao':        'momentum',

    # ── mean reversion: how stretched from fair value? ────────
    # RSI sits here rather than in momentum because its high-conviction
    # readings are the extremes. The regime router then correctly damps it
    # inside trends, which is what stops "RSI 72 → sell" from fading the
    # strongest legs of a bull market.
    'bollinger':          'mean_reversion',
    'rsi':                'mean_reversion',

    # ── structure: where is price relative to levels? ─────────
    'smc_order_block':    'structure',
    'smc_fvg':            'structure',
    'smc_bos_choch':      'structure',
    'smc_liquidity':      'structure',
    'williams_fractals':  'structure',
    'avwap_pinch':        'structure',

    # ── volume: is anyone actually participating? ─────────────
    'rvol_filter':        'volume',

    # ── reversal: explicit counter-evidence ───────────────────
    'trap_detector':      'reversal',
    'elliott_divergence': 'reversal',

    # ── context: the wider market ─────────────────────────────
    'btc_anchor':         'context',
    'atr_trend':          'context',
}

# How much each family matters before the regime router adjusts it.
BLOCK_BASE_WEIGHTS: dict[str, float] = {
    'trend':          1.00,
    'momentum':       0.80,
    'mean_reversion': 0.70,
    'structure':      1.20,   # SMC levels are the stack's most specific edge
    'volume':         0.60,
    'reversal':       1.00,
    'context':        0.90,
    'derivatives':    0.90,   # futures-only, injected separately
}


@dataclass
class BlockScore:
    name:      str
    score:     float           # -1..+1, weighted mean of its votes
    weight:    float           # base × regime multiplier
    n_votes:   int
    contributors: list[str] = field(default_factory=list)


@dataclass
class BlockAggregate:
    net_score:  float = 0.0     # -1..+1
    bull_score: float = 0.0     # 0..1
    bear_score: float = 0.0     # 0..1
    blocks:     dict[str, BlockScore] = field(default_factory=dict)

    def block(self, name: str) -> Optional[BlockScore]:
        return self.blocks.get(name)

    def summary(self) -> str:
        parts = [f"{b.name}={b.score:+.2f}" for b in self.blocks.values() if b.n_votes]
        return ' '.join(parts) or 'no blocks'


def aggregate_blocks(votes: Iterable,
                     regime_weights: Optional[dict[str, float]] = None,
                     extra_blocks: Optional[dict[str, float]] = None) -> BlockAggregate:
    """
    Two-stage aggregation: within factor families, then across them.

    Args:
        votes:          IndicatorVote objects (name, vote, weight, reason)
        regime_weights: per-block multipliers from the regime router
        extra_blocks:   pre-computed block scores in -1..+1 (e.g. derivatives),
                        merged in as if they were their own family

    Returns a BlockAggregate whose net_score is on the same -1..+1 scale as
    the old one, so thresholds stay comparable — but is no longer inflated
    by correlated indicators agreeing with themselves.
    """
    regime_weights = regime_weights or {}

    # ── stage 1: within each family ───────────────────────────
    grouped: dict[str, list] = {}
    for vote in votes:
        block = FACTOR_BLOCKS.get(vote.name)
        if block is None:
            continue
        grouped.setdefault(block, []).append(vote)

    blocks: dict[str, BlockScore] = {}
    for name, block_votes in grouped.items():
        total_weight = sum(v.weight for v in block_votes)
        if total_weight <= 0:
            continue

        # Weighted mean of directional votes, bounded to [-1, +1] no matter
        # how many indicators live in the family.
        score = sum(v.vote * v.weight for v in block_votes) / total_weight
        weight = BLOCK_BASE_WEIGHTS.get(name, 1.0) * regime_weights.get(name, 1.0)

        blocks[name] = BlockScore(
            name=name,
            score=max(-1.0, min(1.0, score)),
            weight=weight,
            n_votes=sum(1 for v in block_votes if v.vote != 0),
            contributors=[v.name for v in block_votes if v.vote != 0],
        )

    # ── merge externally-computed blocks ──────────────────────
    for name, score in (extra_blocks or {}).items():
        blocks[name] = BlockScore(
            name=name,
            score=max(-1.0, min(1.0, score)),
            weight=BLOCK_BASE_WEIGHTS.get(name, 1.0) * regime_weights.get(name, 1.0),
            n_votes=1 if score else 0,
            contributors=[name],
        )

    # ── stage 2: across families ──────────────────────────────
    total_weight = sum(b.weight for b in blocks.values())
    if total_weight <= 0:
        return BlockAggregate(blocks=blocks)

    net = sum(b.score * b.weight for b in blocks.values()) / total_weight
    bull = sum(max(0.0, b.score) * b.weight for b in blocks.values()) / total_weight
    bear = sum(max(0.0, -b.score) * b.weight for b in blocks.values()) / total_weight

    return BlockAggregate(net_score=net, bull_score=bull, bear_score=bear, blocks=blocks)


def agreement(aggregate: BlockAggregate, direction: int) -> float:
    """
    Fraction of weighted block evidence pointing the same way.

    Genuinely independent families agreeing is worth far more than one
    family shouting. This is the number that should drive conviction —
    it is what the old confidence stack was reaching for with its chain of
    ×1.20 / ×1.15 / ×1.10 multipliers, but computed once and bounded.
    """
    active = [b for b in aggregate.blocks.values() if b.n_votes and b.weight > 0]
    if not active:
        return 0.0

    total = sum(b.weight for b in active)
    agreeing = sum(b.weight for b in active if b.score * direction > 0)
    return agreeing / total if total else 0.0
