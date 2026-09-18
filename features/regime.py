# features/regime.py — what KIND of market are we in?
# ─────────────────────────────────────────────────────────────
# The single biggest avoidable loss source in the previous version was
# running one strategy in every condition. Trend-following logic (EMA
# crosses, BOS continuation, AVWAP breakouts) is *designed* to be wrong in
# a range: it buys the top of every swing and sells the bottom. Choppy
# markets are where win rate collapses, not because the indicators broke
# but because they were being asked the wrong question.
#
# The old code's only defence was the Alligator "sleeping" veto — a single
# binary check on one indicator's line spread. This replaces it with three
# orthogonal measurements that have to agree:
#
#   ADX               is there a trend worth following?
#   Efficiency Ratio  is price going somewhere, or just moving?
#   Realised vol %ile is volatility survivable, or is this a panic?
#
# Regime does not pick a direction. It picks which evidence to trust.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd

from indicators import adx, efficiency_ratio, realised_volatility, squeeze


class Regime(str, Enum):
    TREND_UP   = 'trend_up'
    TREND_DOWN = 'trend_down'
    RANGE      = 'range'
    VOLATILE   = 'volatile_chop'   # moving hard, going nowhere — stand down
    SQUEEZE    = 'squeeze'         # compressed, expansion pending
    UNKNOWN    = 'unknown'


# How much each factor block is trusted, per regime.
#
# The point of this table: in a range, trend and momentum evidence is
# actively misleading, so it is down-weighted rather than merely ignored,
# while mean-reversion and structure carry the decision. In a trend the
# reverse holds — fading a strong trend because RSI reads "overbought" is
# how the old stack sold into the strongest moves of 2024-25.
REGIME_BLOCK_WEIGHTS: dict[Regime, dict[str, float]] = {
    Regime.TREND_UP: {
        'trend': 1.30, 'momentum': 1.20, 'structure': 1.00,
        'mean_reversion': 0.30, 'volume': 1.00, 'context': 1.00, 'reversal': 0.80,
    },
    Regime.TREND_DOWN: {
        'trend': 1.30, 'momentum': 1.20, 'structure': 1.00,
        'mean_reversion': 0.30, 'volume': 1.00, 'context': 1.00, 'reversal': 0.80,
    },
    Regime.RANGE: {
        'trend': 0.35, 'momentum': 0.60, 'structure': 1.20,
        'mean_reversion': 1.40, 'volume': 0.90, 'context': 1.00, 'reversal': 1.20,
    },
    Regime.SQUEEZE: {
        'trend': 0.70, 'momentum': 0.90, 'structure': 1.30,
        'mean_reversion': 0.60, 'volume': 1.30, 'context': 1.00, 'reversal': 0.90,
    },
    # Deliberately flat and low: in volatile chop nothing is trusted much.
    Regime.VOLATILE: {
        'trend': 0.50, 'momentum': 0.50, 'structure': 0.60,
        'mean_reversion': 0.50, 'volume': 0.70, 'context': 1.00, 'reversal': 0.70,
    },
    Regime.UNKNOWN: {
        'trend': 1.00, 'momentum': 1.00, 'structure': 1.00,
        'mean_reversion': 1.00, 'volume': 1.00, 'context': 1.00, 'reversal': 1.00,
    },
}


@dataclass
class RegimeState:
    regime:        Regime = Regime.UNKNOWN
    adx:           float = 0.0
    efficiency:    float = 0.0
    vol_percentile: float = 0.5
    directional_bias: int = 0        # +1 / -1 / 0, from +DI vs -DI
    tradeable:     bool = True       # False → stand down entirely
    reason:        str = ''
    block_weights: dict[str, float] = field(default_factory=dict)

    def weight_for(self, block: str) -> float:
        return self.block_weights.get(block, 1.0)


def classify_regime(df_1h: pd.DataFrame,
                    adx_threshold: float = 22.0,
                    efficiency_threshold: float = 0.35,
                    vol_panic_percentile: float = 0.92) -> RegimeState:
    """
    Classify the current market regime from 1H bars.

    Order of checks matters. Volatility panic is tested first because in a
    genuine liquidation cascade every other reading is unreliable — ADX can
    look strongly "trending" during a waterfall that reverses 15% in an
    hour, and that is precisely when a trend entry is worst.
    """
    if df_1h is None or len(df_1h) < 80:
        return RegimeState(reason='insufficient bars', tradeable=False,
                           block_weights=REGIME_BLOCK_WEIGHTS[Regime.UNKNOWN])

    adx_result = adx(df_1h, threshold=adx_threshold)
    eff_result = efficiency_ratio(df_1h, threshold=efficiency_threshold)
    vol_result = realised_volatility(df_1h)
    sqz_result = squeeze(df_1h)

    adx_value = adx_result.adx if adx_result else 0.0
    efficiency = eff_result.ratio if eff_result else 0.0
    vol_pct = vol_result.percentile if vol_result else 0.5
    bias = 0
    if adx_result:
        bias = 1 if adx_result.bullish else -1

    def build(regime: Regime, tradeable: bool, reason: str) -> RegimeState:
        return RegimeState(
            regime=regime, adx=adx_value, efficiency=efficiency,
            vol_percentile=vol_pct, directional_bias=bias,
            tradeable=tradeable, reason=reason,
            block_weights=dict(REGIME_BLOCK_WEIGHTS[regime]),
        )

    # 1. Volatility panic — stand down regardless of what else reads well.
    if vol_pct >= vol_panic_percentile and efficiency < efficiency_threshold:
        return build(Regime.VOLATILE, False,
                     f'vol at {vol_pct:.0%} percentile with efficiency {efficiency:.2f} '
                     f'— violent but directionless')

    # 2. A genuine trend needs BOTH strength and efficiency. ADX alone is
    #    routinely high in a wide, whipsawing range.
    trending = (adx_result is not None and adx_result.trending
                and eff_result is not None and eff_result.trending)
    if trending:
        regime = Regime.TREND_UP if adx_result.bullish else Regime.TREND_DOWN
        return build(regime, True,
                     f'ADX {adx_value:.0f} with efficiency {efficiency:.2f} '
                     f'— {"up" if adx_result.bullish else "down"}trend')

    # 3. Compression: nothing to trade yet, but the release is worth waiting for.
    if sqz_result is not None and sqz_result.squeezed:
        return build(Regime.SQUEEZE, True,
                     f'volatility squeeze (bandwidth {sqz_result.bandwidth:.2%}) '
                     f'— expansion pending')

    # 4. Otherwise: a range. Tradeable, but only by mean-reversion logic.
    return build(Regime.RANGE, True,
                 f'ADX {adx_value:.0f}, efficiency {efficiency:.2f} — ranging')
