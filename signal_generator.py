# signal_generator.py — aggregates sentiment → BUY / SELL / HOLD
# ─────────────────────────────────────────────────────────────
from dataclasses import dataclass
from typing import Literal
from loguru import logger
import statistics
import config
from data_fetcher import fetch_all_text
from sentiment_engine import score_batch


@dataclass
class Signal:
    ticker: str
    action: Literal[
        'BUY', 'SELL', 'HOLD']
    confidence: float  # 0.0–1.0 — used to scale position size
    avg_score: float  # average sentiment score across all texts
    bull_pct: float  # % of articles that are bullish
    bear_pct: float  # % of articles that are bearish
    article_count: int
    reason: str  # human-readable explanation for logging


def generate_signal(ticker: str) -> Signal:
    texts = fetch_all_text(ticker)
    results = score_batch(texts)

    if len(results) < config.MIN_ARTICLES:
        return Signal(ticker, 'HOLD', 0.0, 0.0, 0.0, 0.0, len(results),
                      f"Only {len(results)} articles — need {config.MIN_ARTICLES} minimum")

    scores    = [r.score for r in results]
    avg_score = statistics.mean(scores)
    total     = len(results)
    bulls     = sum(1 for r in results if r.label == 'bullish')
    bears     = sum(1 for r in results if r.label == 'bearish')
    bull_pct  = bulls / total
    bear_pct  = bears / total

    logger.info(
        f"{ticker}: {total} articles | bulls={bull_pct:.0%} bears={bear_pct:.0%} avg_score={avg_score:+.3f}"
    )

    # ── Pick thresholds based on asset type ───────────────────
    from config import is_crypto
    buy_threshold  = config.CRYPTO_BUY_THRESHOLD  if is_crypto(ticker) else config.BUY_THRESHOLD
    sell_threshold = config.CRYPTO_SELL_THRESHOLD if is_crypto(ticker) else config.SELL_THRESHOLD
    min_avg_score  = config.CRYPTO_MIN_AVG_SCORE  if is_crypto(ticker) else config.MIN_AVG_SCORE

    if bull_pct >= buy_threshold and avg_score >= min_avg_score:
        # Use avg_score as confidence proxy, clamped to 0.3–1.0
        confidence = min(1.0, max(0.3, avg_score * 3))
        return Signal(ticker, 'BUY', confidence, avg_score, bull_pct, bear_pct, total,
                      f"{bull_pct:.0%} bullish articles, avg score={avg_score:+.3f}")

    elif bear_pct >= sell_threshold and avg_score <= -min_avg_score:
        return Signal(ticker, 'SELL', bear_pct, avg_score, bull_pct, bear_pct, total,
                      f"{bear_pct:.0%} bearish articles, avg score={avg_score:+.3f}")

    return Signal(ticker, 'HOLD', 0.5, avg_score, bull_pct, bear_pct, total,
                  "Mixed or insufficient sentiment strength")