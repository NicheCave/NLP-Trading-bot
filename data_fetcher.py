# data_fetcher.py — NewsAPI + Finnhub (replaces StockTwits/Reddit)
# ─────────────────────────────────────────────────────────────
# Finnhub: free tier, 60 calls/min, company news + pre-scored sentiment
# NewsAPI: free tier, 100 requests/day
# ─────────────────────────────────────────────────────────────
import finnhub
from newsapi import NewsApiClient
from loguru import logger
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
import os
import config

# ── Initialise clients once at import ────────────────────────
_newsapi      = NewsApiClient(api_key=config.NEWS_API_KEY)
_finnhub      = finnhub.Client(api_key=os.getenv('FINNHUB_API_KEY'))

# ── Keyword map for NewsAPI (Finnhub uses ticker directly) ───
TICKER_KEYWORDS = {
    'TSLA':   ['TSLA', 'Tesla'],
    'NVDA':   ['NVDA', 'Nvidia'],
    'AAPL':   ['AAPL', 'Apple stock'],
    'MSFT':   ['MSFT', 'Microsoft'],
    'BTCUSD': ['Bitcoin', 'BTC'],
    'ETHUSD': ['Ethereum', 'ETH'],
}

# ─────────────────────────────────────────────────────────────
# FINNHUB NEWS DATACLASS
# Stores the headline text AND Finnhub's own pre-computed
# sentiment score (-1 to +1) — double signal like StockTwits had.
# ─────────────────────────────────────────────────────────────
@dataclass
class FinnhubArticle:
    headline:        str
    summary:         str            # longer article summary if available
    finnhub_score:   Optional[float]  # pre-computed sentiment -1.0 to +1.0
    source:          str

# ─────────────────────────────────────────────────────────────
# SOURCE 1: NEWSAPI
# ─────────────────────────────────────────────────────────────
def fetch_news(ticker: str) -> list[str]:
    """Fetch recent financial headlines from NewsAPI."""
    keywords = TICKER_KEYWORDS.get(ticker, [ticker])
    query    = ' OR '.join(keywords)
    try:
        resp = _newsapi.get_everything(
            q         = query,
            language  = 'en',
            sort_by   = 'publishedAt',
            page_size = config.NEWS_ARTICLES
        )
        headlines = [
            a['title'] for a in resp.get('articles', [])
            if a.get('title') and a['title'] != '[Removed]'
        ]
        logger.debug(f"NewsAPI → {len(headlines)} headlines for {ticker}")
        return headlines
    except Exception as e:
        logger.warning(f"NewsAPI error ({ticker}): {e}")
        return []

# ─────────────────────────────────────────────────────────────
# SOURCE 2: FINNHUB COMPANY NEWS
# Uses ticker directly — no keyword guessing needed.
# Returns last 24h of news articles about that exact company.
# ─────────────────────────────────────────────────────────────
FINNHUB_SYMBOL_MAP = {
    'BTCUSD': 'BINANCE:BTCUSDT',
    'ETHUSD': 'BINANCE:ETHUSDT',
}
def _to_finnhub_symbol(ticker: str) -> str:
    return FINNHUB_SYMBOL_MAP.get(ticker, ticker)

def fetch_finnhub_news(ticker: str) -> list[FinnhubArticle]:
    """
    Fetch recent company-specific news from Finnhub.
    Finnhub already filters to articles about the exact ticker —
    much cleaner signal than keyword-searching general news.
    """
    today     = datetime.now().strftime('%Y-%m-%d')
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    try:
        articles = _finnhub.company_news(_to_finnhub_symbol(ticker), _from=yesterday, to=today)
        result = []
        for a in articles[:config.FINNHUB_ARTICLES]:  # cap to avoid rate limits
            headline = a.get('headline', '').strip()
            if not headline:
                continue
            result.append(FinnhubArticle(
                headline       = headline,
                summary        = a.get('summary', ''),
                finnhub_score  = None,   # company_news doesn't include score
                source         = a.get('source', 'finnhub')
            ))

        logger.debug(f"Finnhub → {len(result)} articles for {ticker}")
        return result

    except Exception as e:
        logger.warning(f"Finnhub error ({ticker}): {e}")
        return []

# ─────────────────────────────────────────────────────────────
# SOURCE 3: FINNHUB PRE-SCORED SENTIMENT (bonus — 1 call/ticker)
# This endpoint returns a single aggregated sentiment score
# for the ticker based on recent news. Use as a signal check,
# not as text for FinBERT.
# ─────────────────────────────────────────────────────────────
def fetch_finnhub_sentiment_score(ticker: str) -> Optional[float]:
    """
    Get Finnhub's own pre-computed sentiment score for a ticker.
    Returns a float (positive = bullish, negative = bearish) or None.
    This is a bonus cross-check alongside FinBERT's output.
    """
    try:
        data  = _finnhub.news_sentiment(ticker)
        score = data.get('companyNewsScore')   # float, typically -1 to +1
        if score is not None:
            logger.debug(f"Finnhub sentiment score for {ticker}: {score:+.3f}")
        return score
    except Exception as e:
        logger.warning(f"Finnhub sentiment score error ({ticker}): {e}")
        return None

# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT — drop-in compatible with signal_generator.py
# ─────────────────────────────────────────────────────────────
def fetch_all_text(ticker: str) -> list[str]:
    """
    Returns all text strings for FinBERT scoring.
    Fully compatible with the existing signal_generator.py — no changes needed there.
    """
    news_texts    = fetch_news(ticker)
    finnhub_arts  = fetch_finnhub_news(ticker)

    # Combine headline + summary for richer FinBERT input
    finnhub_texts = []
    for a in finnhub_arts:
        # Use summary if it's meaningfully longer than the headline
        text = a.summary if len(a.summary) > len(a.headline) + 20 else a.headline
        finnhub_texts.append(text)

    all_texts = news_texts + finnhub_texts
    logger.info(
        f"{ticker}: {len(all_texts)} texts "
        f"({len(news_texts)} NewsAPI + {len(finnhub_texts)} Finnhub)"
    )
    return all_texts


def fetch_with_sentiment_check(ticker: str) -> tuple[list[str], Optional[float]]:
    """
    Enhanced version — returns texts for FinBERT AND Finnhub's
    pre-computed score as a secondary signal check.

    In signal_generator.py you can use the finnhub_score to
    cross-validate: if FinBERT says BUY but finnhub_score is
    strongly negative, treat it as a HOLD.
    """
    texts          = fetch_all_text(ticker)
    finnhub_score  = fetch_finnhub_sentiment_score(ticker)
    return texts, finnhub_score