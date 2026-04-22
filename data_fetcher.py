# data_fetcher.py — pulls text from NewsAPI and Reddit
# ─────────────────────────────────────────────────────────────
import praw
from newsapi import NewsApiClient
from loguru import logger
import config

# ── Initialise API clients once at import ─────────────────────
_newsapi = NewsApiClient(api_key=config.NEWS_API_KEY)

_reddit  = praw.Reddit(
    client_id     = config.REDDIT_CLIENT_ID,
    client_secret = config.REDDIT_CLIENT_SECRET,
    user_agent    = config.REDDIT_USER_AGENT,
    check_for_updates =
False
)

# ── Keyword map: ticker → search terms ───────────────────────
# Add your own tickers + search terms here
TICKER_KEYWORDS = {
    'TSLA': ['TSLA', 'Tesla', 'Elon Musk Tesla'],
    'NVDA': ['NVDA', 'Nvidia', 'NVIDIA stock'],
    'AAPL': ['AAPL', 'Apple stock', 'Apple Inc'],
    'BTC':  ['Bitcoin', 'BTC', 'crypto market'],
    'ETH':  ['Ethereum', 'ETH', 'DeFi'],
}

def _get_keywords(ticker: str) -> list[str]:
    return TICKER_KEYWORDS.get(ticker, [ticker])

def fetch_news(ticker: str) -> list[str]:
    """Fetch recent news headlines for a ticker from NewsAPI."""
    keywords = _get_keywords(ticker)
    query    = ' OR '.join(keywords)
    headlines = []
    try:
        resp = _newsapi.get_everything(
            q          = query,
            language   = 'en',
            sort_by    = 'publishedAt',
            page_size  = config.NEWS_ARTICLES
        )
        headlines = [
            a['title'] for a in resp.get('articles', [])
            if a.get('title') and a['title'] != '[Removed]'
        ]
        logger.debug(f"NewsAPI: {len(headlines)} headlines for {ticker}")
    except Exception as e:
        logger.warning(f"NewsAPI error for {ticker}: {e}")
    return headlines

def fetch_reddit(ticker: str) -> list[str]:
    """Fetch recent Reddit post titles about a ticker."""
    keywords = _get_keywords(ticker)
    subs     = 'wallstreetbets+investing+stocks+SecurityAnalysis'
    posts    = []
    try:
        for keyword in keywords[:2]:  # limit to 2 keywords to save API calls
            for post in _reddit.subreddit(subs).search(
                keyword, limit=config.REDDIT_POSTS // 2, sort='new'
            ):
                if post.title:
                    posts.append(post.title)
        logger.debug(f"Reddit: {len(posts)} posts for {ticker}")
    except Exception as e:
        logger.warning(f"Reddit error for {ticker}: {e}")
    return posts

def fetch_all_text(ticker: str) -> list[str]:
    """Get all news + Reddit text for a ticker."""
    news   = fetch_news(ticker)
    reddit = fetch_reddit(ticker)
    all_text = news + reddit
    logger.info(f"{ticker}: {len(all_text)} total texts ({len(news)} news, {len(reddit)} reddit)")
    return all_text