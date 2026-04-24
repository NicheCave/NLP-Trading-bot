import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from newsapi import NewsApiClient

load_dotenv()

def test_alpaca():
    try:
        api = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )
        acc = api.get_account()
        print(f"✅ Alpaca OK — Portfolio: ${float(acc.portfolio_value):,.2f}")
    except Exception as e:
        print(f"❌ Alpaca Failed: {e}")

def test_news():
    try:
        n = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))
        r = n.get_everything(q='Tesla', language='en', page_size=1)
        print(f"✅ NewsAPI OK — Found {r['totalResults']} articles")
    except Exception as e:
        print(f"❌ NewsAPI Failed: {e}")

if __name__ == "__main__":
    test_alpaca()
    test_news()