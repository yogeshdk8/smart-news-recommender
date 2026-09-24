import os
import json
import time
from datetime import datetime, timezone

import requests
import feedparser
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

if not NEWSAPI_KEY:
    raise ValueError(
        "NEWSAPI_KEY not found. Make sure your .env file exists "
        "in the project root and contains NEWSAPI_KEY=your_key"
    )


OUTPUT_FILE = "data/raw/articles.json"

# NewsAPI categories
CATEGORIES = [
    "technology",
    "business",
    "science",
    "health",
    "sports",
]


# RSS feeds
RSS_FEEDS = {
    "BBC": "https://feeds.bbci.co.uk/news/rss.xml",
}


def collect_from_newsapi():
    """Collect news articles from NewsAPI."""

    articles = []

    url = "https://newsapi.org/v2/top-headlines"

    for category in CATEGORIES:

        print(f"Collecting {category} news...")

        params = {
            "apiKey": NEWSAPI_KEY,
            "category": category,
            "language": "en",
            "pageSize": 20,
        }

        try:
            response = requests.get(
                url,
                params=params,
                timeout=15
            )

            response.raise_for_status()

            data = response.json()

            for article in data.get("articles", []):

                articles.append({
                    "title": article.get("title"),
                    "description": article.get("description"),
                    "content": article.get("content"),
                    "url": article.get("url"),
                    "source": (
                        article.get("source", {}).get("name")
                    ),
                    "category": category,
                    "origin": "newsapi",
                    "published_at": article.get("publishedAt"),
                    "collected_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                })

        except requests.exceptions.RequestException as e:
            print(f"NewsAPI error for {category}: {e}")

        # Small delay between API requests
        time.sleep(1)

    return articles


def collect_from_rss():
    """Collect news articles from RSS feeds."""

    articles = []

    for feed_name, feed_url in RSS_FEEDS.items():

        print(f"Collecting RSS feed: {feed_name}...")

        try:
            feed = feedparser.parse(feed_url)

            for entry in feed.entries[:30]:

                articles.append({
                    "title": entry.get("title"),
                    "description": entry.get("summary"),
                    "content": entry.get("summary"),
                    "url": entry.get("link"),
                    "source": feed_name,
                    "category": "general",
                    "origin": "rss",
                    "published_at": entry.get(
                        "published"
                    ),
                    "collected_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                })

        except Exception as e:
            print(f"RSS error for {feed_name}: {e}")

    return articles


def save_articles(articles):
    """Save collected articles to JSON."""

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            articles,
            file,
            indent=4,
            ensure_ascii=False
        )

    print()
    print(
        f"Saved {len(articles)} total articles "
        f"to {OUTPUT_FILE}"
    )


def main():

    print("=" * 50)
    print("SMART NEWS - DATA COLLECTION")
    print("=" * 50)

    newsapi_articles = collect_from_newsapi()

    rss_articles = collect_from_rss()

    all_articles = newsapi_articles + rss_articles

    save_articles(all_articles)

    print()
    print("Data collection completed successfully.")


if __name__ == "__main__":
    main()