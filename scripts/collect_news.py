"""
scripts/collect_news.py

Fetches current news headlines from GNews and stores new articles
in the application's Article database.

The API key is read from the GNEWS_API_KEY environment variable.
Never put the real API key directly into this file.
"""

import os

import requests
from dotenv import load_dotenv

from app import create_app, db
from app.models import Article


# Load variables from .env when running locally.
load_dotenv()

API_KEY = os.environ.get("GNEWS_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GNEWS_API_KEY is not set. "
        "Add your GNews API key to the environment or .env file."
    )


GNEWS_URL = "https://gnews.io/api/v4/top-headlines"


# Map GNews categories to the categories already used
# by the Smart News Recommender project.
CATEGORY_MAP = {
    "general": "General",
    "world": "World",
    "nation": "India",
    "business": "Business",
    "technology": "Technology",
    "entertainment": "Entertainment",
    "sports": "Sports",
    "science": "Science",
    "health": "Health",
}


def fetch_news(
    category="general",
    country="in",
    language="en",
    max_articles=10,
):
    """Fetch top headlines from GNews."""

    params = {
        "category": category,
        "country": country,
        "lang": language,
        "max": max_articles,
    }

    headers = {
        "X-Api-Key": API_KEY,
    }

    response = requests.get(
        GNEWS_URL,
        params=params,
        headers=headers,
        timeout=15,
    )

    if response.status_code != 200:
        print(f"GNews API error: HTTP {response.status_code}")
        print(f"GNews response: {response.text}")
        response.raise_for_status()

    return response.json()


def save_articles(articles, project_category):
    """
    Save new GNews articles to the database.

    Articles are considered duplicates when their URL already
    exists in the Article table.
    """

    added = 0
    skipped = 0

    for article_data in articles:
        title = (
            article_data.get("title")
            or "Untitled"
        )

        description = (
            article_data.get("description")
            or article_data.get("content")
            or ""
        )

        source_data = article_data.get("source") or {}

        source = source_data.get("name") or "Unknown"

        url = article_data.get("url")

        published_at = article_data.get("publishedAt")

        # We need a URL for reliable duplicate prevention.
        if not url:
            print(f"Skipping article without URL: {title}")
            skipped += 1
            continue

        # Check whether this URL is already in our database.
        existing_article = Article.query.filter_by(
            url=url
        ).first()

        if existing_article:
            skipped += 1
            continue

        article = Article(
            title=title[:300],
            content=description,
            category=project_category,
            source=source[:100],
            url=url[:500],
            published_at=published_at,
        )

        db.session.add(article)
        added += 1

    db.session.commit()

    return added, skipped


def main():
    print("Connecting to GNews...")

    app = create_app()

    total_added = 0
    total_skipped = 0

    with app.app_context():

        # Fetch each category separately.
        for gnews_category, project_category in CATEGORY_MAP.items():

            print()
            print(
                f"Fetching {gnews_category} "
                f"-> {project_category}..."
            )

            data = fetch_news(
                category=gnews_category
            )

            articles = data.get("articles", [])

            print(
                f"Articles returned: {len(articles)}"
            )

            added, skipped = save_articles(
                articles,
                project_category
            )

            total_added += added
            total_skipped += skipped

            print(f"New articles added: {added}")
            print(f"Duplicates skipped: {skipped}")

        print()
        print("========================================")
        print("GNews database update complete.")
        print("========================================")
        print(f"New articles added: {total_added}")
        print(f"Duplicate articles skipped: {total_skipped}")
        print(
            f"Total articles in DB: "
            f"{Article.query.count()}"
        )


if __name__ == "__main__":
    main()