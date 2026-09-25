"""
scripts/seed_articles.py

Loads data/processed/articles_cleaned.json (from Phase 3) into the
Article table so you have something to view/like/save in Phase 5.

Run from the project root with your venv active:
    python -m scripts.seed_articles
"""

import os
import json

from app import create_app, db
from app.models import Article

PROCESSED_PATH = os.path.join("data", "processed", "articles_cleaned.json")


def main():
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(
            f"Couldn't find {PROCESSED_PATH}. Run ml/run_preprocessing.py first (Phase 3)."
        )

    with open(PROCESSED_PATH, "r", encoding="utf-8") as f:
        articles = json.load(f)

    app = create_app()
    with app.app_context():
        added = 0
        for a in articles:
            # Skip duplicates if you run this script more than once
            exists = Article.query.filter_by(title=a.get("title")).first()
            if exists:
                continue

            article = Article(
                title=a.get("title") or "Untitled",
                content=a.get("description") or a.get("content") or "",
                category=a.get("category"),
                source=a.get("source"),
                url=a.get("url"),
                published_at=a.get("published_at"),
            )
            db.session.add(article)
            added += 1

        db.session.commit()
        print(f"Added {added} new articles to the database.")
        print(f"Total articles in DB: {Article.query.count()}")


if __name__ == "__main__":
    main()