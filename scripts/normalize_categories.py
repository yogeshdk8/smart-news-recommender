from app import create_app
from app.models import Article
from app import db


CATEGORY_MAP = {
    "business": "Business",
    "general": "General",
    "health": "Health",
    "science": "Science",
    "sports": "Sports",
    "technology": "Technology",
}


def main():
    app = create_app()

    with app.app_context():
        changed = 0

        for article in Article.query.all():
            old_category = article.category

            if old_category in CATEGORY_MAP:
                article.category = CATEGORY_MAP[old_category]
                changed += 1

        db.session.commit()

        print(f"Normalized {changed} article categories.")


if __name__ == "__main__":
    main()