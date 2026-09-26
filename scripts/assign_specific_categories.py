from app import create_app, db
from app.models import Article


CATEGORY_MAP = {
    # Technology
    111: "Technology",
    115: "Technology",

    # Politics
    95: "Politics",
    96: "Politics",
    102: "Politics",
    106: "Politics",
    108: "Politics",
    112: "Politics",
    114: "Politics",

    # Education
    98: "Education",

    # World
    94: "World",
    101: "World",
    103: "World",
    116: "World",

    # India
    110: "India",

    # Health
    93: "Health",
    109: "Health",
    117: "Health",

    # Sports
    97: "Sports",
    119: "Sports",
    120: "Sports",
    121: "Sports",
    122: "Sports",

    # Entertainment
    9: "Entertainment",
    14: "Entertainment",
    16: "Entertainment",

    # Lifestyle
    29: "Lifestyle",
    105: "Lifestyle",

    # Business
    12: "Business",
}


def main():
    app = create_app()

    with app.app_context():

        print("\nProposed category changes:\n")

        for article_id, new_category in CATEGORY_MAP.items():

            article = db.session.get(Article, article_id)

            if article is None:
                print(f"{article_id}: ARTICLE NOT FOUND")
                continue

            print(
                f"{article.id}: "
                f"{article.category} -> {new_category} | "
                f"{article.title}"
            )

        print(
            f"\nTotal proposed changes: "
            f"{len(CATEGORY_MAP)}"
        )

        answer = input(
            "\nApply these changes to the database? "
            "Type YES to continue: "
        )

        if answer != "YES":
            print("No changes made.")
            return

        for article_id, new_category in CATEGORY_MAP.items():

            article = db.session.get(Article, article_id)

            if article is not None:
                article.category = new_category

        db.session.commit()

        print(
            "\nCategory assignment completed successfully."
        )


if __name__ == "__main__":
    main()