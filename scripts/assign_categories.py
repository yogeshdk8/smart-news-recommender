"""
Assign more specific categories to the existing news articles.

Categories:
    Technology
    Sports
    Politics
    Business
    Science
    Health
    Entertainment
    World
    India
    Education
    Travel
    Lifestyle
"""

from app import create_app, db
from app.models import Article


CATEGORY_KEYWORDS = {
    "Technology": [
        "AI",
        "OpenAI",
        "Anthropic",
        "Apple",
        "Google",
        "Android",
        "iPhone",
        "Microsoft",
        "Xbox",
        "PC",
        "laptop",
        "GPU",
        "NVIDIA",
        "Qualcomm",
        "Snapdragon",
        "Adobe",
        "Minecraft",
        "Pokémon",
        "PlayStation",
        "PS5",
        "Game Pass",
        "GTA",
        "Halo",
        "cyber",
        "satellite",
        "hackers",
        "data",
        "software",
    ],

    "Sports": [
        "NFL",
        "NBA",
        "MLB",
        "NHL",
        "UFC",
        "football",
        "soccer",
        "cricket",
        "Mets",
        "Chiefs",
        "Colts",
        "Giants",
        "Vikings",
        "Eagles",
        "Cubs",
        "Tigers",
        "Rangers",
        "49ers",
        "NASCAR",
        "NHL",
        "marathon",
        "F1",
        "Formula 1",
        "Ronaldo",
        "Eriksen",
        "Verlander",
        "Sanders",
        "Bregman",
        "Bettman",
        "Biles",
        "Pederson",
    ],

    "Politics": [
        "Trump",
        "president",
        "government",
        "senate",
        "Senate",
        "Congress",
        "congress",
        "minister",
        "minister",
        "Tories",
        "Home secretary",
        "regulators",
        "policy",
        "election",
        "political",
        "Gaza",
        "China",
        "Xi",
        "UN",
        "migrant",
        "military",
        "Navy",
        "FBI",
    ],

    "Business": [
        "stock",
        "stocks",
        "market",
        "Nasdaq",
        "Fed",
        "inflation",
        "earnings",
        "investor",
        "settlement",
        "sale",
        "company",
        "business",
        "Walmart",
        "DoorDash",
        "Six Flags",
        "oil prices",
        "crypto",
        "job",
        "workers",
        "lawsuit",
    ],

    "Science": [
        "scientist",
        "scientists",
        "science",
        "NASA",
        "planet",
        "exoplanet",
        "galaxy",
        "space",
        "spacecraft",
        "Mars",
        "quantum",
        "neutrino",
        "asteroid",
        "Earth",
        "telescope",
        "pulsar",
        "gamma rays",
        "fungal",
        "amoeba",
        "Tyrannosaurus",
        "evolution",
        "research",
        "study",
    ],

    "Health": [
        "health",
        "healthcare",
        "medical",
        "doctor",
        "disease",
        "vaccine",
        "COVID",
        "flu",
        "measles",
        "cancer",
        "migraine",
        "brain",
        "virus",
        "infection",
        "antibiotic",
        "sleep",
        "GLP-1",
        "Ozempic",
        "Wegovy",
        "Mounjaro",
        "dementia",
        "surgery",
        "hospital",
        "mental health",
        "stroke",
        "bacteria",
    ],

    "Entertainment": [
        "movie",
        "film",
        "actor",
        "actress",
        "singer",
        "music",
        "album",
        "concert",
        "Netflix",
        "Disney",
        "game",
        "gaming",
        "GTA",
        "Minecraft",
        "Pokémon",
        "Sims",
        "Rayman",
        "Halo",
        "Xbox",
    ],

    "World": [
        "UK",
        "British",
        "Russia",
        "Russian",
        "China",
        "Chinese",
        "Canada",
        "Canadian",
        "Australia",
        "Australian",
        "Europe",
        "European",
        "Gaza",
        "Israel",
        "international",
        "world",
        "Ukraine",
    ],

    "India": [
        "India",
        "Indian",
        "Indians",
        "Delhi",
        "Mumbai",
        "Bengaluru",
        "Bangalore",
        "Kolkata",
        "Chennai",
        "Hyderabad",
        "Modi",
        "rupee",
    ],

    "Education": [
        "school",
        "schools",
        "student",
        "students",
        "teacher",
        "teachers",
        "education",
        "GCSE",
        "university",
        "college",
        "vocational",
        "classroom",
    ],

    "Travel": [
        "travel",
        "tourism",
        "tourist",
        "airport",
        "flight",
        "hotel",
        "holiday",
        "vacation",
        "destination",
    ],

    "Lifestyle": [
        "lifestyle",
        "food",
        "coffee",
        "shopping",
        "fashion",
        "restaurant",
        "home",
        "garden",
        "fitness",
        "exercise",
        "sleep",
        "protein",
        "alcohol",
    ],
}


def classify_article(article):
    """
    Assign a category using keyword matches in the title.

    Existing specific categories are preserved unless the title
    clearly matches a more specific category.
    """

    title = (article.title or "").lower()

    scores = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0

        for keyword in keywords:
            if keyword.lower() in title:
                score += 1

        if score > 0:
            scores[category] = score

    if not scores:
        return article.category or "World"

    return max(scores, key=scores.get)


def main():
    app = create_app()

    with app.app_context():
        articles = Article.query.order_by(Article.id).all()

        changes = []

        for article in articles:
            old_category = article.category
            new_category = classify_article(article)

            if old_category != new_category:
                changes.append(
                    (
                        article.id,
                        old_category,
                        new_category,
                        article.title,
                    )
                )

        print("\nProposed category changes:\n")

        for article_id, old, new, title in changes:
            print(
                f"{article_id}: "
                f"{old} -> {new} | {title}"
            )

        print(f"\nTotal proposed changes: {len(changes)}")

        answer = input(
            "\nApply these changes to the database? "
            "Type YES to continue: "
        )

        if answer != "YES":
            print("No changes made.")
            return

        for article_id, _, new_category, _ in changes:
            article = db.session.get(Article, article_id)
            article.category = new_category

        db.session.commit()

        print("\nCategory assignment completed successfully.")


if __name__ == "__main__":
    main()