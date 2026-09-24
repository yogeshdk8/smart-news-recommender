import json
from pathlib import Path

from ml.preprocessing import clean_article


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "raw" / "articles.json"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "articles_cleaned.json"


def main():

    # Check if input file exists
    if not INPUT_FILE.exists():
        print(f"Input file not found: {INPUT_FILE}")
        return

    # Create output directory if it does not exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read raw articles
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        articles = json.load(file)

    print(f"Loaded {len(articles)} articles.")

    # Clean every article
    cleaned_articles = []

    for article in articles:
        cleaned_article = clean_article(article)
        cleaned_articles.append(cleaned_article)

    # Save cleaned articles
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            cleaned_articles,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(f"Cleaned {len(cleaned_articles)} articles.")
    print(f"Output saved to: {OUTPUT_FILE}")

    # Display first article as an example
    if cleaned_articles:
        example = cleaned_articles[0]

        print("\nExample cleaned article:")
        print("--------------------------------")

        print("Original title:")
        print(example.get("title", ""))

        print("\nCleaned title:")
        print(example.get("cleaned_title", ""))

        print("\nCleaned text:")
        print(example.get("cleaned_text", "")[:500])


if __name__ == "__main__":
    main()