"""
ml/feature_extraction.py

Phase 4 — Feature Extraction
Converts cleaned article text into a TF-IDF matrix, saves the fitted
vectorizer + matrix for reuse, and provides a similarity helper.
"""

import os
import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROCESSED_PATH = os.path.join("data", "processed", "articles_cleaned.json")

MODEL_DIR = os.path.join("data", "processed")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
MATRIX_PATH = os.path.join(MODEL_DIR, "tfidf_matrix.joblib")
ARTICLES_INDEX_PATH = os.path.join(MODEL_DIR, "articles_index.json")


def load_cleaned_articles():
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(
            f"Couldn't find {PROCESSED_PATH}. Run ml/run_preprocessing.py first (Phase 3)."
        )
    with open(PROCESSED_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_tfidf_matrix(articles, max_features=5000):
    """
    Fit a TfidfVectorizer on the cleaned_text of every article.

    Returns (vectorizer, matrix) where matrix is shape
    (n_articles, n_vocab_terms).
    """
    texts = [a["cleaned_text"] for a in articles]

    vectorizer = TfidfVectorizer(
        max_features=max_features,  # cap vocabulary size to keep things fast
        min_df=1,                   # keep terms that appear in at least 1 doc
        ngram_range=(1, 1),         # single words only for now (unigrams)
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def save_artifacts(vectorizer, matrix, articles):
    """Persist the vectorizer, matrix, and a lightweight index of article
    metadata (title/source/category/url) so later phases can map matrix
    rows back to real articles without reloading the full cleaned file."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(matrix, MATRIX_PATH)

    index = [
        {
            "row": i,
            "title": a.get("title"),
            "source": a.get("source"),
            "category": a.get("category"),
            "url": a.get("url"),
        }
        for i, a in enumerate(articles)
    ]
    with open(ARTICLES_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"Saved vectorizer to {VECTORIZER_PATH}")
    print(f"Saved TF-IDF matrix ({matrix.shape[0]} articles x {matrix.shape[1]} terms) to {MATRIX_PATH}")
    print(f"Saved article index to {ARTICLES_INDEX_PATH}")


def load_artifacts():
    """Load a previously saved vectorizer + matrix + index (no retraining)."""
    vectorizer = joblib.load(VECTORIZER_PATH)
    matrix = joblib.load(MATRIX_PATH)
    with open(ARTICLES_INDEX_PATH, "r", encoding="utf-8") as f:
        index = json.load(f)
    return vectorizer, matrix, index


def similarity_between(matrix, i, j):
    """Cosine similarity between article rows i and j in the TF-IDF matrix."""
    sim = cosine_similarity(matrix[i], matrix[j])
    return float(sim[0][0])


def main():
    articles = load_cleaned_articles()
    print(f"Loaded {len(articles)} cleaned articles.")

    vectorizer, matrix = build_tfidf_matrix(articles)
    save_artifacts(vectorizer, matrix, articles)

    # ------------------------------------------------------------------
    # Sanity check: find one pair of same-category articles and one pair
    # of different-category articles, then compare their similarity.
    # ------------------------------------------------------------------
    by_category = {}
    for i, a in enumerate(articles):
        by_category.setdefault(a.get("category", "unknown"), []).append(i)

    same_category_pair = None
    for cat, rows in by_category.items():
        if len(rows) >= 2:
            same_category_pair = (rows[0], rows[1], cat)
            break

    diff_category_pair = None
    cats = [c for c, rows in by_category.items() if rows]
    if len(cats) >= 2:
        i = by_category[cats[0]][0]
        j = by_category[cats[1]][0]
        diff_category_pair = (i, j, cats[0], cats[1])

    print("\n--- Sanity check ---")
    if same_category_pair:
        i, j, cat = same_category_pair
        sim = similarity_between(matrix, i, j)
        print(f"Same-category pair ({cat}): '{articles[i]['title']}' vs '{articles[j]['title']}'")
        print(f"  Similarity: {sim:.4f}")
    else:
        print("Not enough articles in a single category to test a same-category pair.")

    if diff_category_pair:
        i, j, cat_i, cat_j = diff_category_pair
        sim = similarity_between(matrix, i, j)
        print(f"Different-category pair ({cat_i} vs {cat_j}): '{articles[i]['title']}' vs '{articles[j]['title']}'")
        print(f"  Similarity: {sim:.4f}")
    else:
        print("Not enough categories to test a different-category pair.")

    if same_category_pair and diff_category_pair:
        same_sim = similarity_between(matrix, same_category_pair[0], same_category_pair[1])
        diff_sim = similarity_between(matrix, diff_category_pair[0], diff_category_pair[1])
        print(f"\nSame-category similarity ({same_sim:.4f}) {'>' if same_sim > diff_sim else '<='} different-category similarity ({diff_sim:.4f})")
        if same_sim > diff_sim:
            print("Looks correct: related articles score higher than unrelated ones.")
        else:
            print("Unexpected: unrelated articles scored as similar or more similar. "
                  "This can happen with small/noisy datasets — check the actual article pairs above.")


if __name__ == "__main__":
    main()