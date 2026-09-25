"""
ml/build_features_from_db.py

Bridges Phase 4 (TF-IDF) and Phase 5 (database). The vectorizer/matrix
built in Phase 4 was indexed by row position in a JSON file, but the
database assigns its own article IDs. This script rebuilds the TF-IDF
matrix directly from the Article table so every row is keyed by the
real article_id used in the Interaction table.

Run from the project root:
    python -m ml.build_features_from_db
"""

import os
import json
import sys

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from preprocessing import clean_text  # reuse your Phase 3 cleaner

from app import create_app
from app.models import Article

OUTPUT_DIR = os.path.join("data", "processed")
VECTORIZER_PATH = os.path.join(OUTPUT_DIR, "db_tfidf_vectorizer.joblib")
MATRIX_PATH = os.path.join(OUTPUT_DIR, "db_tfidf_matrix.joblib")
ARTICLE_ID_MAP_PATH = os.path.join(OUTPUT_DIR, "article_id_map.json")


def main():
    app = create_app()
    with app.app_context():
        articles = Article.query.order_by(Article.id.asc()).all()

        if not articles:
            raise RuntimeError(
                "No articles in the database. Run scripts/seed_articles.py first (Phase 5)."
            )

        texts = []
        article_ids = []  # row i of the matrix <-> article_ids[i]
        for a in articles:
            combined = f"{a.title or ''} {a.content or ''}"
            texts.append(clean_text(combined))
            article_ids.append(a.id)

    vectorizer = TfidfVectorizer(max_features=5000, min_df=1)
    matrix = vectorizer.fit_transform(texts)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(matrix, MATRIX_PATH)
    with open(ARTICLE_ID_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(article_ids, f)  # list where index i -> article_id

    print(f"Built TF-IDF matrix: {matrix.shape[0]} articles x {matrix.shape[1]} terms")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")
    print(f"Saved matrix to {MATRIX_PATH}")
    print(f"Saved article_id map to {ARTICLE_ID_MAP_PATH}")


if __name__ == "__main__":
    main()