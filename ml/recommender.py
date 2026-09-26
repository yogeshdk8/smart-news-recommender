"""
ml/recommender.py

Phase 6 — Recommendation Engine

Stage A: content_based_recommend()  — profile vector + cosine similarity
Stage B: ncf_recommend() + hybrid_recommend() — added after training the
         neural model with ml/train_ncf.py

Cold start: if a user has too little history, fall back to popularity.
"""

import os
import json

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app import create_app, db
from app.models import Interaction, Article

DATA_DIR = os.path.join("data", "processed")
VECTORIZER_PATH = os.path.join(DATA_DIR, "db_tfidf_vectorizer.joblib")
MATRIX_PATH = os.path.join(DATA_DIR, "db_tfidf_matrix.joblib")
ARTICLE_ID_MAP_PATH = os.path.join(DATA_DIR, "article_id_map.json")

# How much each interaction type counts toward a user's profile / as a
# positive training signal. Saves/likes are a stronger signal than a view.
INTERACTION_WEIGHTS = {"view": 1.0, "like": 2.0, "save": 3.0}

# Below this many interactions, we don't trust a personal profile yet.
COLD_START_THRESHOLD = 3


# ---------------------------------------------------------------------
# Load Stage A artifacts once
# ---------------------------------------------------------------------

def load_content_artifacts():
    vectorizer = joblib.load(VECTORIZER_PATH)
    matrix = joblib.load(MATRIX_PATH)
    with open(ARTICLE_ID_MAP_PATH, "r", encoding="utf-8") as f:
        article_ids = json.load(f)  # row i -> article_ids[i]

    id_to_row = {article_id: i for i, article_id in enumerate(article_ids)}
    return vectorizer, matrix, article_ids, id_to_row


# ---------------------------------------------------------------------
# Popularity fallback (cold start)
# ---------------------------------------------------------------------

def popularity_ranking(exclude_ids=None, top_n=10):
    """
    Rank articles by total interaction count across ALL users.
    Used when a user has no/too little history, or as a final tie-breaker.
    """
    exclude_ids = exclude_ids or set()

    counts = (
        db.session.query(Interaction.article_id, db.func.count(Interaction.id))
        .group_by(Interaction.article_id)
        .all()
    )
    count_map = {article_id: count for article_id, count in counts}

    all_articles = Article.query.all()
    scored = []
    for a in all_articles:
        if a.id in exclude_ids:
            continue
        # Articles with zero interactions still get a score of 0, so brand
        # new articles aren't hidden forever — they just rank at the bottom.
        scored.append((a.id, float(count_map.get(a.id, 0))))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


# ---------------------------------------------------------------------
# Stage A: Content-based filtering
# ---------------------------------------------------------------------

def get_user_interactions(user_id):
    """Returns list of (article_id, interaction_type) for a user."""
    rows = Interaction.query.filter_by(user_id=user_id).all()
    return [(r.article_id, r.type) for r in rows]


def build_user_profile_vector(user_id, matrix, id_to_row):
    """
    Weighted average of TF-IDF rows for articles the user interacted with.
    Returns None if the user has no usable history.
    """
    interactions = get_user_interactions(user_id)
    if not interactions:
        return None, set()

    rows = []
    weights = []
    interacted_ids = set()

    for article_id, itype in interactions:
        interacted_ids.add(article_id)
        row_idx = id_to_row.get(article_id)
        if row_idx is None:
            continue  # article not in the feature matrix (e.g. deleted)
        rows.append(matrix[row_idx].toarray()[0])
        weights.append(INTERACTION_WEIGHTS.get(itype, 1.0))

    if not rows:
        return None, interacted_ids

    rows = np.array(rows)
    weights = np.array(weights).reshape(-1, 1)
    profile = (rows * weights).sum(axis=0) / weights.sum()
    return profile, interacted_ids


def content_based_recommend(user_id, top_n=10):
    """
    Returns [(article_id, score), ...] ranked by cosine similarity to the
    user's profile vector. Falls back to popularity if the user is cold.
    """
    vectorizer, matrix, article_ids, id_to_row = load_content_artifacts()

    interactions = get_user_interactions(user_id)
    if len(interactions) < COLD_START_THRESHOLD:
        interacted_ids = {a_id for a_id, _ in interactions}
        return popularity_ranking(exclude_ids=interacted_ids, top_n=top_n)

    profile, interacted_ids = build_user_profile_vector(user_id, matrix, id_to_row)
    if profile is None:
        return popularity_ranking(exclude_ids=interacted_ids, top_n=top_n)

    profile = profile.reshape(1, -1)
    sims = cosine_similarity(profile, matrix).flatten()  # shape (n_articles,)

    scored = [
        (article_ids[i], float(sims[i]))
        for i in range(len(article_ids))
        if article_ids[i] not in interacted_ids
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def recommend_from_interactions(interactions, exclude_ids=None, top_n=10):
    """
    Lower-level version of content_based_recommend: builds a profile from
    an explicit list of (article_id, type) pairs instead of querying the
    live database. Used by ml/evaluate.py to test the model on held-out
    data (Phase 7's precision@K), and reusable anywhere you want to
    simulate "what would this user's recommendations look like."
    """
    vectorizer, matrix, article_ids, id_to_row = load_content_artifacts()
    exclude_ids = exclude_ids or set()

    rows, weights = [], []
    for article_id, itype in interactions:
        row_idx = id_to_row.get(article_id)
        if row_idx is None:
            continue
        rows.append(matrix[row_idx].toarray()[0])
        weights.append(INTERACTION_WEIGHTS.get(itype, 1.0))

    if not rows:
        return []

    rows = np.array(rows)
    weights = np.array(weights).reshape(-1, 1)
    profile = (rows * weights).sum(axis=0) / weights.sum()

    sims = cosine_similarity(profile.reshape(1, -1), matrix).flatten()
    scored = [
        (article_ids[i], float(sims[i]))
        for i in range(len(article_ids))
        if article_ids[i] not in exclude_ids
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n] if top_n else scored


# ---------------------------------------------------------------------
# Stage B: Neural Collaborative Filtering (requires ml/train_ncf.py to
# have been run at least once)
# ---------------------------------------------------------------------

NCF_MODEL_PATH = os.path.join(DATA_DIR, "ncf_model.keras")
NCF_USER_ID_MAP_PATH = os.path.join(DATA_DIR, "ncf_user_id_map.json")
NCF_ARTICLE_ID_MAP_PATH = os.path.join(DATA_DIR, "ncf_article_id_map.json")

_ncf_model_cache = None  # loaded lazily so Stage A works without TensorFlow installed


def ncf_available():
    return (
        os.path.exists(NCF_MODEL_PATH)
        and os.path.exists(NCF_USER_ID_MAP_PATH)
        and os.path.exists(NCF_ARTICLE_ID_MAP_PATH)
    )


def load_ncf_artifacts():
    global _ncf_model_cache
    import tensorflow as tf  # imported here so Stage A never requires TensorFlow

    if _ncf_model_cache is None:
        _ncf_model_cache = tf.keras.models.load_model(NCF_MODEL_PATH)

    with open(NCF_USER_ID_MAP_PATH, "r", encoding="utf-8") as f:
        user_id_map = {int(k): v for k, v in json.load(f).items()}
    with open(NCF_ARTICLE_ID_MAP_PATH, "r", encoding="utf-8") as f:
        article_id_map = {int(k): v for k, v in json.load(f).items()}

    return _ncf_model_cache, user_id_map, article_id_map


def ncf_scores_for_user(user_id, candidate_article_ids):
    """
    Returns {article_id: predicted_score} for the given candidates,
    using the trained NCF model. Returns {} if the user or model is
    unknown (e.g. brand-new user never seen during training).
    """
    model, user_id_map, article_id_map = load_ncf_artifacts()

    if user_id not in user_id_map:
        return {}

    known_candidates = [a for a in candidate_article_ids if a in article_id_map]
    if not known_candidates:
        return {}

    user_idx = user_id_map[user_id]
    users_arr = np.array([user_idx] * len(known_candidates))
    items_arr = np.array([article_id_map[a] for a in known_candidates])

    preds = model.predict([users_arr, items_arr], verbose=0).flatten()
    return {aid: float(score) for aid, score in zip(known_candidates, preds)}


# ---------------------------------------------------------------------
# Hybrid: combine content-based + NCF scores
# ---------------------------------------------------------------------

def hybrid_recommend(user_id, top_n=10, alpha=0.5):
    """
    Combines Stage A (content) and Stage B (NCF) scores:
        final_score = alpha * content_score + (1 - alpha) * ncf_score

    alpha=1.0 -> pure content-based, alpha=0.0 -> pure NCF.
    Falls back to content-based-only (which itself falls back to
    popularity for cold-start users) if the NCF model isn't trained yet
    or has never seen this user.
    """
    content_results = content_based_recommend(user_id, top_n=top_n * 3)  # wider pool to re-rank

    if not ncf_available():
        return content_results[:top_n]

    candidate_ids = [aid for aid, _ in content_results]
    ncf_scores = ncf_scores_for_user(user_id, candidate_ids)

    if not ncf_scores:
        # Unknown to the NCF model (e.g. brand-new user) -> content-only
        return content_results[:top_n]

    combined = []
    for article_id, content_score in content_results:
        ncf_score = ncf_scores.get(article_id, 0.0)
        final_score = alpha * content_score + (1 - alpha) * ncf_score
        combined.append((article_id, final_score))

    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:top_n]


# ---------------------------------------------------------------------
# Manual smoke test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    app = create_app()
    with app.app_context():
        user_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1

        print(f"--- Content-based only (Stage A) ---")
        for article_id, score in content_based_recommend(user_id, top_n=10):
            article = Article.query.get(article_id)
            title = article.title if article else "(unknown)"
            print(f"  [{score:.4f}] {title}")

        print(f"\n--- Hybrid (Stage A + B) ---")
        for article_id, score in hybrid_recommend(user_id, top_n=10):
            article = Article.query.get(article_id)
            title = article.title if article else "(unknown)"
            print(f"  [{score:.4f}] {title}")