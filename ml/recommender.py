"""
ml/recommender.py

Strict per-user personalized recommendation engine.

IMPORTANT DESIGN RULE
---------------------

Production recommendations must depend ONLY on the currently logged-in
user's own interaction history.

Allowed user signals:
    - views
    - likes
    - saves

NOT used for production recommendations:
    - other users' interactions
    - global popularity
    - collaborative filtering
    - NCF predictions trained across users
    - shared user profiles

The article catalogue itself is shared. The user's interest profile is
built exclusively from that user's own interactions.
"""

import json
import os
import sys

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import func

from app import create_app, db
from app.models import Article, Interaction


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

DATA_DIR = os.path.join(
    "data",
    "processed",
)

VECTORIZER_PATH = os.path.join(
    DATA_DIR,
    "db_tfidf_vectorizer.joblib",
)

MATRIX_PATH = os.path.join(
    DATA_DIR,
    "db_tfidf_matrix.joblib",
)

ARTICLE_ID_MAP_PATH = os.path.join(
    DATA_DIR,
    "article_id_map.json",
)


# ---------------------------------------------------------------------
# USER INTERACTION WEIGHTS
# ---------------------------------------------------------------------
#
# Stronger actions receive larger weights.
#
# View  -> weak interest signal
# Like  -> stronger interest signal
# Save  -> strongest interest signal
#
# These weights are applied ONLY to the current user's interactions.
# ---------------------------------------------------------------------

INTERACTION_WEIGHTS = {
    "view": 1.0,
    "like": 2.0,
    "save": 3.0,
}


# ---------------------------------------------------------------------
# COMPATIBILITY CONSTANT
# ---------------------------------------------------------------------
#
# Older recommender tests import this constant.
#
# IMPORTANT:
# personalized_recommend() DOES NOT use this value to fall back to
# popularity.
#
# Strict per-user personalization must remain strict.
# ---------------------------------------------------------------------

COLD_START_THRESHOLD = 3


# ---------------------------------------------------------------------
# LOAD CONTENT ARTIFACTS
# ---------------------------------------------------------------------

def load_content_artifacts():
    """
    Load the TF-IDF vectorizer, article matrix, and article ID mapping.

    These artifacts describe article content.

    They do NOT contain another user's personal recommendation profile.

    Returns
    -------
    tuple
        (
            vectorizer,
            matrix,
            article_ids,
            id_to_row
        )
    """

    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            f"TF-IDF vectorizer not found: {VECTORIZER_PATH}"
        )

    if not os.path.exists(MATRIX_PATH):
        raise FileNotFoundError(
            f"TF-IDF matrix not found: {MATRIX_PATH}"
        )

    if not os.path.exists(ARTICLE_ID_MAP_PATH):
        raise FileNotFoundError(
            f"Article ID map not found: {ARTICLE_ID_MAP_PATH}"
        )

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    matrix = joblib.load(
        MATRIX_PATH
    )

    with open(
        ARTICLE_ID_MAP_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        article_ids = json.load(f)

    id_to_row = {
        article_id: index
        for index, article_id in enumerate(article_ids)
    }

    return (
        vectorizer,
        matrix,
        article_ids,
        id_to_row,
    )


# ---------------------------------------------------------------------
# POPULARITY RANKING
# ---------------------------------------------------------------------
#
# COMPATIBILITY / TEST HELPER ONLY
#
# IMPORTANT:
# This function is NOT used by personalized_recommend().
#
# It exists because older tests and older parts of the project expect
# popularity_ranking() to exist.
#
# Production /feed recommendations remain strictly personalized.
# ---------------------------------------------------------------------

def popularity_ranking(
    exclude_ids=None,
    top_n=10,
):
    """
    Return articles ordered by total interaction count.

    This function is retained for backward compatibility with the
    existing test suite.

    IMPORTANT:
        This function must NOT be used by personalized_recommend().

    Parameters
    ----------
    exclude_ids : iterable, optional
        Article IDs that should not appear in the result.

    top_n : int, default=10
        Maximum number of articles to return.

    Returns
    -------
    list
        A list of tuples:

            [
                (article_id, score),
                ...
            ]

        where score is the total number of interactions.

    NOTE
    ----
    This function can query interactions from all users because it is
    only a compatibility/test helper.

    Production personalization NEVER calls this function.
    """

    exclude_ids = set(
        exclude_ids or []
    )

    if top_n <= 0:
        return []

    rows = (
        db.session.query(
            Interaction.article_id,
            func.count(
                Interaction.id
            ).label(
                "interaction_count"
            ),
        )
        .group_by(
            Interaction.article_id
        )
        .order_by(
            func.count(
                Interaction.id
            ).desc()
        )
        .all()
    )

    recommendations = []

    for article_id, interaction_count in rows:

        if article_id in exclude_ids:
            continue

        recommendations.append(
            (
                article_id,
                float(
                    interaction_count
                ),
            )
        )

        if len(recommendations) >= top_n:
            break

    return recommendations


# ---------------------------------------------------------------------
# GET ONLY ONE USER'S INTERACTIONS
# ---------------------------------------------------------------------

def get_user_interactions(user_id):
    """
    Return ONLY the interaction history belonging to user_id.

    Returns
    -------
    list
        [
            (article_id, interaction_type),
            ...
        ]

    IMPORTANT
    ---------
    No other user's interactions are queried here.
    """

    rows = (
        Interaction.query
        .filter_by(
            user_id=user_id
        )
        .order_by(
            Interaction.timestamp.asc(),
            Interaction.id.asc(),
        )
        .all()
    )

    return [
        (
            row.article_id,
            row.type,
        )
        for row in rows
    ]


# ---------------------------------------------------------------------
# BUILD PERSONAL USER PROFILE
# ---------------------------------------------------------------------

def build_user_profile_vector(
    user_id,
    matrix,
    id_to_row,
):
    """
    Build a personalized TF-IDF interest profile.

    CRITICAL:
        The profile is created ONLY from interactions belonging to
        user_id.

        Other users are never queried here.

    Returns
    -------
    tuple
        (
            profile_vector,
            interacted_article_ids
        )

    If the user has no usable interactions:

        (
            None,
            set()
        )
    """

    interactions = get_user_interactions(
        user_id
    )

    if not interactions:
        return (
            None,
            set(),
        )

    rows = []
    weights = []

    interacted_ids = set()

    for article_id, interaction_type in interactions:

        interacted_ids.add(
            article_id
        )

        row_index = id_to_row.get(
            article_id
        )

        # The interaction may refer to an article that is not present
        # in the current TF-IDF artifact.
        if row_index is None:
            continue

        article_vector = (
            matrix[row_index]
            .toarray()[0]
        )

        weight = INTERACTION_WEIGHTS.get(
            interaction_type,
            1.0,
        )

        rows.append(
            article_vector
        )

        weights.append(
            weight
        )

    if not rows:
        return (
            None,
            interacted_ids,
        )

    rows = np.asarray(
        rows,
        dtype=float,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    ).reshape(
        -1,
        1,
    )

    total_weight = weights.sum()

    if total_weight <= 0:
        return (
            None,
            interacted_ids,
        )

    # Weighted average of ONLY this user's
    # interacted article vectors.
    profile = (
        rows * weights
    ).sum(
        axis=0
    ) / total_weight

    return (
        profile,
        interacted_ids,
    )


# ---------------------------------------------------------------------
# STRICT PERSONALIZED RECOMMENDATIONS
# ---------------------------------------------------------------------

def personalized_recommend(
    user_id,
    top_n=20,
):
    """
    Generate recommendations ONLY from this user's own interactions.

    This function intentionally does NOT use:

        - popularity_ranking()
        - NCF
        - collaborative filtering
        - other users' interactions
        - global popularity
        - another user's profile

    Recommendation process:

        1. Read only user_id's interactions.
        2. Build a TF-IDF interest profile from those interactions.
        3. Compare the profile against article content.
        4. Exclude articles already interacted with by this user.
        5. Return the highest-content-similarity articles.

    For users with no interaction history, there is no personal profile.

    In that case the function returns an empty list rather than using
    another user's data or global popularity.

    Returns
    -------
    list
        [
            (article_id, similarity_score),
            ...
        ]
    """

    if top_n <= 0:
        return []

    (
        _vectorizer,
        matrix,
        article_ids,
        id_to_row,
    ) = load_content_artifacts()

    # -------------------------------------------------------------
    # STEP 1
    # Get ONLY this user's interaction history.
    # -------------------------------------------------------------

    interactions = get_user_interactions(
        user_id
    )

    if not interactions:
        return []

    # -------------------------------------------------------------
    # STEP 2
    # Build ONLY this user's profile.
    # -------------------------------------------------------------

    (
        profile,
        interacted_ids,
    ) = build_user_profile_vector(
        user_id,
        matrix,
        id_to_row,
    )

    if profile is None:
        return []

    # -------------------------------------------------------------
    # STEP 3
    # Compare this user's profile against article content.
    # -------------------------------------------------------------

    profile = profile.reshape(
        1,
        -1,
    )

    similarities = cosine_similarity(
        profile,
        matrix,
    ).flatten()

    # -------------------------------------------------------------
    # STEP 4
    # Score candidate articles.
    #
    # IMPORTANT:
    # Articles already interacted with by THIS USER are excluded.
    # -------------------------------------------------------------

    scored = []

    for index, article_id in enumerate(
        article_ids
    ):

        if article_id in interacted_ids:
            continue

        score = float(
            similarities[index]
        )

        scored.append(
            (
                article_id,
                score,
            )
        )

    # -------------------------------------------------------------
    # STEP 5
    # Highest similarity first.
    # -------------------------------------------------------------

    scored.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return scored[:top_n]


# ---------------------------------------------------------------------
# RECOMMENDATIONS FROM EXPLICIT INTERACTIONS
# ---------------------------------------------------------------------
#
# Used by the offline evaluator.
#
# This function also uses ONLY the supplied interaction list.
# ---------------------------------------------------------------------

def recommend_from_interactions(
    interactions,
    exclude_ids=None,
    top_n=10,
):
    """
    Build a recommendation profile from an explicit list of
    interactions.

    Example
    -------

        interactions = [
            (12, "view"),
            (25, "like"),
            (31, "save"),
        ]

    This function never queries other users.

    Parameters
    ----------
    interactions : iterable
        Iterable containing:

            (article_id, interaction_type)

    exclude_ids : iterable, optional
        Article IDs to exclude from recommendations.

    top_n : int, default=10
        Maximum number of recommendations.

    Returns
    -------
    list
        [
            (article_id, similarity_score),
            ...
        ]
    """

    if top_n <= 0:
        return []

    (
        _vectorizer,
        matrix,
        article_ids,
        id_to_row,
    ) = load_content_artifacts()

    exclude_ids = set(
        exclude_ids or []
    )

    rows = []
    weights = []

    for article_id, interaction_type in interactions:

        row_index = id_to_row.get(
            article_id
        )

        if row_index is None:
            continue

        rows.append(
            matrix[row_index]
            .toarray()[0]
        )

        weights.append(
            INTERACTION_WEIGHTS.get(
                interaction_type,
                1.0,
            )
        )

    if not rows:
        return []

    rows = np.asarray(
        rows,
        dtype=float,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    ).reshape(
        -1,
        1,
    )

    total_weight = weights.sum()

    if total_weight <= 0:
        return []

    profile = (
        rows * weights
    ).sum(
        axis=0
    ) / total_weight

    similarities = cosine_similarity(
        profile.reshape(
            1,
            -1,
        ),
        matrix,
    ).flatten()

    scored = []

    for index, article_id in enumerate(
        article_ids
    ):

        if article_id in exclude_ids:
            continue

        scored.append(
            (
                article_id,
                float(
                    similarities[index]
                ),
            )
        )

    scored.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return scored[:top_n]


# ---------------------------------------------------------------------
# COMPATIBILITY FUNCTION
# ---------------------------------------------------------------------
#
# The Flask application previously called hybrid_recommend().
#
# We keep the function name so existing imports/routes do not break.
#
# IMPORTANT:
# This is NO LONGER a hybrid recommender.
#
# It simply calls personalized_recommend().
#
# Therefore production recommendations are still strictly per-user.
# ---------------------------------------------------------------------

def hybrid_recommend(
    user_id,
    top_n=20,
    alpha=1.0,
):
    """
    Compatibility wrapper.

    Despite the old name, this function performs STRICT
    per-user content-based recommendation.

    Parameters
    ----------
    user_id : int
        Currently logged-in user's ID.

    top_n : int
        Number of recommendations requested.

    alpha : float
        Retained only for compatibility with older calling code.

        It has NO effect on the recommendations.

    Returns
    -------
    list
        Same result as personalized_recommend().
    """

    return personalized_recommend(
        user_id=user_id,
        top_n=top_n,
    )


# ---------------------------------------------------------------------
# MANUAL TEST
# ---------------------------------------------------------------------

if __name__ == "__main__":

    app = create_app()

    with app.app_context():

        if len(sys.argv) > 1:
            try:
                user_id = int(
                    sys.argv[1]
                )
            except ValueError:
                print(
                    "ERROR: user_id must be an integer."
                )
                sys.exit(1)
        else:
            user_id = 1

        print()
        print("=" * 70)
        print(
            "STRICT PER-USER RECOMMENDATION TEST"
        )
        print("=" * 70)

        print(
            f"User ID: {user_id}"
        )

        interactions = (
            get_user_interactions(
                user_id
            )
        )

        print(
            f"User interactions: "
            f"{len(interactions)}"
        )

        if interactions:
            print()
            print(
                "User interaction history:"
            )

            for article_id, interaction_type in interactions:
                print(
                    f"  Article {article_id}: "
                    f"{interaction_type}"
                )

        print()
        print(
            "Recommendations:"
        )

        recommendations = (
            personalized_recommend(
                user_id=user_id,
                top_n=10,
            )
        )

        if not recommendations:
            print(
                "  No personalized recommendations available."
            )
            print(
                "  The user needs at least one usable interaction."
            )

        else:
            for article_id, score in recommendations:

                # SQLAlchemy 2.x API.
                # This avoids the deprecated Article.query.get().
                article = db.session.get(
                    Article,
                    article_id,
                )

                title = (
                    article.title
                    if article
                    else "(unknown)"
                )

                print(
                    f"[{score:.4f}] "
                    f"{title}"
                )

        print("=" * 70)