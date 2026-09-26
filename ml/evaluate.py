"""
ml/evaluate.py

Phase 7 — Offline evaluation of the content-based recommender.

Method: leave-one-out Hit Rate@K.

For each user with enough history:
1. Sort their interactions by timestamp.
2. Hide their most recent interaction.
3. Build recommendations from the remaining interactions.
4. Check whether the hidden article appears in the top-K recommendations.

Because there is exactly one held-out article per evaluated user,
the metric is called Hit Rate@K rather than conventional Precision@K.

Run from the project root:

    python -m ml.evaluate
"""

from collections import defaultdict

from app import create_app
from app.models import Interaction
from ml.recommender import recommend_from_interactions


def hit_rate_at_k(k=5, min_interactions=2):
    """
    Calculate leave-one-out Hit Rate@K.

    Returns:
        (score, evaluated_user_count)

    score:
        Hit Rate@K as a decimal, or None if no users had enough
        interaction history to evaluate.
    """

    app = create_app()

    with app.app_context():
        interactions = (
            Interaction.query
            .order_by(
                Interaction.timestamp.asc(),
                Interaction.id.asc(),
            )
            .all()
        )

        by_user = defaultdict(list)

        for interaction in interactions:
            by_user[interaction.user_id].append(
                (
                    interaction.article_id,
                    interaction.type,
                )
            )

    hits = 0
    evaluated_users = 0

    for user_id, user_interactions in by_user.items():

        if len(user_interactions) < min_interactions:
            continue

        # Hold out the user's most recent interaction.
        train = user_interactions[:-1]
        held_out = user_interactions[-1]

        held_out_article_id, _ = held_out

        # Do not recommend articles already used in the training history.
        train_article_ids = {
            article_id
            for article_id, _ in train
        }

        recommendations = recommend_from_interactions(
            train,
            exclude_ids=train_article_ids,
            top_n=k,
        )

        recommended_ids = {
            article_id
            for article_id, _ in recommendations
        }

        if held_out_article_id in recommended_ids:
            hits += 1

        evaluated_users += 1

    if evaluated_users == 0:
        return None, 0

    score = hits / evaluated_users

    return score, evaluated_users


def main():
    score, evaluated_users = hit_rate_at_k(k=5)

    if score is None:
        print(
            "Not enough users with 2+ interactions yet to evaluate. "
            "Interact with a few more articles as 2-3 different users, "
            "then rerun."
        )
    else:
        print(
            f"Hit Rate@5: {score:.3f} "
            f"(evaluated on {evaluated_users} users)"
        )


if __name__ == "__main__":
    main()