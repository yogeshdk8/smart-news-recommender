"""
ml/evaluate.py

Offline evaluation of the content-based recommender.

Metrics:
    Hit Rate@5
    Hit Rate@10
    Hit Rate@20

Method:
    Leave-one-article-out evaluation.

For each user:

1. Sort interactions chronologically.
2. Find the user's most recently interacted unique article.
3. Hold out that article completely.
4. Remove all interactions with that article from training.
5. Generate recommendations from the remaining interactions.
6. Check whether the held-out article appears in the top K.

The evaluator reports several K values because Hit Rate@5 can be
very strict when the article catalogue is large and user history
is still small.
"""

from collections import defaultdict

from app import create_app
from app.models import Interaction
from ml.recommender import recommend_from_interactions


def evaluate_hit_rate(k=5, min_interactions=2):
    """
    Calculate leave-one-article-out Hit Rate@K.

    Returns:
        score, evaluated_users
    """

    app = create_app()

    with app.app_context():
        interactions = (
            Interaction.query
            .order_by(
                Interaction.user_id.asc(),
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

        # Preserve chronological order while removing duplicate articles.
        unique_articles = []
        seen_articles = set()

        for article_id, interaction_type in user_interactions:
            if article_id not in seen_articles:
                seen_articles.add(article_id)
                unique_articles.append(article_id)

        if len(unique_articles) < min_interactions:
            continue

        # Hold out the most recently interacted unique article.
        held_out_article_id = unique_articles[-1]

        # Remove ALL interactions with the held-out article.
        train = [
            (article_id, interaction_type)
            for article_id, interaction_type in user_interactions
            if article_id != held_out_article_id
        ]

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

    return hits / evaluated_users, evaluated_users


def evaluate_all():
    """
    Calculate Hit Rate@5, Hit Rate@10 and Hit Rate@20.
    """

    results = {}

    for k in (5, 10, 20):
        score, users = evaluate_hit_rate(k=k)

        results[k] = {
            "score": score,
            "users": users,
        }

    return results


def main():

    results = evaluate_all()

    print()
    print("=" * 50)
    print("Recommendation System Evaluation")
    print("=" * 50)

    for k in (5, 10, 20):

        score = results[k]["score"]
        users = results[k]["users"]

        if score is None:
            print(
                f"Hit Rate@{k}: Not enough data "
                f"(evaluated on {users} users)"
            )
        else:
            print(
                f"Hit Rate@{k}: {score * 100:.1f}% "
                f"(evaluated on {users} users)"
            )

    print("=" * 50)


if __name__ == "__main__":
    main()