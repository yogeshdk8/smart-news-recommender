"""
ml/evaluate.py

Offline evaluation of the personalized content-based recommender.

Metrics:
    Hit Rate@5
    Hit Rate@10
    Hit Rate@20

    MRR@5
    MRR@10
    MRR@20

    NDCG@5
    NDCG@10
    NDCG@20

    Average Rank

Method:
    Leave-one-article-out evaluation.

For each user:
1. Sort interactions chronologically.
2. Find the user's most recently interacted unique article.
3. Hold out that article completely.
4. Remove all interactions with that article from training.
5. Generate recommendations from the remaining interactions.
6. Check whether the held-out article appears in the recommendation list.
7. Calculate ranking-based metrics.

Important:
    Evaluation is strictly user-specific.

    The evaluator never uses another user's interaction history to
    construct the evaluated user's profile.

    The evaluation therefore matches the production requirement that
    recommendations depend only on the currently logged-in user's
    own interaction history.
"""

from collections import defaultdict
from math import log2

from app import create_app
from app.models import Interaction
from ml.recommender import recommend_from_interactions


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

# Recommendation cutoffs to evaluate.
EVALUATION_KS = (5, 10, 20)

# Minimum number of unique articles a user must have interacted with
# before leave-one-out evaluation is meaningful.
DEFAULT_MIN_INTERACTIONS = 2


# ---------------------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------------------

def load_user_interactions():
    """
    Load all interactions from the database and group them by user.

    Returns:
        dict:
            {
                user_id: [
                    (article_id, interaction_type),
                    ...
                ]
            }

    Interactions are ordered chronologically.

    This function creates its own Flask application context so it
    can safely be called independently.
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

    return dict(by_user)


def get_unique_article_history(user_interactions):
    """
    Preserve chronological order while removing duplicate articles.

    Example:

        [
            (101, "view"),
            (101, "like"),
            (102, "view"),
            (101, "view"),
            (103, "save"),
        ]

    becomes:

        [101, 102, 103]

    Returns:
        list[int]
    """

    unique_articles = []
    seen_articles = set()

    for article_id, _ in user_interactions:

        if article_id not in seen_articles:

            seen_articles.add(article_id)
            unique_articles.append(article_id)

    return unique_articles


# ---------------------------------------------------------------------
# RANKING METRICS
# ---------------------------------------------------------------------

def reciprocal_rank(recommended_ids, target_id, k):
    """
    Calculate Reciprocal Rank for one target article.

    If the target is ranked first:

        RR = 1

    If ranked second:

        RR = 1 / 2

    If ranked fifth:

        RR = 1 / 5

    If the target is not in the top K:

        RR = 0
    """

    for rank, article_id in enumerate(
        recommended_ids[:k],
        start=1,
    ):

        if article_id == target_id:

            return 1.0 / rank

    return 0.0


def dcg_at_k(recommended_ids, target_id, k):
    """
    Calculate DCG@K for a single target.

    Because this evaluation has one held-out target article,
    relevance is:

        1 = target found
        0 = target not found
    """

    for rank, article_id in enumerate(
        recommended_ids[:k],
        start=1,
    ):

        if article_id == target_id:

            return 1.0 / log2(rank + 1)

    return 0.0


def ndcg_at_k(recommended_ids, target_id, k):
    """
    Calculate NDCG@K.

    Because there is exactly one relevant held-out article,
    the ideal DCG is always 1.
    """

    dcg = dcg_at_k(
        recommended_ids,
        target_id,
        k,
    )

    ideal_dcg = 1.0

    return dcg / ideal_dcg


def rank_of_target(recommended_ids, target_id):
    """
    Return the 1-based rank of the target article.

    Returns:
        int | None
    """

    for rank, article_id in enumerate(
        recommended_ids,
        start=1,
    ):

        if article_id == target_id:

            return rank

    return None


# ---------------------------------------------------------------------
# SINGLE USER EVALUATION
# ---------------------------------------------------------------------

def evaluate_user(
    user_id,
    user_interactions,
    k_values=EVALUATION_KS,
    min_interactions=DEFAULT_MIN_INTERACTIONS,
):
    """
    Evaluate one user's personalized recommendations.

    IMPORTANT:

        This function must be executed while a Flask application
        context is active because recommend_from_interactions()
        may query Article from the database.

    Returns:
        dict | None

    None is returned when the user does not have enough unique
    article interactions for leave-one-out evaluation.
    """

    unique_articles = get_unique_article_history(
        user_interactions
    )

    if len(unique_articles) < min_interactions:

        return None

    # -------------------------------------------------------------
    # HOLD OUT THE MOST RECENT UNIQUE ARTICLE
    # -------------------------------------------------------------

    held_out_article_id = unique_articles[-1]

    # -------------------------------------------------------------
    # BUILD TRAINING HISTORY
    #
    # Remove every interaction with the held-out article.
    #
    # This prevents information leakage. The recommender cannot
    # use views, likes, saves, etc. for the held-out article.
    # -------------------------------------------------------------

    training_interactions = [
        (article_id, interaction_type)
        for article_id, interaction_type in user_interactions
        if article_id != held_out_article_id
    ]

    training_article_ids = {
        article_id
        for article_id, _ in training_interactions
    }

    # -------------------------------------------------------------
    # GENERATE PERSONALIZED RECOMMENDATIONS
    #
    # recommend_from_interactions() receives ONLY this user's
    # training interactions.
    #
    # It does not receive another user's interaction history.
    # -------------------------------------------------------------

    max_k = max(k_values)

    recommendations = recommend_from_interactions(
        training_interactions,
        exclude_ids=training_article_ids,
        top_n=max_k,
    )

    # The recommender returns:
    #
    #     [(article_id, score), ...]
    #
    # We only need the article IDs for evaluation.

    recommended_ids = [
        article_id
        for article_id, _ in recommendations
    ]

    # -------------------------------------------------------------
    # CALCULATE METRICS
    # -------------------------------------------------------------

    result = {
        "user_id": user_id,
        "held_out_article_id": held_out_article_id,
        "training_article_count": len(training_article_ids),
        "recommendation_count": len(recommended_ids),
        "rank": rank_of_target(
            recommended_ids,
            held_out_article_id,
        ),
        "metrics": {},
    }

    for k in k_values:

        hit = (
            held_out_article_id
            in recommended_ids[:k]
        )

        rr = reciprocal_rank(
            recommended_ids,
            held_out_article_id,
            k,
        )

        ndcg = ndcg_at_k(
            recommended_ids,
            held_out_article_id,
            k,
        )

        result["metrics"][k] = {
            "hit": 1 if hit else 0,
            "mrr": rr,
            "ndcg": ndcg,
        }

    return result


# ---------------------------------------------------------------------
# COMPLETE EVALUATION
# ---------------------------------------------------------------------

def evaluate_recommender(
    k_values=EVALUATION_KS,
    min_interactions=DEFAULT_MIN_INTERACTIONS,
):
    """
    Evaluate the recommender across all eligible users.

    IMPORTANT:

        The complete evaluation is executed inside ONE Flask
        application context.

        This is required because recommend_from_interactions()
        may access SQLAlchemy models such as Article.query.

    This fixes the error:

        RuntimeError: Working outside of application context.

    Returns:

        {
            "users": ...,
            "evaluated_users": ...,
            "skipped_users": ...,
            "results": ...,
            "metrics": ...,
            "average_rank": ...
        }
    """

    # -------------------------------------------------------------
    # CREATE ONE FLASK APPLICATION CONTEXT
    # -------------------------------------------------------------

    app = create_app()

    with app.app_context():

        # ---------------------------------------------------------
        # LOAD ALL DATABASE INTERACTIONS
        # ---------------------------------------------------------

        interactions = (
            Interaction.query
            .order_by(
                Interaction.user_id.asc(),
                Interaction.timestamp.asc(),
                Interaction.id.asc(),
            )
            .all()
        )

        # ---------------------------------------------------------
        # GROUP INTERACTIONS BY USER
        # ---------------------------------------------------------

        by_user = defaultdict(list)

        for interaction in interactions:

            by_user[interaction.user_id].append(
                (
                    interaction.article_id,
                    interaction.type,
                )
            )

        # ---------------------------------------------------------
        # EVALUATE EACH USER
        # ---------------------------------------------------------

        user_results = []

        skipped_users = 0

        for user_id, user_interactions in by_user.items():

            result = evaluate_user(
                user_id=user_id,
                user_interactions=user_interactions,
                k_values=k_values,
                min_interactions=min_interactions,
            )

            if result is None:

                skipped_users += 1
                continue

            user_results.append(result)

        evaluated_users = len(user_results)

        # ---------------------------------------------------------
        # AGGREGATE METRICS
        # ---------------------------------------------------------

        metrics = {}

        for k in k_values:

            if evaluated_users == 0:

                metrics[k] = {
                    "hit_rate": None,
                    "mrr": None,
                    "ndcg": None,
                }

                continue

            total_hits = sum(
                result["metrics"][k]["hit"]
                for result in user_results
            )

            total_mrr = sum(
                result["metrics"][k]["mrr"]
                for result in user_results
            )

            total_ndcg = sum(
                result["metrics"][k]["ndcg"]
                for result in user_results
            )

            metrics[k] = {
                "hit_rate": (
                    total_hits
                    / evaluated_users
                ),

                "mrr": (
                    total_mrr
                    / evaluated_users
                ),

                "ndcg": (
                    total_ndcg
                    / evaluated_users
                ),
            }

        # ---------------------------------------------------------
        # AVERAGE RANK
        # ---------------------------------------------------------

        ranked_users = [
            result["rank"]
            for result in user_results
            if result["rank"] is not None
        ]

        if ranked_users:

            average_rank = (
                sum(ranked_users)
                / len(ranked_users)
            )

        else:

            average_rank = None

        # ---------------------------------------------------------
        # RETURN COMPLETE EVALUATION
        # ---------------------------------------------------------

        return {
            "users": len(by_user),
            "evaluated_users": evaluated_users,
            "skipped_users": skipped_users,
            "results": user_results,
            "metrics": metrics,
            "average_rank": average_rank,
        }


# ---------------------------------------------------------------------
# BACKWARD-COMPATIBLE API
# ---------------------------------------------------------------------

def evaluate_hit_rate(
    k=5,
    min_interactions=DEFAULT_MIN_INTERACTIONS,
):
    """
    Backward-compatible Hit Rate@K function.

    Returns:
        score, evaluated_users
    """

    evaluation = evaluate_recommender(
        k_values=(k,),
        min_interactions=min_interactions,
    )

    score = evaluation["metrics"][k]["hit_rate"]

    evaluated_users = evaluation["evaluated_users"]

    return score, evaluated_users


def evaluate_all(
    min_interactions=DEFAULT_MIN_INTERACTIONS,
):
    """
    Calculate Hit Rate, MRR and NDCG for K=5, 10 and 20.

    Returns:

        {
            5: {
                "score": ...,
                "hit_rate": ...,
                "mrr": ...,
                "ndcg": ...,
                "users": ...
            },
            10: {
                ...
            },
            20: {
                ...
            }
        }
    """

    evaluation = evaluate_recommender(
        k_values=EVALUATION_KS,
        min_interactions=min_interactions,
    )

    results = {}

    for k in EVALUATION_KS:

        metrics = evaluation["metrics"][k]

        results[k] = {

            # "score" is retained for compatibility
            # with the previous evaluator.

            "score": metrics["hit_rate"],

            "hit_rate": metrics["hit_rate"],

            "mrr": metrics["mrr"],

            "ndcg": metrics["ndcg"],

            "users": evaluation["evaluated_users"],
        }

    return results


# ---------------------------------------------------------------------
# FORMATTING HELPERS
# ---------------------------------------------------------------------

def format_percentage(value):
    """
    Format a decimal metric as a percentage.
    """

    if value is None:

        return "N/A"

    return f"{value * 100:.1f}%"


def format_decimal(value):
    """
    Format a decimal metric.
    """

    if value is None:

        return "N/A"

    return f"{value:.4f}"


# ---------------------------------------------------------------------
# PRINTING
# ---------------------------------------------------------------------

def print_evaluation_report(evaluation):
    """
    Print a complete evaluation report.
    """

    print()

    print("=" * 70)

    print(
        "PERSONALIZED RECOMMENDATION SYSTEM EVALUATION"
    )

    print("=" * 70)

    print()

    print("Evaluation method:")

    print(
        "  Leave-one-article-out"
    )

    print()

    print("Personalization:")

    print(
        "  Current user's interactions only"
    )

    print()

    print("-" * 70)

    print("USER COVERAGE")

    print("-" * 70)

    print(
        f"Total users:       "
        f"{evaluation['users']}"
    )

    print(
        f"Evaluated users:   "
        f"{evaluation['evaluated_users']}"
    )

    print(
        f"Skipped users:     "
        f"{evaluation['skipped_users']}"
    )

    print()

    print("-" * 70)

    print("RANKING METRICS")

    print("-" * 70)

    print(
        f"{'K':<8}"
        f"{'Hit Rate':<15}"
        f"{'MRR':<15}"
        f"{'NDCG':<15}"
    )

    print("-" * 70)

    for k in EVALUATION_KS:

        metrics = evaluation["metrics"][k]

        print(
            f"@{k:<7}"
            f"{format_percentage(metrics['hit_rate']):<15}"
            f"{format_decimal(metrics['mrr']):<15}"
            f"{format_decimal(metrics['ndcg']):<15}"
        )

    print()

    print(
        f"Average rank: "
        f"{format_decimal(evaluation['average_rank'])}"
    )

    print("=" * 70)


def print_per_user_results(evaluation):
    """
    Print detailed per-user evaluation results.

    This is useful for diagnosing why recommendations work
    differently for different users.
    """

    print()

    print("=" * 70)

    print("PER-USER EVALUATION")

    print("=" * 70)

    if not evaluation["results"]:

        print()

        print(
            "No users were eligible for evaluation."
        )

        print("=" * 70)

        return

    for result in evaluation["results"]:

        print()

        print(
            f"User ID: "
            f"{result['user_id']}"
        )

        print(
            f"Held-out article: "
            f"{result['held_out_article_id']}"
        )

        print(
            f"Training articles: "
            f"{result['training_article_count']}"
        )

        print(
            f"Recommendations generated: "
            f"{result['recommendation_count']}"
        )

        rank = result["rank"]

        if rank is None:

            print(
                "Held-out article rank: "
                "Not found"
            )

        else:

            print(
                "Held-out article rank: "
                f"{rank}"
            )

        for k in EVALUATION_KS:

            metrics = result["metrics"][k]

            print(
                f"  @{k}: "
                f"Hit={metrics['hit']} "
                f"MRR={metrics['mrr']:.4f} "
                f"NDCG={metrics['ndcg']:.4f}"
            )

    print()

    print("=" * 70)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    """
    Run the complete offline evaluation.
    """

    evaluation = evaluate_recommender()

    print_evaluation_report(
        evaluation
    )

    print_per_user_results(
        evaluation
    )


if __name__ == "__main__":

    main()