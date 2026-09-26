from app import db
from app.models import Article, Interaction, User
from ml.recommender import (
    COLD_START_THRESHOLD,
    popularity_ranking,
    recommend_from_interactions,
)


def create_user(username, email):
    user = User(
        username=username,
        email=email,
    )
    user.set_password("TestPassword123")

    db.session.add(user)
    db.session.commit()

    return user


def create_article(title, category="Technology", url=None):
    if url is None:
        url = (
            "https://example.com/"
            + title.lower().replace(" ", "-")
        )

    article = Article(
        title=title,
        content=f"Content for {title}",
        category=category,
        source="Test Source",
        url=url,
    )

    db.session.add(article)
    db.session.commit()

    return article


def create_interaction(
    user_id,
    article_id,
    interaction_type,
):
    interaction = Interaction(
        user_id=user_id,
        article_id=article_id,
        type=interaction_type,
    )

    db.session.add(interaction)
    db.session.commit()

    return interaction


def test_popularity_ranking_returns_articles(app):
    with app.app_context():
        user = create_user(
            "popuser",
            "popuser@example.com",
        )

        article1 = create_article("Popular Article")
        article2 = create_article("Less Popular Article")

        create_interaction(
            user.id,
            article1.id,
            "view",
        )

        create_interaction(
            user.id,
            article2.id,
            "view",
        )

        recommendations = popularity_ranking(top_n=10)

        recommended_ids = {
            article_id
            for article_id, _ in recommendations
        }

        assert article1.id in recommended_ids
        assert article2.id in recommended_ids


def test_recommend_from_interactions_excludes_seen_articles(app):
    with app.app_context():
        article1 = create_article("Seen Article")
        article2 = create_article("Unseen Article")

        interactions = [
            (article1.id, "view"),
        ]

        recommendations = recommend_from_interactions(
            interactions,
            exclude_ids={article1.id},
            top_n=10,
        )

        recommended_ids = {
            article_id
            for article_id, _ in recommendations
        }

        assert article1.id not in recommended_ids
        assert article2.id in recommended_ids


def test_recommend_from_interactions_returns_at_most_top_n(app):
    with app.app_context():
        for index in range(10):
            create_article(
                f"Recommendation Article {index}"
            )

        recommendations = recommend_from_interactions(
            [],
            exclude_ids=set(),
            top_n=3,
        )

        assert len(recommendations) <= 3


def test_cold_start_threshold_is_defined():
    assert COLD_START_THRESHOLD == 3


def test_article_url_must_be_unique(app):
    with app.app_context():
        first = create_article(
            "First Article",
            url="https://example.com/duplicate",
        )

        assert first.id is not None

        second = Article(
            title="Second Article",
            content="Duplicate URL test",
            category="Technology",
            source="Test Source",
            url="https://example.com/duplicate",
        )

        db.session.add(second)

        duplicate_rejected = False

        try:
            db.session.commit()
        except Exception:
            db.session.rollback()
            duplicate_rejected = True

        assert duplicate_rejected