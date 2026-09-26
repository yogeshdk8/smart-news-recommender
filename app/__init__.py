"""
app/__init__.py

Flask application factory.
"""

import json
import os

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager


db = SQLAlchemy()
login_manager = LoginManager()


def seed_articles_if_empty(app_testing=False):
    """
    Seed the database with the processed articles when the database
    currently contains no articles.

    This is used for the deployed application so a fresh/empty
    database can automatically get the project's initial articles.
    """

    from app.models import Article

    # Do not seed test databases.
    if app_testing:
        return

    # If articles already exist, there is nothing to do.
    if Article.query.count() > 0:
        return

    processed_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "processed",
        "articles_cleaned.json",
    )

    if not os.path.exists(processed_path):
        print(
            f"Article seed file not found: {processed_path}"
        )
        return

    print("No articles found. Seeding initial articles...")

    with open(processed_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    added = 0
    skipped = 0

    for article_data in articles:
        title = article_data.get("title") or "Untitled"
        url = article_data.get("url")

        # Prefer URL-based duplicate detection.
        if url:
            exists = Article.query.filter_by(url=url).first()
        else:
            exists = Article.query.filter_by(title=title).first()

        if exists:
            skipped += 1
            continue

        article = Article(
            title=title,
            content=(
                article_data.get("description")
                or article_data.get("content")
                or ""
            ),
            category=article_data.get("category"),
            source=article_data.get("source"),
            url=url,
            published_at=article_data.get("published_at"),
        )

        db.session.add(article)
        added += 1

    db.session.commit()

    print(
        f"Initial article seed complete: "
        f"{added} added, {skipped} skipped."
    )


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)

    app.config.from_object("config.Config")

    if test_config is not None:
        app.config.update(test_config)

    os.makedirs(app.instance_path, exist_ok=True)

    # Use the normal instance database unless a test configuration
    # explicitly supplies another database URI.
    if "SQLALCHEMY_DATABASE_URI" not in (
        test_config or {}
    ):
        app.config["SQLALCHEMY_DATABASE_URI"] = (
            "sqlite:///"
            + os.path.join(app.instance_path, "app.db")
        )

    db.init_app(app)

    login_manager.init_app(app)
    login_manager.login_view = "main.login"

    from app.models import User

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    from app.routes import main_bp

    app.register_blueprint(main_bp)

    with app.app_context():
        db.create_all()
        seed_articles_if_empty(
            app_testing=app.config.get("TESTING", False)
        )

    return app