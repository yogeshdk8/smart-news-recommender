"""
app/models.py

Database models for:
- users
- news articles
- user interactions
"""

from datetime import datetime

from flask_login import UserMixin
from werkzeug.security import (
    generate_password_hash,
    check_password_hash,
)

from app import db


class User(UserMixin, db.Model):
    """
    Application user.

    Stores:
    - username
    - email
    - hashed password
    """

    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(
        db.String(80),
        unique=True,
        nullable=False,
    )

    email = db.Column(
        db.String(120),
        unique=True,
        nullable=False,
    )

    password_hash = db.Column(
        db.String(255),
        nullable=False,
    )

    interactions = db.relationship(
        "Interaction",
        backref="user",
        lazy=True,
        cascade="all, delete-orphan",
    )

    def set_password(self, password):
        """Hash and store the user's password."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Check a plain-text password against the stored hash."""
        return check_password_hash(
            self.password_hash,
            password,
        )


class Article(db.Model):
    """
    News article.

    Articles can contain:
    - title
    - description/content
    - category
    - source
    - URL
    - image
    - optional video
    - optional audio
    - publication date
    """

    id = db.Column(
        db.Integer,
        primary_key=True,
    )

    title = db.Column(
        db.String(300),
        nullable=False,
    )

    content = db.Column(
        db.Text,
        nullable=False,
        default="",
    )

    category = db.Column(
        db.String(100),
        nullable=True,
    )

    source = db.Column(
        db.String(100),
        nullable=True,
    )

    url = db.Column(
        db.String(500),
        unique=True,
        nullable=True,
    )

    image_url = db.Column(
        db.String(1000),
        nullable=True,
    )

    video_url = db.Column(
        db.String(1000),
        nullable=True,
    )

    audio_url = db.Column(
        db.String(1000),
        nullable=True,
    )

    published_at = db.Column(
        db.String(100),
        nullable=True,
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
    )


class Interaction(db.Model):
    """
    User interaction with an article.

    Types:
    - view
    - like
    - save
    """

    id = db.Column(
        db.Integer,
        primary_key=True,
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False,
    )

    article_id = db.Column(
        db.Integer,
        db.ForeignKey("article.id"),
        nullable=False,
    )

    type = db.Column(
        db.String(20),
        nullable=False,
    )

    timestamp = db.Column(
        db.DateTime,
        default=datetime.utcnow,
    )

    article = db.relationship(
        "Article",
        backref=db.backref(
            "interactions",
            lazy=True,
        ),
    )