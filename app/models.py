"""
app/models.py

Database schema:
  User        - registered users, with hashed passwords
  Article     - news articles (loaded from your Phase 2-4 pipeline)
  Interaction - a log row every time a user views/likes/saves an article;
                this history is the raw material for their "interest profile"
"""

from datetime import datetime, timezone

from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

from app import db


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

    interactions = db.relationship(
        "Interaction", backref="user", lazy=True, cascade="all, delete-orphan"
    )

    def set_password(self, plain_password):
        """Hash and store a password. Never store plain text."""
        self.password_hash = generate_password_hash(plain_password)

    def check_password(self, plain_password):
        """Compare a plain-text password against the stored hash."""
        return check_password_hash(self.password_hash, plain_password)

    def __repr__(self):
        return f"<User {self.username}>"


class Article(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(300), nullable=False)
    content = db.Column(db.Text)
    category = db.Column(db.String(50))
    source = db.Column(db.String(100))
    url = db.Column(db.String(500))
    published_at = db.Column(db.String(50))

    interactions = db.relationship(
        "Interaction", backref="article", lazy=True, cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Article {self.title[:40]!r}>"


class Interaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    article_id = db.Column(db.Integer, db.ForeignKey("article.id"), nullable=False)
    type = db.Column(db.String(20), nullable=False)  # "view" / "like" / "save"
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<Interaction user={self.user_id} article={self.article_id} type={self.type}>"