"""
config.py

Flask configuration. SECRET_KEY should come from an environment variable
in production (it's used to sign session cookies) — the default here is
only for local development.
"""

import os


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-this")
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", "sqlite:///app.db"
    )  # stored in instance/app.db by default (Flask's instance folder)
    SQLALCHEMY_TRACK_MODIFICATIONS = False