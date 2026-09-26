"""
app/__init__.py

Flask application factory.
"""

import os

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager


db = SQLAlchemy()
login_manager = LoginManager()


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

    return app