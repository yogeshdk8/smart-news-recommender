"""
app/__init__.py

Flask "application factory" pattern: create_app() builds and configures
the Flask app, rather than creating it at import time. This is the
standard structure once your app has models + routes in separate files
(avoids circular imports).
"""

import os

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

db = SQLAlchemy()
login_manager = LoginManager()


def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object("config.Config")

    # Flask's "instance" folder holds things that shouldn't be in version
    # control (like the SQLite file) — it's already covered by .gitignore.
    os.makedirs(app.instance_path, exist_ok=True)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(
        app.instance_path, "app.db"
    )

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "main.login"  # where @login_required redirects to

    # Import models here (after db.init_app) so Flask-Login can find User
    from app.models import User

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Register routes
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    # Create tables if they don't exist yet
    with app.app_context():
        db.create_all()

    return app