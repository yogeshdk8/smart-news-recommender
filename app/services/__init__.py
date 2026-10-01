from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy

from config import Config


# ============================================================
# EXTENSIONS
# ============================================================

db = SQLAlchemy()

login_manager = LoginManager()

login_manager.login_view = "main.login"
login_manager.login_message = "Please log in to access this page."
login_manager.login_message_category = "info"


# ============================================================
# APPLICATION FACTORY
# ============================================================

def create_app(config_class=Config):
    """
    Create and configure the Flask application.
    """

    app = Flask(__name__)

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    app.config.from_object(config_class)

    # --------------------------------------------------------
    # Initialize Flask extensions
    # --------------------------------------------------------

    db.init_app(app)

    login_manager.init_app(app)

    # --------------------------------------------------------
    # Register routes and create database tables
    # --------------------------------------------------------

    with app.app_context():

        # Import models here to avoid circular imports.
        from app.models import User, Article, Interaction

        # Create database tables if they do not already exist.
        db.create_all()

        # ----------------------------------------------------
        # Register blueprint
        # ----------------------------------------------------

        from app.routes import main_bp

        app.register_blueprint(main_bp)

        # ----------------------------------------------------
        # Auto-seed articles if database is empty
        # ----------------------------------------------------
        #
        # This keeps a fresh deployment from having an empty
        # Article table.
        #
        # If articles already exist, nothing is added.
        # ----------------------------------------------------

        try:

            article_count = Article.query.count()

            if article_count == 0:

                from scripts.seed_articles import seed_articles

                seed_articles()

        except Exception as exc:

            # Do not prevent the Flask application from starting
            # if automatic seeding fails.
            print(
                f"Auto-seeding skipped/failed: {exc}"
            )

    # --------------------------------------------------------
    # Flask-Login user loader
    # --------------------------------------------------------

    @login_manager.user_loader
    def load_user(user_id):
        """
        Load a user from the database using the stored
        session user ID.
        """

        try:

            return db.session.get(
                User,
                int(user_id),
            )

        except (
            ValueError,
            TypeError,
        ):

            return None

    return app