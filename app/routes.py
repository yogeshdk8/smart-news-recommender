"""
app/routes.py

Routes:
  GET/POST /register           - create a new user
  GET/POST /login              - log in, starts a session
  GET      /logout             - end the session
  GET      /articles           - list all articles (login required)
  GET      /articles/<id>      - view an article -> logs a view Interaction
  POST     /articles/<id>/like - logs a like Interaction
  POST     /articles/<id>/save - logs a save Interaction
  GET      /feed               - personalized recommendation feed
  GET      /analytics          - recommendation system analytics dashboard
"""

from flask import (
    Blueprint,
    render_template,
    redirect,
    url_for,
    request,
    flash,
)

from flask_login import (
    login_user,
    logout_user,
    login_required,
    current_user,
)

from app import db
from app.models import User, Article, Interaction

from ml.recommender import hybrid_recommend
from ml.evaluate import hit_rate_at_k


main_bp = Blueprint("main", __name__)


# ---------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------

@main_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not username or not email or not password:
            flash("All fields are required.")
            return redirect(url_for("main.register"))

        if User.query.filter_by(username=username).first():
            flash("That username is already taken.")
            return redirect(url_for("main.register"))

        if User.query.filter_by(email=email).first():
            flash("That email is already registered.")
            return redirect(url_for("main.register"))

        user = User(
            username=username,
            email=email,
        )

        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        flash("Account created. You can log in now.")

        return redirect(url_for("main.login"))

    return render_template("register.html")


@main_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter_by(username=username).first()

        if user is None or not user.check_password(password):
            flash("Invalid username or password.")
            return redirect(url_for("main.login"))

        login_user(user)

        return redirect(url_for("main.list_articles"))

    return render_template("login.html")


@main_bp.route("/logout")
@login_required
def logout():
    logout_user()

    flash("Logged out.")

    return redirect(url_for("main.login"))


# ---------------------------------------------------------------------
# Article + interaction routes
# ---------------------------------------------------------------------

@main_bp.route("/")
@main_bp.route("/articles")
@login_required
def list_articles():
    articles = (
        Article.query
        .order_by(Article.id.desc())
        .limit(50)
        .all()
    )

    return render_template(
        "articles.html",
        articles=articles,
    )


@main_bp.route("/articles/<int:article_id>")
@login_required
def view_article(article_id):
    article = Article.query.get_or_404(article_id)

    # Log the view every time this page is loaded.
    interaction = Interaction(
        user_id=current_user.id,
        article_id=article.id,
        type="view",
    )

    db.session.add(interaction)
    db.session.commit()

    return render_template(
        "article_detail.html",
        article=article,
    )


@main_bp.route("/articles/<int:article_id>/like", methods=["POST"])
@login_required
def like_article(article_id):
    article = Article.query.get_or_404(article_id)

    interaction = Interaction(
        user_id=current_user.id,
        article_id=article.id,
        type="like",
    )

    db.session.add(interaction)
    db.session.commit()

    flash(f"Liked: {article.title}")

    return redirect(
        url_for(
            "main.view_article",
            article_id=article.id,
        )
    )


@main_bp.route("/articles/<int:article_id>/save", methods=["POST"])
@login_required
def save_article(article_id):
    article = Article.query.get_or_404(article_id)

    interaction = Interaction(
        user_id=current_user.id,
        article_id=article.id,
        type="save",
    )

    db.session.add(interaction)
    db.session.commit()

    flash(f"Saved: {article.title}")

    return redirect(
        url_for(
            "main.view_article",
            article_id=article.id,
        )
    )


# ---------------------------------------------------------------------
# Personalized recommendation feed
# ---------------------------------------------------------------------

@main_bp.route("/feed")
@login_required
def feed():
    """
    Generate personalized recommendations for the logged-in user.

    Recommendations are calculated on every request, so the feed
    automatically reflects newly recorded interactions.
    """

    recommendations = hybrid_recommend(
        current_user.id,
        top_n=10,
    )

    article_ids = [
        article_id
        for article_id, _ in recommendations
    ]

    articles_by_id = {
        article.id: article
        for article in Article.query.filter(
            Article.id.in_(article_ids)
        ).all()
    }

    feed_articles = []

    for article_id, score in recommendations:
        article = articles_by_id.get(article_id)

        if article is not None:
            feed_articles.append(
                {
                    "article": article,
                    "score": score,
                }
            )

    return render_template(
        "feed.html",
        recommendations=feed_articles,
    )


# ---------------------------------------------------------------------
# Analytics dashboard
# ---------------------------------------------------------------------

@main_bp.route("/analytics")
@login_required
def analytics():
    """
    Analytics dashboard.

    Shows:
      - views over time
      - article views by category
      - system Hit Rate@5
    """

    # ---------------------------------------------------------------
    # Get all view interactions
    # ---------------------------------------------------------------

    interactions = (
        Interaction.query
        .filter_by(type="view")
        .order_by(Interaction.timestamp.asc())
        .all()
    )

    # ---------------------------------------------------------------
    # Views over time
    # ---------------------------------------------------------------

    views_by_date = {}

    for interaction in interactions:
        if interaction.timestamp is None:
            continue

        date_key = interaction.timestamp.date().isoformat()

        views_by_date[date_key] = (
            views_by_date.get(date_key, 0) + 1
        )

    views_dates = list(views_by_date.keys())
    views_counts = list(views_by_date.values())

    # ---------------------------------------------------------------
    # Category breakdown
    # ---------------------------------------------------------------

    category_counts = {}

    for interaction in interactions:
        article = interaction.article

        if article is None:
            continue

        category = article.category or "Unknown"

        category_counts[category] = (
            category_counts.get(category, 0) + 1
        )

    categories = list(category_counts.keys())
    category_values = list(category_counts.values())

    # ---------------------------------------------------------------
    # Recommendation system Hit Rate@5
    # ---------------------------------------------------------------

    hit_rate, evaluated_users = hit_rate_at_k(k=5)

    if hit_rate is not None:
        hit_rate_display = f"{hit_rate:.1%}"
    else:
        hit_rate_display = "Not enough data"

    # ---------------------------------------------------------------
    # Render dashboard
    # ---------------------------------------------------------------

    return render_template(
        "analytics.html",
        views_dates=views_dates,
        views_counts=views_counts,
        categories=categories,
        category_values=category_values,
        hit_rate=hit_rate_display,
        evaluated_users=evaluated_users,
    )