"""
app/routes.py

Routes:
  GET/POST /register        - create a new user
  GET/POST /login           - log in, starts a session
  GET      /logout          - end the session
  GET      /articles        - list all articles (login required)
  GET      /articles/<id>   - "view" an article -> logs a view Interaction
  POST     /articles/<id>/like  - logs a like Interaction
  POST     /articles/<id>/save  - logs a save Interaction
"""

from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_user, logout_user, login_required, current_user

from app import db
from app.models import User, Article, Interaction

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

        user = User(username=username, email=email)
        user.set_password(password)  # hashed here — never store plain text
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

        login_user(user)  # Flask-Login sets the session cookie
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
    articles = Article.query.order_by(Article.id.desc()).limit(50).all()
    return render_template("articles.html", articles=articles)


@main_bp.route("/articles/<int:article_id>")
@login_required
def view_article(article_id):
    article = Article.query.get_or_404(article_id)

    # Log the view every time this page is loaded
    interaction = Interaction(
        user_id=current_user.id, article_id=article.id, type="view"
    )
    db.session.add(interaction)
    db.session.commit()

    return render_template("article_detail.html", article=article)


@main_bp.route("/articles/<int:article_id>/like", methods=["POST"])
@login_required
def like_article(article_id):
    article = Article.query.get_or_404(article_id)
    interaction = Interaction(
        user_id=current_user.id, article_id=article.id, type="like"
    )
    db.session.add(interaction)
    db.session.commit()
    flash(f"Liked: {article.title}")
    return redirect(url_for("main.view_article", article_id=article.id))


@main_bp.route("/articles/<int:article_id>/save", methods=["POST"])
@login_required
def save_article(article_id):
    article = Article.query.get_or_404(article_id)
    interaction = Interaction(
        user_id=current_user.id, article_id=article.id, type="save"
    )
    db.session.add(interaction)
    db.session.commit()
    flash(f"Saved: {article.title}")
    return redirect(url_for("main.view_article", article_id=article.id))