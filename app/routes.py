"""
app/routes.py

Application routes for:
- authentication
- latest news
- article details
- likes
- saves
- personalized feed
- saved articles
- GNews news search
- analytics
"""

from collections import Counter
from datetime import datetime

from flask import (
    Blueprint,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)

from flask_login import (
    current_user,
    login_required,
    login_user,
    logout_user,
)

from sqlalchemy import func

from app import db
from app.models import Article, Interaction, User

from app.services.gnews import (
    GNewsError,
    search_news,
)

from ml.evaluate import evaluate_hit_rate

# IMPORTANT:
# Your production "Your Feed" uses ONLY the currently
# logged-in user's own interaction history.
from ml.recommender import personalized_recommend


main_bp = Blueprint("main", __name__)


# ============================================================
# HELPER: CONVERT GNEWS ARTICLE TO DATABASE ARTICLE
# ============================================================

def get_or_create_gnews_article(gnews_article):
    """
    Convert a GNews article dictionary into a local Article
    database record.

    If an article with the same URL already exists, the existing
    Article is returned.

    If it does not exist, a new Article is created.
    """

    url = (
        gnews_article.get("url") or ""
    ).strip()

    if not url:
        return None

    article = Article.query.filter_by(
        url=url
    ).first()

    if article:
        return article

    source_data = (
        gnews_article.get("source")
        or {}
    )

    source_name = source_data.get(
        "name"
    )

    published_at = None

    published_value = (
        gnews_article.get("publishedAt")
    )

    if published_value:
        try:
            parsed_datetime = datetime.fromisoformat(
                published_value.replace(
                    "Z",
                    "+00:00",
                )
            )

            if parsed_datetime.tzinfo is not None:
                parsed_datetime = (
                    parsed_datetime.replace(
                        tzinfo=None
                    )
                )

            published_at = parsed_datetime

        except (
            ValueError,
            TypeError,
        ):
            published_at = None

    article = Article(
        title=(
            gnews_article.get("title")
            or "Untitled"
        ),

        content=(
            gnews_article.get("description")
            or gnews_article.get("content")
            or ""
        ),

        category="General",

        source=source_name,

        url=url,

        published_at=published_at,

        image_url=gnews_article.get(
            "image"
        ),

        video_url=None,

        audio_url=None,
    )

    db.session.add(article)
    db.session.commit()

    return article


# ============================================================
# HOME
# ============================================================

@main_bp.route("/")
def home():
    """
    Send visitors to the appropriate page.
    """

    if current_user.is_authenticated:
        return redirect(
            url_for("main.list_articles")
        )

    return redirect(
        url_for("main.login")
    )


# ============================================================
# REGISTER
# ============================================================

@main_bp.route(
    "/register",
    methods=["GET", "POST"],
)
def register():
    """
    Register a new user.
    """

    if current_user.is_authenticated:
        return redirect(
            url_for("main.list_articles")
        )

    if request.method == "POST":

        username = (
            request.form.get("username")
            or ""
        ).strip()

        email = (
            request.form.get("email")
            or ""
        ).strip().lower()

        password = (
            request.form.get("password")
            or ""
        )

        confirm_password = (
            request.form.get(
                "confirm_password"
            )
            or request.form.get(
                "password_confirm"
            )
            or ""
        )

        if (
            not username
            or not email
            or not password
        ):
            flash(
                "Please fill in all required fields.",
                "danger",
            )

            return render_template(
                "register.html"
            )

        if password != confirm_password:
            flash(
                "Passwords do not match.",
                "danger",
            )

            return render_template(
                "register.html"
            )

        existing_username = (
            User.query.filter_by(
                username=username
            ).first()
        )

        if existing_username:
            flash(
                "That username is already registered.",
                "danger",
            )

            return render_template(
                "register.html"
            )

        existing_email = (
            User.query.filter_by(
                email=email
            ).first()
        )

        if existing_email:
            flash(
                "That email address is already registered.",
                "danger",
            )

            return render_template(
                "register.html"
            )

        user = User(
            username=username,
            email=email,
        )

        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        flash(
            "Registration successful. Please log in.",
            "success",
        )

        return redirect(
            url_for("main.login")
        )

    return render_template(
        "register.html"
    )


# ============================================================
# LOGIN
# ============================================================

@main_bp.route(
    "/login",
    methods=["GET", "POST"],
)
def login():
    """
    Log an existing user in.
    """

    if current_user.is_authenticated:
        return redirect(
            url_for("main.list_articles")
        )

    if request.method == "POST":

        username = (
            request.form.get("username")
            or ""
        ).strip()

        password = (
            request.form.get("password")
            or ""
        )

        user = User.query.filter_by(
            username=username
        ).first()

        if (
            user is None
            or not user.check_password(password)
        ):
            flash(
                "Invalid username or password.",
                "danger",
            )

            return render_template(
                "login.html"
            )

        login_user(user)

        next_page = request.args.get(
            "next"
        )

        if (
            next_page
            and next_page.startswith("/")
        ):
            return redirect(
                next_page
            )

        return redirect(
            url_for("main.list_articles")
        )

    return render_template(
        "login.html"
    )


# ============================================================
# LOGOUT
# ============================================================

@main_bp.route("/logout")
@login_required
def logout():
    """
    Log the current user out.
    """

    logout_user()

    flash(
        "You have been logged out.",
        "success",
    )

    return redirect(
        url_for("main.login")
    )


# ============================================================
# SEARCH NEWS
# ============================================================

@main_bp.route("/search")
@login_required
def search():
    """
    Search GNews for news articles.

    GNews results are converted into local Article records
    so they can use:
    - Like
    - Save
    - View
    - Analytics
    - Recommendations
    """

    query = (
        request.args.get("q") or ""
    ).strip()

    if not query:
        return render_template(
            "search.html",
            query="",
            articles=[],
            error=None,
        )

    try:

        gnews_articles = search_news(
            query=query,
            max_articles=10,
        )

    except GNewsError as exc:

        print(
            f"GNews search error: {exc}"
        )

        return render_template(
            "search.html",
            query=query,
            articles=[],
            error=str(exc),
        )

    articles = []

    for gnews_article in gnews_articles:

        try:

            article = (
                get_or_create_gnews_article(
                    gnews_article
                )
            )

            if article is not None:
                articles.append(
                    article
                )

        except Exception as exc:

            db.session.rollback()

            print(
                f"Could not save GNews article: {exc}"
            )

    liked_ids, saved_ids = (
        get_user_action_ids()
    )

    return render_template(
        "search.html",
        query=query,
        articles=articles,
        liked_ids=liked_ids,
        saved_ids=saved_ids,
        error=None,
    )


# ============================================================
# HELPER: USER INTERACTION IDS
# ============================================================

def get_user_action_ids():
    """
    Return sets containing the current user's liked and saved
    article IDs.

    IMPORTANT:
    These IDs belong ONLY to the currently logged-in user.
    """

    interactions = (
        Interaction.query
        .filter(
            Interaction.user_id
            == current_user.id,

            Interaction.type.in_(
                [
                    "like",
                    "save",
                ]
            ),
        )
        .all()
    )

    liked_ids = set()
    saved_ids = set()

    for interaction in interactions:

        if interaction.type == "like":

            liked_ids.add(
                interaction.article_id
            )

        elif interaction.type == "save":

            saved_ids.add(
                interaction.article_id
            )

    return (
        liked_ids,
        saved_ids,
    )


# ============================================================
# LATEST NEWS
# ============================================================

@main_bp.route("/articles")
@login_required
def list_articles():
    """
    Display the latest news articles.
    """

    articles = (
        Article.query
        .order_by(
            Article.created_at.desc(),
            Article.id.desc(),
        )
        .limit(50)
        .all()
    )

    liked_ids, saved_ids = (
        get_user_action_ids()
    )

    return render_template(
        "articles.html",
        articles=articles,
        liked_ids=liked_ids,
        saved_ids=saved_ids,
    )


# ============================================================
# ARTICLE OPEN / DIRECT REDIRECT
# ============================================================

@main_bp.route(
    "/articles/<int:article_id>"
)
@login_required
def view_article(article_id):
    """
    Record a view and immediately open the original news article.

    The application does not show an intermediate article-detail
    page.

    Flow:

        User clicks "Read Full Article"
                    |
                    v
            Find local Article
                    |
                    v
            Record view interaction
                    |
                    v
          Redirect to article.url
                    |
                    v
          Original news website
    """

    article = db.session.get(
        Article,
        article_id,
    )

    if article is None:

        flash(
            "Article not found.",
            "danger",
        )

        return redirect(
            url_for("main.list_articles")
        )

    # --------------------------------------------------------
    # Record view.
    #
    # Only create one view interaction for the article/user.
    # --------------------------------------------------------

    existing_view = (
        Interaction.query
        .filter_by(
            user_id=current_user.id,
            article_id=article.id,
            type="view",
        )
        .first()
    )

    if existing_view is None:

        interaction = Interaction(
            user_id=current_user.id,
            article_id=article.id,
            type="view",
        )

        db.session.add(
            interaction
        )

        db.session.commit()

    # --------------------------------------------------------
    # Open original article.
    # --------------------------------------------------------

    if article.url:
        return redirect(
            article.url
        )

    flash(
        "The original article link is not available.",
        "warning",
    )

    return redirect(
        url_for("main.list_articles")
    )


# ============================================================
# LIKE ARTICLE
# ============================================================

@main_bp.route(
    "/articles/<int:article_id>/like",
    methods=["POST"],
)
@login_required
def like_article(article_id):
    """
    Like an article.

    A duplicate like is not created if the user already
    liked the article.
    """

    article = db.session.get(
        Article,
        article_id,
    )

    if article is None:

        flash(
            "Article not found.",
            "danger",
        )

        return redirect(
            url_for("main.list_articles")
        )

    existing_like = (
        Interaction.query
        .filter_by(
            user_id=current_user.id,
            article_id=article.id,
            type="like",
        )
        .first()
    )

    if existing_like:

        flash(
            "You already liked this article.",
            "info",
        )

    else:

        interaction = Interaction(
            user_id=current_user.id,
            article_id=article.id,
            type="like",
        )

        db.session.add(
            interaction
        )

        db.session.commit()

        flash(
            "Article liked.",
            "success",
        )

    return redirect(
        request.referrer
        or url_for("main.list_articles")
    )


# ============================================================
# SAVE ARTICLE
# ============================================================

@main_bp.route(
    "/articles/<int:article_id>/save",
    methods=["POST"],
)
@login_required
def save_article(article_id):
    """
    Save an article.

    A duplicate save is not created if the user already
    saved the article.
    """

    article = db.session.get(
        Article,
        article_id,
    )

    if article is None:

        flash(
            "Article not found.",
            "danger",
        )

        return redirect(
            url_for("main.list_articles")
        )

    existing_save = (
        Interaction.query
        .filter_by(
            user_id=current_user.id,
            article_id=article.id,
            type="save",
        )
        .first()
    )

    if existing_save:

        flash(
            "You already saved this article.",
            "info",
        )

    else:

        interaction = Interaction(
            user_id=current_user.id,
            article_id=article.id,
            type="save",
        )

        db.session.add(
            interaction
        )

        db.session.commit()

        flash(
            "Article saved.",
            "success",
        )

    return redirect(
        request.referrer
        or url_for("main.list_articles")
    )


# ============================================================
# SAVED ARTICLES
# ============================================================

@main_bp.route("/saved")
@login_required
def saved_articles():
    """
    Display articles saved by the current user.
    """

    saved_interactions = (
        Interaction.query
        .filter_by(
            user_id=current_user.id,
            type="save",
        )
        .order_by(
            Interaction.timestamp.desc(),
            Interaction.id.desc(),
        )
        .all()
    )

    articles = []
    seen_article_ids = set()

    for interaction in saved_interactions:

        article = interaction.article

        if article is None:
            continue

        if article.id in seen_article_ids:
            continue

        articles.append(
            article
        )

        seen_article_ids.add(
            article.id
        )

    liked_ids, saved_ids = (
        get_user_action_ids()
    )

    return render_template(
        "saved.html",
        articles=articles,
        liked_ids=liked_ids,
        saved_ids=saved_ids,
    )


# ============================================================
# PERSONALIZED FEED
# ============================================================

@main_bp.route("/feed")
@login_required
def feed():
    """
    Generate a strictly personalized feed for the
    currently logged-in user.

    IMPORTANT:

    The recommendation engine receives ONLY:

        current_user.id

    The recommender then uses ONLY that user's:

        - views
        - likes
        - saves

    It does NOT use:

        - other users' interactions
        - global popularity
        - collaborative filtering
        - NCF recommendations

    Therefore:

        User 1 -> User 1 interaction history -> User 1 feed

        User 2 -> User 2 interaction history -> User 2 feed
    """

    try:

        # ----------------------------------------------------
        # THIS IS THE MOST IMPORTANT LINE.
        #
        # current_user.id identifies the user who is currently
        # logged into the application.
        # ----------------------------------------------------

        user_id = current_user.id

        recommendations = personalized_recommend(
            user_id=user_id,
            top_n=20,
        )

    except Exception as exc:

        print(
            f"Personalized recommendation error: {exc}"
        )

        recommendations = []

    articles = []

    for item in recommendations:

        article = None
        score = None

        # ----------------------------------------------------
        # Dictionary output.
        # ----------------------------------------------------

        if isinstance(
            item,
            dict,
        ):

            article_id = item.get(
                "article_id"
            )

            score = item.get(
                "score"
            )

            if article_id is not None:

                article = db.session.get(
                    Article,
                    article_id,
                )

        # ----------------------------------------------------
        # Tuple/list output.
        #
        # personalized_recommend() currently returns:
        #
        #     (article_id, score)
        #
        # ----------------------------------------------------

        elif isinstance(
            item,
            (tuple, list),
        ):

            if len(item) >= 1:

                article_id = item[0]

                article = db.session.get(
                    Article,
                    article_id,
                )

            if len(item) >= 2:

                score = item[1]

        # ----------------------------------------------------
        # Plain article ID.
        # ----------------------------------------------------

        elif isinstance(
            item,
            int,
        ):

            article = db.session.get(
                Article,
                item,
            )

        if article is not None:

            articles.append(
                {
                    "article": article,
                    "score": score,
                }
            )

    liked_ids, saved_ids = (
        get_user_action_ids()
    )

    return render_template(
        "feed.html",
        recommendations=articles,
        liked_ids=liked_ids,
        saved_ids=saved_ids,
    )


# ============================================================
# ANALYTICS
# ============================================================

@main_bp.route("/analytics")
@login_required
def analytics():
    """
    User analytics dashboard.

    Shows:

    - total views
    - total likes
    - total saves
    - unique articles viewed
    - available article categories
    - views over time
    - article view counts
    - views by category
    - liked categories
    - saved categories
    - Hit Rate@5
    - Hit Rate@10
    - Hit Rate@20
    """

    user_id = current_user.id

    # ========================================================
    # CURRENT USER INTERACTIONS
    # ========================================================

    user_interactions = (
        Interaction.query
        .filter_by(
            user_id=user_id
        )
        .order_by(
            Interaction.timestamp.asc()
        )
        .all()
    )

    views = [
        interaction
        for interaction in user_interactions
        if interaction.type == "view"
    ]

    likes = [
        interaction
        for interaction in user_interactions
        if interaction.type == "like"
    ]

    saves = [
        interaction
        for interaction in user_interactions
        if interaction.type == "save"
    ]

    total_views = len(views)
    total_likes = len(likes)
    total_saves = len(saves)

    unique_viewed_articles = len(
        {
            interaction.article_id
            for interaction in views
        }
    )

    # ========================================================
    # VIEWS OVER TIME
    # ========================================================

    views_by_date = Counter()

    for interaction in views:

        if interaction.timestamp is None:
            continue

        date_key = (
            interaction.timestamp
            .date()
            .isoformat()
        )

        views_by_date[date_key] += 1

    views_dates = sorted(
        views_by_date.keys()
    )

    views_counts = [
        views_by_date[date]
        for date in views_dates
    ]

    # ========================================================
    # ALL AVAILABLE ARTICLE CATEGORIES
    # ========================================================

    available_category_rows = (
        db.session.query(
            Article.category,
            func.count(Article.id),
        )
        .group_by(
            Article.category
        )
        .order_by(
            func.count(Article.id).desc()
        )
        .all()
    )

    available_category_counts = {}

    for category, count in (
        available_category_rows
    ):

        category_name = (
            category
            or "Unknown"
        )

        available_category_counts[
            category_name
        ] = count

    available_categories = list(
        available_category_counts.keys()
    )

    available_category_values = list(
        available_category_counts.values()
    )

    # ========================================================
    # ARTICLE VIEW COUNTS
    # ========================================================

    article_view_rows = (
        db.session.query(
            Article.id,
            Article.title,
            Article.category,
            func.count(Interaction.id),
        )
        .join(
            Interaction,
            Interaction.article_id == Article.id,
        )
        .filter(
            Interaction.user_id == user_id,
            Interaction.type == "view",
        )
        .group_by(
            Article.id,
            Article.title,
            Article.category,
        )
        .order_by(
            func.count(Interaction.id).desc()
        )
        .all()
    )

    viewed_articles = []

    for (
        article_id,
        title,
        category,
        count,
    ) in article_view_rows:

        viewed_articles.append(
            {
                "id": article_id,
                "title": title,
                "category": (
                    category
                    or "Unknown"
                ),
                "views": count,
            }
        )

    # ========================================================
    # VIEWS BY CATEGORY
    # ========================================================

    views_by_category = Counter()

    for interaction in views:

        if interaction.article is None:
            continue

        category = (
            interaction.article.category
            or "Unknown"
        )

        views_by_category[
            category
        ] += 1

    view_categories = list(
        views_by_category.keys()
    )

    view_category_values = [
        views_by_category[category]
        for category in view_categories
    ]

    # ========================================================
    # LIKED CATEGORIES
    # ========================================================

    liked_categories_counter = Counter()

    for interaction in likes:

        if interaction.article is None:
            continue

        category = (
            interaction.article.category
            or "Unknown"
        )

        liked_categories_counter[
            category
        ] += 1

    liked_categories = list(
        liked_categories_counter.keys()
    )

    liked_category_values = [
        liked_categories_counter[category]
        for category in liked_categories
    ]

    # ========================================================
    # SAVED CATEGORIES
    # ========================================================

    saved_categories_counter = Counter()

    for interaction in saves:

        if interaction.article is None:
            continue

        category = (
            interaction.article.category
            or "Unknown"
        )

        saved_categories_counter[
            category
        ] += 1

    saved_categories = list(
        saved_categories_counter.keys()
    )

    saved_category_values = [
        saved_categories_counter[category]
        for category in saved_categories
    ]

    # ========================================================
    # SYSTEM RECOMMENDATION METRICS
    # ========================================================

    # --------------------------------------------------------
    # Hit Rate@5
    # --------------------------------------------------------

    try:

        hit_rate_5, evaluated_users_5 = (
            evaluate_hit_rate(
                k=5
            )
        )

    except Exception as exc:

        print(
            f"Analytics Hit Rate@5 error: {exc}"
        )

        hit_rate_5 = None
        evaluated_users_5 = 0

    # --------------------------------------------------------
    # Hit Rate@10
    # --------------------------------------------------------

    try:

        hit_rate_10, evaluated_users_10 = (
            evaluate_hit_rate(
                k=10
            )
        )

    except Exception as exc:

        print(
            f"Analytics Hit Rate@10 error: {exc}"
        )

        hit_rate_10 = None
        evaluated_users_10 = 0

    # --------------------------------------------------------
    # Hit Rate@20
    # --------------------------------------------------------

    try:

        hit_rate_20, evaluated_users_20 = (
            evaluate_hit_rate(
                k=20
            )
        )

    except Exception as exc:

        print(
            f"Analytics Hit Rate@20 error: {exc}"
        )

        hit_rate_20 = None
        evaluated_users_20 = 0

    # --------------------------------------------------------
    # Format percentages for the template.
    # --------------------------------------------------------

    if hit_rate_5 is None:

        hit_rate_5_display = (
            "Not enough data"
        )

    else:

        hit_rate_5_display = (
            f"{hit_rate_5:.1%}"
        )

    if hit_rate_10 is None:

        hit_rate_10_display = (
            "Not enough data"
        )

    else:

        hit_rate_10_display = (
            f"{hit_rate_10:.1%}"
        )

    if hit_rate_20 is None:

        hit_rate_20_display = (
            "Not enough data"
        )

    else:

        hit_rate_20_display = (
            f"{hit_rate_20:.1%}"
        )

    # --------------------------------------------------------
    # Use the same evaluation population for the dashboard.
    # --------------------------------------------------------

    evaluated_users = max(
        evaluated_users_5,
        evaluated_users_10,
        evaluated_users_20,
    )

    # ========================================================
    # RENDER
    # ========================================================

    return render_template(
        "analytics.html",

        # Summary cards
        total_views=total_views,
        total_likes=total_likes,
        total_saves=total_saves,
        unique_viewed_articles=(
            unique_viewed_articles
        ),

        # Views over time
        views_dates=views_dates,
        views_counts=views_counts,

        # Available categories
        available_categories=(
            available_categories
        ),

        available_category_values=(
            available_category_values
        ),

        # Views by category
        view_categories=(
            view_categories
        ),

        view_category_values=(
            view_category_values
        ),

        # Article views
        viewed_articles=(
            viewed_articles
        ),

        # Liked categories
        liked_categories=(
            liked_categories
        ),

        liked_category_values=(
            liked_category_values
        ),

        # Saved categories
        saved_categories=(
            saved_categories
        ),

        saved_category_values=(
            saved_category_values
        ),

        # Recommendation metrics
        hit_rate=hit_rate_5_display,

        hit_rate_5=hit_rate_5_display,

        hit_rate_10=hit_rate_10_display,

        hit_rate_20=hit_rate_20_display,

        evaluated_users=evaluated_users,
    )