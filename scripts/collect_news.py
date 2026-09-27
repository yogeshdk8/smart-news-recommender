"""
scripts/collect_news.py

Fetches current news from GNews and stores new articles
in the Smart News Recommender database.

Media support:
- image_url -> directly from GNews
- video_url -> optionally discovered from publisher page
- audio_url -> optionally discovered from publisher page

The API key is read from GNEWS_API_KEY.
Never put the real API key directly into this file.
"""

import os

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from app import create_app, db
from app.models import Article


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

API_KEY = os.environ.get("GNEWS_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GNEWS_API_KEY is not set. "
        "Add your GNews API key to the environment or .env file."
    )


# ============================================================
# GNEWS CONFIGURATION
# ============================================================

GNEWS_URL = "https://gnews.io/api/v4/top-headlines"


# Map GNews categories to project categories.
CATEGORY_MAP = {
    "general": "General",
    "world": "World",
    "nation": "India",
    "business": "Business",
    "technology": "Technology",
    "entertainment": "Entertainment",
    "sports": "Sports",
    "science": "Science",
    "health": "Health",
}


# ============================================================
# FETCH NEWS FROM GNEWS
# ============================================================

def fetch_news(
    category="general",
    country="in",
    language="en",
    max_articles=10,
):
    """
    Fetch top headlines from GNews.
    """

    params = {
        "category": category,
        "country": country,
        "lang": language,
        "max": max_articles,
    }

    headers = {
        "X-Api-Key": API_KEY,
    }

    response = requests.get(
        GNEWS_URL,
        params=params,
        headers=headers,
        timeout=15,
    )

    if response.status_code != 200:
        print(
            f"GNews API error: HTTP {response.status_code}"
        )
        print(
            f"GNews response: {response.text}"
        )

        response.raise_for_status()

    return response.json()


# ============================================================
# OPTIONAL MEDIA DISCOVERY
# ============================================================

def discover_media_from_page(url):
    """
    Try to discover video/audio URLs from the publisher page.

    This is optional because GNews itself does not return
    dedicated video_url/audio_url fields.

    Returns:
        {
            "video_url": "...",
            "audio_url": "..."
        }
    """

    result = {
        "video_url": None,
        "audio_url": None,
    }

    if not url:
        return result

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/153.0.0.0 Safari/537.36"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=8,
        )

        if response.status_code != 200:
            return result

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        # ----------------------------------------------------
        # VIDEO DISCOVERY
        # ----------------------------------------------------

        # Open Graph video
        video_meta = soup.find(
            "meta",
            property="og:video",
        )

        if video_meta and video_meta.get("content"):
            result["video_url"] = video_meta["content"]

        # Alternative Open Graph video properties
        if not result["video_url"]:
            video_meta = soup.find(
                "meta",
                property="og:video:url",
            )

            if video_meta and video_meta.get("content"):
                result["video_url"] = video_meta["content"]

        # HTML <video>
        if not result["video_url"]:
            video_tag = soup.find("video")

            if video_tag:
                source_tag = video_tag.find("source")

                if source_tag and source_tag.get("src"):
                    result["video_url"] = source_tag["src"]

                elif video_tag.get("src"):
                    result["video_url"] = video_tag["src"]

        # ----------------------------------------------------
        # AUDIO DISCOVERY
        # ----------------------------------------------------

        # Open Graph audio
        audio_meta = soup.find(
            "meta",
            property="og:audio",
        )

        if audio_meta and audio_meta.get("content"):
            result["audio_url"] = audio_meta["content"]

        # Alternative Open Graph audio property
        if not result["audio_url"]:
            audio_meta = soup.find(
                "meta",
                property="og:audio:url",
            )

            if audio_meta and audio_meta.get("content"):
                result["audio_url"] = audio_meta["content"]

        # HTML <audio>
        if not result["audio_url"]:
            audio_tag = soup.find("audio")

            if audio_tag:
                source_tag = audio_tag.find("source")

                if source_tag and source_tag.get("src"):
                    result["audio_url"] = source_tag["src"]

                elif audio_tag.get("src"):
                    result["audio_url"] = audio_tag["src"]

        return result

    except requests.RequestException:
        # Media discovery is optional.
        # If a publisher blocks the request or times out,
        # the article itself should still be saved.
        return result

    except Exception:
        # Never allow media extraction to stop the
        # complete news collection process.
        return result


# ============================================================
# SAVE ARTICLES
# ============================================================

def save_articles(
    articles,
    project_category,
):
    """
    Save new GNews articles to the database.

    Duplicate prevention:
        URL is used as the unique identifier.

    Media:
        image_url is taken directly from GNews.
        video_url/audio_url are optionally discovered
        from the publisher page.
    """

    added = 0
    skipped = 0

    for article_data in articles:

        # ----------------------------------------------------
        # BASIC ARTICLE DATA
        # ----------------------------------------------------

        title = (
            article_data.get("title")
            or "Untitled"
        )

        description = (
            article_data.get("description")
            or article_data.get("content")
            or ""
        )

        source_data = (
            article_data.get("source")
            or {}
        )

        source = (
            source_data.get("name")
            or "Unknown"
        )

        url = article_data.get("url")

        published_at = (
            article_data.get("publishedAt")
        )

        # ----------------------------------------------------
        # IMAGE FROM GNEWS
        # ----------------------------------------------------

        image_url = (
            article_data.get("image")
        )

        # ----------------------------------------------------
        # URL IS REQUIRED
        # ----------------------------------------------------

        if not url:
            print(
                f"Skipping article without URL: "
                f"{title}"
            )

            skipped += 1
            continue

        # ----------------------------------------------------
        # DUPLICATE CHECK
        # ----------------------------------------------------

        existing_article = (
            Article.query
            .filter_by(url=url)
            .first()
        )

        if existing_article:

            # If an old article does not have an image,
            # update it from the new GNews response.
            changed = False

            if (
                not existing_article.image_url
                and image_url
            ):
                existing_article.image_url = (
                    image_url[:1000]
                )
                changed = True

            if changed:
                db.session.add(
                    existing_article
                )

            skipped += 1
            continue

        # ----------------------------------------------------
        # DISCOVER VIDEO / AUDIO
        # ----------------------------------------------------

        print(
            f"Checking media: {title[:70]}"
        )

        media = discover_media_from_page(url)

        video_url = media.get(
            "video_url"
        )

        audio_url = media.get(
            "audio_url"
        )

        # ----------------------------------------------------
        # CREATE ARTICLE
        # ----------------------------------------------------

        article = Article(
            title=title[:300],

            content=description,

            category=project_category,

            source=source[:100],

            url=url[:500],

            published_at=published_at,

            image_url=(
                image_url[:1000]
                if image_url
                else None
            ),

            video_url=(
                video_url[:1000]
                if video_url
                else None
            ),

            audio_url=(
                audio_url[:1000]
                if audio_url
                else None
            ),
        )

        db.session.add(article)

        added += 1

        print(
            f"  Image: "
            f"{'YES' if image_url else 'NO'}"
        )

        print(
            f"  Video: "
            f"{'YES' if video_url else 'NO'}"
        )

        print(
            f"  Audio: "
            f"{'YES' if audio_url else 'NO'}"
        )

    # --------------------------------------------------------
    # COMMIT
    # --------------------------------------------------------

    db.session.commit()

    return added, skipped


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("========================================")
    print("SMART NEWS RECOMMENDER")
    print("GNEWS MEDIA NEWS COLLECTOR")
    print("========================================")
    print()

    print("Connecting to GNews...")

    app = create_app()

    total_added = 0
    total_skipped = 0

    with app.app_context():

        # ----------------------------------------------------
        # FETCH EACH CATEGORY
        # ----------------------------------------------------

        for (
            gnews_category,
            project_category,
        ) in CATEGORY_MAP.items():

            print()
            print("----------------------------------------")

            print(
                f"Fetching {gnews_category} "
                f"-> {project_category}..."
            )

            print("----------------------------------------")

            data = fetch_news(
                category=gnews_category
            )

            articles = data.get(
                "articles",
                []
            )

            print(
                f"Articles returned: "
                f"{len(articles)}"
            )

            added, skipped = save_articles(
                articles,
                project_category,
            )

            total_added += added
            total_skipped += skipped

            print(
                f"New articles added: "
                f"{added}"
            )

            print(
                f"Duplicates skipped: "
                f"{skipped}"
            )

        # ----------------------------------------------------
        # FINAL SUMMARY
        # ----------------------------------------------------

        print()
        print("========================================")
        print("GNews database update complete.")
        print("========================================")

        print(
            f"New articles added: "
            f"{total_added}"
        )

        print(
            f"Duplicate articles skipped: "
            f"{total_skipped}"
        )

        print(
            f"Total articles in DB: "
            f"{Article.query.count()}"
        )

        # ----------------------------------------------------
        # MEDIA SUMMARY
        # ----------------------------------------------------

        image_count = (
            Article.query
            .filter(
                Article.image_url.isnot(None),
                Article.image_url != "",
            )
            .count()
        )

        video_count = (
            Article.query
            .filter(
                Article.video_url.isnot(None),
                Article.video_url != "",
            )
            .count()
        )

        audio_count = (
            Article.query
            .filter(
                Article.audio_url.isnot(None),
                Article.audio_url != "",
            )
            .count()
        )

        print()
        print("MEDIA SUMMARY")
        print("----------------------------------------")

        print(
            f"Articles with images: "
            f"{image_count}"
        )

        print(
            f"Articles with videos: "
            f"{video_count}"
        )

        print(
            f"Articles with audio: "
            f"{audio_count}"
        )

        print("----------------------------------------")
        print()


if __name__ == "__main__":
    main()