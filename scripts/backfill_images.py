"""
scripts/backfill_images.py

Finds missing article images from the original publisher pages.

The script:
1. Finds database articles with no image_url.
2. Opens the original article URL.
3. Looks for og:image.
4. Falls back to Twitter image metadata.
5. Saves the discovered image URL into the database.

Run from the project root:

    python -m scripts.backfill_images
"""

import time

import requests
from bs4 import BeautifulSoup

from app import create_app, db
from app.models import Article


REQUEST_TIMEOUT = 15
REQUEST_DELAY = 0.5

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/153.0.0.0 Safari/537.36"
)


def extract_image_url(article_url):
    """Extract the main image from an article web page."""

    if not article_url:
        return None

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,image/avif,image/webp,"
            "image/apng,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        response = requests.get(
            article_url,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )

        response.raise_for_status()

    except requests.RequestException as exc:
        print(f"    Request failed: {exc}")
        return None

    try:
        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

    except Exception as exc:
        print(f"    HTML parsing failed: {exc}")
        return None

    # --------------------------------------------------------
    # 1. Open Graph image
    # --------------------------------------------------------

    og_image = soup.find(
        "meta",
        attrs={"property": "og:image"},
    )

    if og_image:
        image_url = og_image.get("content")

        if image_url:
            image_url = image_url.strip()

            if image_url:
                return image_url

    # --------------------------------------------------------
    # 2. Twitter image
    # --------------------------------------------------------

    twitter_image = soup.find(
        "meta",
        attrs={"name": "twitter:image"},
    )

    if twitter_image:
        image_url = twitter_image.get("content")

        if image_url:
            image_url = image_url.strip()

            if image_url:
                return image_url

    # --------------------------------------------------------
    # 3. Twitter image source
    # --------------------------------------------------------

    twitter_image_src = soup.find(
        "meta",
        attrs={"name": "twitter:image:src"},
    )

    if twitter_image_src:
        image_url = twitter_image_src.get("content")

        if image_url:
            image_url = image_url.strip()

            if image_url:
                return image_url

    return None


def main():
    app = create_app()

    with app.app_context():

        articles = (
            Article.query
            .filter(
                Article.url.isnot(None),
                Article.url != "",
                (
                    Article.image_url.is_(None)
                    | (Article.image_url == "")
                ),
            )
            .order_by(Article.id.asc())
            .all()
        )

        total = len(articles)

        print()
        print("=" * 70)
        print("ARTICLE IMAGE BACKFILL")
        print("=" * 70)
        print(f"Articles needing images: {total}")
        print("=" * 70)
        print()

        if total == 0:
            print("All articles already have images.")
            return

        found = 0
        not_found = 0
        failed = 0

        for number, article in enumerate(
            articles,
            start=1,
        ):

            print(
                f"[{number}/{total}] "
                f"Article ID: {article.id}"
            )

            print(
                f"    Title: {article.title}"
            )

            print(
                f"    URL: {article.url}"
            )

            image_url = extract_image_url(
                article.url
            )

            if image_url:

                article.image_url = image_url

                db.session.commit()

                found += 1

                print(
                    f"    IMAGE FOUND: {image_url}"
                )

            else:

                not_found += 1

                print(
                    "    No image metadata found."
                )

            print()

            time.sleep(REQUEST_DELAY)

        print()
        print("=" * 70)
        print("IMAGE BACKFILL COMPLETE")
        print("=" * 70)
        print(f"Images found:    {found}")
        print(f"No image found:  {not_found}")
        print(f"Failed requests: {failed}")
        print(f"Processed:        {total}")
        print("=" * 70)


if __name__ == "__main__":
    main()