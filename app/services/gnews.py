"""
app/services/gnews.py

GNews API integration for the Smart News Recommender.
"""

import requests
from flask import current_app


GNEWS_SEARCH_URL = "https://gnews.io/api/v4/search"


class GNewsError(Exception):
    """Raised when the GNews API request fails."""


def search_news(query, max_articles=10, page=1):
    """
    Search GNews for articles matching the user's query.

    Parameters:
        query: User's search text.
        max_articles: Number of articles to request.
        page: GNews result page number.

    Returns:
        List of GNews article dictionaries.

    Raises:
        GNewsError: If the API key is missing or GNews fails.
    """

    query = (query or "").strip()

    if not query:
        return []

    api_key = current_app.config.get("GNEWS_API_KEY")

    if not api_key:
        raise GNewsError(
            "GNEWS_API_KEY is not configured."
        )

    # Make sure page is always a valid positive integer.
    try:
        page = int(page)
    except (TypeError, ValueError):
        page = 1

    page = max(page, 1)

    # GNews allows a maximum of 10 articles per request.
    try:
        max_articles = int(max_articles)
    except (TypeError, ValueError):
        max_articles = 10

    max_articles = max(1, min(max_articles, 10))

    params = {
        "q": query,
        "lang": "en",
        "sortby": "publishedAt",
        "max": max_articles,
        "page": page,
    }

    headers = {
        "X-Api-Key": api_key,
    }

    try:
        response = requests.get(
            GNEWS_SEARCH_URL,
            params=params,
            headers=headers,
            timeout=15,
        )

    except requests.RequestException as exc:
        raise GNewsError(
            f"Unable to connect to GNews: {exc}"
        ) from exc

    if response.status_code != 200:
        try:
            error_data = response.json()
        except ValueError:
            error_data = {}

        error_message = error_data.get(
            "errors",
            "GNews API request failed."
        )

        raise GNewsError(
            f"GNews returned HTTP "
            f"{response.status_code}: "
            f"{error_message}"
        )

    try:
        data = response.json()

    except ValueError as exc:
        raise GNewsError(
            "GNews returned an invalid JSON response."
        ) from exc

    return data.get("articles", [])