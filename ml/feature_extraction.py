"""
ml/feature_extraction.py

Database-Aligned Feature Extraction
------------------------------------

Builds a richer representation of EVERY article currently stored in
the Flask/SQLAlchemy database.

Feature groups:

1. TF-IDF from article text
2. TF-IDF from title + description/content
3. Category features
4. Recency features
5. Combined normalized article representation

IMPORTANT:

The database is now the SINGLE SOURCE OF TRUTH.

The old implementation used:

    data/processed/articles_cleaned.json

That file contained only the original 122 articles, while the live
database may contain hundreds of articles added later.

This implementation instead reads:

    Article.query.order_by(Article.id.asc()).all()

Therefore:

    Database article row
        <-> feature row
        <-> Article.id

remain aligned.

Generated artifacts are saved under:

    data/processed/

Artifacts:

    tfidf_vectorizer.joblib
    tfidf_matrix.joblib

    title_tfidf_vectorizer.joblib
    title_tfidf_matrix.joblib

    category_encoder.joblib
    category_matrix.joblib

    recency_matrix.joblib
    article_feature_matrix.joblib

    articles_index.json
    article_feature_metadata.json
"""

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, normalize

from flask import has_app_context

from app import create_app
from app.models import Article


# =====================================================================
# PATHS
# =====================================================================

MODEL_DIR = os.path.join(
    "data",
    "processed",
)

# ---------------------------------------------------------------------
# Original/compatible TF-IDF artifacts
# ---------------------------------------------------------------------

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.joblib",
)

MATRIX_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_matrix.joblib",
)

ARTICLES_INDEX_PATH = os.path.join(
    MODEL_DIR,
    "articles_index.json",
)

# ---------------------------------------------------------------------
# Improved feature artifacts
# ---------------------------------------------------------------------

TITLE_VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "title_tfidf_vectorizer.joblib",
)

TITLE_MATRIX_PATH = os.path.join(
    MODEL_DIR,
    "title_tfidf_matrix.joblib",
)

CATEGORY_ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "category_encoder.joblib",
)

CATEGORY_MATRIX_PATH = os.path.join(
    MODEL_DIR,
    "category_matrix.joblib",
)

RECENCY_MATRIX_PATH = os.path.join(
    MODEL_DIR,
    "recency_matrix.joblib",
)

FEATURE_MATRIX_PATH = os.path.join(
    MODEL_DIR,
    "article_feature_matrix.joblib",
)

FEATURE_METADATA_PATH = os.path.join(
    MODEL_DIR,
    "article_feature_metadata.json",
)


# =====================================================================
# CONFIGURATION
# =====================================================================

TEXT_MAX_FEATURES = 5000

TITLE_MAX_FEATURES = 2000

# Relative influence when building the combined article vector.

TEXT_WEIGHT = 1.0

TITLE_WEIGHT = 1.5

CATEGORY_WEIGHT = 1.0

RECENCY_WEIGHT = 1.0

# Approximate number of days for exponential recency decay.

RECENCY_DECAY_DAYS = 30.0


# =====================================================================
# GENERIC HELPERS
# =====================================================================

def safe_text(value):
    """
    Convert a value into a safe string.

    None becomes an empty string.
    """
    if value is None:
        return ""

    return str(value).strip()


def get_value(article, field, default=None):
    """
    Read a field from either:

        - a dictionary
        - a SQLAlchemy Article object

    This makes the feature-building helper functions reusable.
    """

    if isinstance(article, dict):
        return article.get(
            field,
            default,
        )

    return getattr(
        article,
        field,
        default,
    )


# =====================================================================
# DATABASE LOADING
# =====================================================================

def load_database_articles():
    """
    Load every article currently stored in the database.

    IMPORTANT:
    Articles are ordered by their real database primary key.

    Therefore feature row 0 corresponds to the lowest Article.id,
    feature row 1 corresponds to the next Article.id, and so on.

    Returns
    -------
    list
        A list of normalized dictionaries containing the database
        article information needed by the feature extractor.
    """

    if not has_app_context():
        raise RuntimeError(
            "load_database_articles() must be called inside "
            "a Flask application context."
        )

    database_articles = (
        Article.query
        .order_by(
            Article.id.asc()
        )
        .all()
    )

    records = []

    for article in database_articles:

        article_id = get_value(
            article,
            "id",
        )

        if article_id is None:
            raise ValueError(
                "An Article record has no database ID."
            )

        try:
            article_id = int(
                article_id
            )

        except (
            TypeError,
            ValueError,
        ):
            raise ValueError(
                f"Invalid Article.id value: "
                f"{article_id!r}"
            )

        title = safe_text(
            get_value(
                article,
                "title",
                "",
            )
        )

        content = safe_text(
            get_value(
                article,
                "content",
                "",
            )
        )

        # Some database schemas may have a description column.
        # Your current application may store GNews descriptions
        # in the Article.content field, so we support both.
        description = safe_text(
            get_value(
                article,
                "description",
                "",
            )
        )

        if not description:
            description = content

        source = safe_text(
            get_value(
                article,
                "source",
                "",
            )
        )

        category = safe_text(
            get_value(
                article,
                "category",
                "",
            )
        )

        url = safe_text(
            get_value(
                article,
                "url",
                "",
            )
        )

        published_at = get_value(
            article,
            "published_at",
            None,
        )

        # -------------------------------------------------------------
        # Determine the text used for the main TF-IDF representation.
        # -------------------------------------------------------------

        cleaned_text = safe_text(
            get_value(
                article,
                "cleaned_text",
                "",
            )
        )

        if cleaned_text:
            article_text = cleaned_text

        else:
            article_text = " ".join(
                part
                for part in [
                    title,
                    description,
                    content,
                ]
                if part
            ).strip()

        # If an article somehow has no textual content at all,
        # provide a small fallback so TfidfVectorizer never receives
        # an entirely empty document.
        if not article_text:
            article_text = (
                category
                or source
                or "article"
            )

        # -------------------------------------------------------------
        # Title + description representation.
        # -------------------------------------------------------------

        title_description_text = " ".join(
            part
            for part in [
                title,
                title,
                description,
            ]
            if part
        ).strip()

        if not title_description_text:
            title_description_text = (
                title
                or category
                or source
                or "article"
            )

        # -------------------------------------------------------------
        # Convert publication datetime to a JSON-safe form.
        # -------------------------------------------------------------

        published_at_serialized = None

        if isinstance(
            published_at,
            datetime,
        ):
            published_at_serialized = (
                published_at.isoformat()
            )

        elif published_at is not None:
            published_at_serialized = safe_text(
                published_at
            ) or None

        records.append(
            {
                "id": article_id,
                "title": title,
                "description": description,
                "content": content,
                "cleaned_text": article_text,
                "title_description_text": (
                    title_description_text
                ),
                "source": source,
                "category": category,
                "url": url,
                "published_at": (
                    published_at_serialized
                ),
            }
        )

    return records


# =====================================================================
# BACKWARD-COMPATIBILITY LOADER
# =====================================================================

def load_cleaned_articles():
    """
    Backward-compatible function name.

    The old implementation loaded articles from:

        data/processed/articles_cleaned.json

    The new implementation deliberately does NOT do that.

    It now loads the current database articles so that feature rows
    always correspond to the live Article IDs.
    """

    return load_database_articles()


# =====================================================================
# TEXT FEATURE EXTRACTION
# =====================================================================

def build_text_tfidf(
    articles,
    max_features=TEXT_MAX_FEATURES,
):
    """
    Build TF-IDF features from article text.

    The text comes from the current database article records.
    """

    texts = [
        safe_text(
            article.get(
                "cleaned_text"
            )
        )
        for article in articles
    ]

    # Make absolutely sure no document is completely empty.
    texts = [
        text if text else "article"
        for text in texts
    ]

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=1,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(
        texts
    )

    return vectorizer, matrix


# =====================================================================
# TITLE + DESCRIPTION FEATURE EXTRACTION
# =====================================================================

def build_title_description_tfidf(
    articles,
    max_features=TITLE_MAX_FEATURES,
):
    """
    Build TF-IDF features from:

        title + title + description

    Repeating the title gives it additional influence while still
    keeping the representation sparse.
    """

    texts = [
        safe_text(
            article.get(
                "title_description_text"
            )
        )
        for article in articles
    ]

    texts = [
        text if text else "article"
        for text in texts
    ]

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=1,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(
        texts
    )

    return vectorizer, matrix


# =====================================================================
# CATEGORY FEATURES
# =====================================================================

def build_category_features(articles):
    """
    Convert article categories into one-hot encoded features.

    Example:

        technology -> [1, 0, 0]

        sports     -> [0, 1, 0]

        business   -> [0, 0, 1]
    """

    categories = []

    for article in articles:

        category = safe_text(
            article.get(
                "category"
            )
        ).lower()

        if not category:
            category = "unknown"

        categories.append(
            category
        )

    values = np.array(
        categories,
        dtype=object,
    ).reshape(
        -1,
        1,
    )

    # Modern scikit-learn uses sparse_output=True.
    # The fallback keeps compatibility with older versions.
    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
        )

    except TypeError:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
        )

    matrix = encoder.fit_transform(
        values
    )

    return encoder, matrix


# =====================================================================
# DATE PARSING
# =====================================================================

def parse_datetime(value):
    """
    Convert a value into a timezone-aware datetime.

    Supports:

        datetime objects
        ISO-8601 strings
        common date formats

    Returns None when parsing fails.
    """

    if value is None:
        return None

    if isinstance(
        value,
        datetime,
    ):
        parsed = value

        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=timezone.utc
            )

        return parsed

    value = safe_text(
        value
    )

    if not value:
        return None

    # Common UTC suffix.
    if value.endswith(
        "Z"
    ):
        value = (
            value[:-1]
            + "+00:00"
        )

    # First attempt: ISO-8601.
    try:

        parsed = datetime.fromisoformat(
            value
        )

        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=timezone.utc
            )

        return parsed

    except ValueError:
        pass

    # Common fallback formats.
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m/%d/%Y",
    ]

    for fmt in formats:

        try:

            parsed = datetime.strptime(
                value,
                fmt,
            )

            return parsed.replace(
                tzinfo=timezone.utc
            )

        except ValueError:
            continue

    return None


# =====================================================================
# ARTICLE DATE
# =====================================================================

def get_article_date(article):
    """
    Find the article publication date.

    Multiple field names are supported so the feature extractor
    remains robust if the input structure changes.
    """

    possible_fields = [
        "published_at",
        "publishedAt",
        "published",
        "published_date",
        "publication_date",
        "date",
        "datetime",
        "created_at",
    ]

    for field in possible_fields:

        value = article.get(
            field
        )

        if value:

            parsed = parse_datetime(
                value
            )

            if parsed is not None:
                return parsed

    return None


# =====================================================================
# RECENCY SCORE
# =====================================================================

def calculate_recency_score(
    article,
    reference_time=None,
):
    """
    Calculate a recency score between approximately 0 and 1.

    Newer articles receive larger values.

    Formula:

        score = exp(-age_days / RECENCY_DECAY_DAYS)

    Articles without usable publication dates receive 0.5.
    """

    if reference_time is None:
        reference_time = datetime.now(
            timezone.utc
        )

    published = get_article_date(
        article
    )

    if published is None:
        return 0.5

    age_seconds = (
        reference_time
        - published
    ).total_seconds()

    # Protect against future publication timestamps.
    age_days = max(
        age_seconds / 86400.0,
        0.0,
    )

    score = np.exp(
        -age_days
        / RECENCY_DECAY_DAYS
    )

    return float(
        score
    )


# =====================================================================
# RECENCY MATRIX
# =====================================================================

def build_recency_features(articles):
    """
    Build a one-dimensional recency feature for every article.
    """

    reference_time = datetime.now(
        timezone.utc
    )

    values = [
        calculate_recency_score(
            article,
            reference_time=reference_time,
        )
        for article in articles
    ]

    matrix = csr_matrix(
        np.array(
            values,
            dtype=np.float32,
        ).reshape(
            -1,
            1,
        )
    )

    return matrix


# =====================================================================
# COMBINED ARTICLE FEATURE MATRIX
# =====================================================================

def build_article_feature_matrix(
    text_matrix,
    title_matrix,
    category_matrix,
    recency_matrix,
):
    """
    Combine all article feature groups into one normalized sparse
    representation.

    Groups:

        text
        title/description
        category
        recency
    """

    weighted_text = (
        text_matrix
        * TEXT_WEIGHT
    )

    weighted_title = (
        title_matrix
        * TITLE_WEIGHT
    )

    weighted_category = (
        category_matrix
        * CATEGORY_WEIGHT
    )

    weighted_recency = (
        recency_matrix
        * RECENCY_WEIGHT
    )

    combined = hstack(
        [
            weighted_text,
            weighted_title,
            weighted_category,
            weighted_recency,
        ],
        format="csr",
    )

    # Normalize every article vector.
    combined = normalize(
        combined,
        norm="l2",
        axis=1,
    )

    return combined


# =====================================================================
# VALIDATION
# =====================================================================

def validate_alignment(
    articles,
    text_matrix,
    title_matrix,
    category_matrix,
    recency_matrix,
    combined_matrix,
):
    """
    Validate that every feature matrix has exactly one row for every
    database Article record.

    Also validates that database Article IDs are explicit and unique.
    """

    article_count = len(
        articles
    )

    print(
        "\n--- Alignment validation ---"
    )

    print(
        f"Database articles: "
        f"{article_count}"
    )

    print(
        f"Text matrix rows: "
        f"{text_matrix.shape[0]}"
    )

    print(
        f"Title matrix rows: "
        f"{title_matrix.shape[0]}"
    )

    print(
        f"Category matrix rows: "
        f"{category_matrix.shape[0]}"
    )

    print(
        f"Recency matrix rows: "
        f"{recency_matrix.shape[0]}"
    )

    print(
        f"Combined matrix rows: "
        f"{combined_matrix.shape[0]}"
    )

    if (
        text_matrix.shape[0]
        != article_count
    ):
        raise RuntimeError(
            "Text TF-IDF row count does not "
            "match the database article count."
        )

    if (
        title_matrix.shape[0]
        != article_count
    ):
        raise RuntimeError(
            "Title TF-IDF row count does not "
            "match the database article count."
        )

    if (
        category_matrix.shape[0]
        != article_count
    ):
        raise RuntimeError(
            "Category matrix row count does not "
            "match the database article count."
        )

    if (
        recency_matrix.shape[0]
        != article_count
    ):
        raise RuntimeError(
            "Recency matrix row count does not "
            "match the database article count."
        )

    if (
        combined_matrix.shape[0]
        != article_count
    ):
        raise RuntimeError(
            "Combined feature matrix row count does "
            "not match the database article count."
        )

    article_ids = [
        int(
            article["id"]
        )
        for article in articles
    ]

    if len(
        article_ids
    ) != len(
        set(article_ids)
    ):
        raise RuntimeError(
            "Duplicate Article IDs detected "
            "while creating the feature catalogue."
        )

    if any(
        article_id <= 0
        for article_id in article_ids
    ):
        raise RuntimeError(
            "Invalid Article ID detected. "
            "Every Article.id must be a positive integer."
        )

    print(
        "All feature matrices have the same "
        "number of rows as the database."
    )

    print(
        "All database Article IDs are explicit "
        "and unique."
    )


# =====================================================================
# ARTICLE INDEX
# =====================================================================

def build_article_index(articles):
    """
    Build the article index used by the recommender.

    CRITICAL:

    The database Article.id is stored explicitly.

    Example:

        {
            "id": 623,
            "row": 622,
            "title": "...",
            ...
        }

    Therefore article ID -> feature row mapping is direct and reliable.
    """

    index = []

    for row_number, article in enumerate(
        articles
    ):

        article_id = int(
            article["id"]
        )

        index.append(
            {
                "id": article_id,
                "article_id": article_id,
                "row": row_number,
                "title": article.get(
                    "title"
                ),
                "description": article.get(
                    "description"
                ),
                "content": article.get(
                    "content"
                ),
                "source": article.get(
                    "source"
                ),
                "category": article.get(
                    "category"
                ),
                "url": article.get(
                    "url"
                ),
                "published_at": article.get(
                    "published_at"
                ),
            }
        )

    return index


# =====================================================================
# SAVE ARTIFACTS
# =====================================================================

def save_artifacts(
    text_vectorizer,
    text_matrix,
    title_vectorizer,
    title_matrix,
    category_encoder,
    category_matrix,
    recency_matrix,
    combined_matrix,
    articles,
):
    """
    Save all generated feature artifacts.
    """

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Existing compatible artifacts
    # -----------------------------------------------------------------

    joblib.dump(
        text_vectorizer,
        VECTORIZER_PATH,
    )

    joblib.dump(
        text_matrix,
        MATRIX_PATH,
    )

    # -----------------------------------------------------------------
    # Improved artifacts
    # -----------------------------------------------------------------

    joblib.dump(
        title_vectorizer,
        TITLE_VECTORIZER_PATH,
    )

    joblib.dump(
        title_matrix,
        TITLE_MATRIX_PATH,
    )

    joblib.dump(
        category_encoder,
        CATEGORY_ENCODER_PATH,
    )

    joblib.dump(
        category_matrix,
        CATEGORY_MATRIX_PATH,
    )

    joblib.dump(
        recency_matrix,
        RECENCY_MATRIX_PATH,
    )

    joblib.dump(
        combined_matrix,
        FEATURE_MATRIX_PATH,
    )

    # -----------------------------------------------------------------
    # Database-aligned article index
    # -----------------------------------------------------------------

    index = build_article_index(
        articles
    )

    with open(
        ARTICLES_INDEX_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            index,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # -----------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------

    article_ids = [
        int(
            article["id"]
        )
        for article in articles
    ]

    categories = sorted(
        {
            (
                safe_text(
                    article.get(
                        "category"
                    )
                ).lower()
                or "unknown"
            )
            for article in articles
        }
    )

    metadata = {
        "number_of_articles": len(
            articles
        ),
        "database_source": True,
        "database_order": "Article.id ASC",
        "article_id_alignment": (
            "articles_index.json id -> feature row"
        ),
        "explicit_article_ids": True,
        "unique_article_ids": True,
        "minimum_article_id": (
            min(article_ids)
            if article_ids
            else None
        ),
        "maximum_article_id": (
            max(article_ids)
            if article_ids
            else None
        ),
        "text_features": int(
            text_matrix.shape[1]
        ),
        "title_description_features": int(
            title_matrix.shape[1]
        ),
        "category_features": int(
            category_matrix.shape[1]
        ),
        "recency_features": int(
            recency_matrix.shape[1]
        ),
        "combined_features": int(
            combined_matrix.shape[1]
        ),
        "text_weight": TEXT_WEIGHT,
        "title_weight": TITLE_WEIGHT,
        "category_weight": CATEGORY_WEIGHT,
        "recency_weight": RECENCY_WEIGHT,
        "recency_decay_days": (
            RECENCY_DECAY_DAYS
        ),
        "feature_groups": [
            "tfidf_article_text",
            "tfidf_title_description",
            "category",
            "recency",
        ],
        "categories": categories,
    }

    with open(
        FEATURE_METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # -----------------------------------------------------------------
    # Console output
    # -----------------------------------------------------------------

    print(
        "\n--- Saved artifacts ---"
    )

    print(
        f"Saved text TF-IDF vectorizer to "
        f"{VECTORIZER_PATH}"
    )

    print(
        f"Saved text TF-IDF matrix "
        f"({text_matrix.shape[0]} articles x "
        f"{text_matrix.shape[1]} terms) to "
        f"{MATRIX_PATH}"
    )

    print(
        f"Saved title/description vectorizer to "
        f"{TITLE_VECTORIZER_PATH}"
    )

    print(
        f"Saved title/description matrix "
        f"({title_matrix.shape[0]} articles x "
        f"{title_matrix.shape[1]} terms) to "
        f"{TITLE_MATRIX_PATH}"
    )

    print(
        f"Saved category encoder to "
        f"{CATEGORY_ENCODER_PATH}"
    )

    print(
        f"Saved category matrix "
        f"({category_matrix.shape[0]} articles x "
        f"{category_matrix.shape[1]} categories) to "
        f"{CATEGORY_MATRIX_PATH}"
    )

    print(
        f"Saved recency matrix "
        f"({recency_matrix.shape[0]} articles x "
        f"{recency_matrix.shape[1]} feature) to "
        f"{RECENCY_MATRIX_PATH}"
    )

    print(
        f"Saved combined article feature matrix "
        f"({combined_matrix.shape[0]} articles x "
        f"{combined_matrix.shape[1]} features) to "
        f"{FEATURE_MATRIX_PATH}"
    )

    print(
        f"Saved database-aligned article index to "
        f"{ARTICLES_INDEX_PATH}"
    )

    print(
        f"Saved feature metadata to "
        f"{FEATURE_METADATA_PATH}"
    )


# =====================================================================
# LOAD EXISTING ARTIFACTS
# =====================================================================

def load_artifacts():
    """
    Load the original-compatible TF-IDF artifacts.
    """

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    matrix = joblib.load(
        MATRIX_PATH
    )

    with open(
        ARTICLES_INDEX_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        index = json.load(
            f
        )

    return (
        vectorizer,
        matrix,
        index,
    )


# =====================================================================
# LOAD IMPROVED ARTIFACTS
# =====================================================================

def load_improved_artifacts():
    """
    Load the complete improved feature representation.
    """

    required_paths = [
        VECTORIZER_PATH,
        MATRIX_PATH,
        TITLE_VECTORIZER_PATH,
        TITLE_MATRIX_PATH,
        CATEGORY_ENCODER_PATH,
        CATEGORY_MATRIX_PATH,
        RECENCY_MATRIX_PATH,
        FEATURE_MATRIX_PATH,
        ARTICLES_INDEX_PATH,
        FEATURE_METADATA_PATH,
    ]

    for path in required_paths:

        if not os.path.exists(
            path
        ):
            raise FileNotFoundError(
                f"Required feature artifact not found: "
                f"{path}\n\n"
                "Run:\n"
                "    python -m ml.feature_extraction\n"
                "before using the recommender."
            )

    text_vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    text_matrix = joblib.load(
        MATRIX_PATH
    )

    title_vectorizer = joblib.load(
        TITLE_VECTORIZER_PATH
    )

    title_matrix = joblib.load(
        TITLE_MATRIX_PATH
    )

    category_encoder = joblib.load(
        CATEGORY_ENCODER_PATH
    )

    category_matrix = joblib.load(
        CATEGORY_MATRIX_PATH
    )

    recency_matrix = joblib.load(
        RECENCY_MATRIX_PATH
    )

    combined_matrix = joblib.load(
        FEATURE_MATRIX_PATH
    )

    with open(
        ARTICLES_INDEX_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        index = json.load(
            f
        )

    with open(
        FEATURE_METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        metadata = json.load(
            f
        )

    return {
        "text_vectorizer": (
            text_vectorizer
        ),
        "text_matrix": (
            text_matrix
        ),
        "title_vectorizer": (
            title_vectorizer
        ),
        "title_matrix": (
            title_matrix
        ),
        "category_encoder": (
            category_encoder
        ),
        "category_matrix": (
            category_matrix
        ),
        "recency_matrix": (
            recency_matrix
        ),
        "combined_matrix": (
            combined_matrix
        ),
        "index": index,
        "metadata": metadata,
    }


# =====================================================================
# SIMILARITY HELPER
# =====================================================================

def similarity_between(
    matrix,
    i,
    j,
):
    """
    Calculate cosine similarity between two feature rows.
    """

    sim = cosine_similarity(
        matrix[i],
        matrix[j],
    )

    return float(
        sim[0][0]
    )


# =====================================================================
# FEATURE SUMMARY
# =====================================================================

def print_feature_summary(
    articles,
    text_matrix,
    title_matrix,
    category_matrix,
    recency_matrix,
    combined_matrix,
):
    """
    Print a useful summary of the generated feature representation.
    """

    print(
        "\n--- Feature summary ---"
    )

    print(
        f"Articles: "
        f"{len(articles)}"
    )

    print(
        f"TF-IDF text features: "
        f"{text_matrix.shape[1]}"
    )

    print(
        f"Title/description features: "
        f"{title_matrix.shape[1]}"
    )

    print(
        f"Category features: "
        f"{category_matrix.shape[1]}"
    )

    print(
        f"Recency features: "
        f"{recency_matrix.shape[1]}"
    )

    print(
        f"Combined feature dimensions: "
        f"{combined_matrix.shape[1]}"
    )

    categories = sorted(
        {
            (
                safe_text(
                    article.get(
                        "category"
                    )
                ).lower()
                or "unknown"
            )
            for article in articles
        }
    )

    print(
        f"Categories detected: "
        f"{len(categories)}"
    )

    print(
        f"Category list: "
        f"{', '.join(categories)}"
    )

    if articles:

        article_ids = [
            int(
                article["id"]
            )
            for article in articles
        ]

        print(
            f"Minimum Article ID: "
            f"{min(article_ids)}"
        )

        print(
            f"Maximum Article ID: "
            f"{max(article_ids)}"
        )


# =====================================================================
# SANITY-CHECK ARTICLE PAIRS
# =====================================================================

def run_sanity_check(
    articles,
    combined_matrix,
):
    """
    Compare a same-category pair against a different-category pair.

    This is only a diagnostic check.
    """

    by_category = {}

    for row_number, article in enumerate(
        articles
    ):

        category = (
            safe_text(
                article.get(
                    "category"
                )
            ).lower()
            or "unknown"
        )

        by_category.setdefault(
            category,
            [],
        ).append(
            row_number
        )

    same_category_pair = None

    for category, rows in (
        by_category.items()
    ):

        if len(rows) >= 2:

            same_category_pair = (
                rows[0],
                rows[1],
                category,
            )

            break

    categories = [
        category
        for category, rows
        in by_category.items()
        if rows
    ]

    different_category_pair = None

    if len(categories) >= 2:

        category_a = categories[0]

        category_b = categories[1]

        i = by_category[
            category_a
        ][0]

        j = by_category[
            category_b
        ][0]

        different_category_pair = (
            i,
            j,
            category_a,
            category_b,
        )

    print(
        "\n--- Improved feature sanity check ---"
    )

    same_similarity = None

    different_similarity = None

    # -----------------------------------------------------------------
    # Same category
    # -----------------------------------------------------------------

    if same_category_pair:

        i, j, category = (
            same_category_pair
        )

        same_similarity = similarity_between(
            combined_matrix,
            i,
            j,
        )

        print(
            f"Same-category pair "
            f"({category}):"
        )

        print(
            f"  '{articles[i].get('title')}'"
        )

        print(
            f"  vs "
            f"'{articles[j].get('title')}'"
        )

        print(
            f"  Combined similarity: "
            f"{same_similarity:.4f}"
        )

    else:

        print(
            "Not enough articles in one category "
            "for same-category testing."
        )

    # -----------------------------------------------------------------
    # Different category
    # -----------------------------------------------------------------

    if different_category_pair:

        (
            i,
            j,
            category_i,
            category_j,
        ) = different_category_pair

        different_similarity = similarity_between(
            combined_matrix,
            i,
            j,
        )

        print(
            f"\nDifferent-category pair "
            f"({category_i} vs {category_j}):"
        )

        print(
            f"  '{articles[i].get('title')}'"
        )

        print(
            f"  vs "
            f"'{articles[j].get('title')}'"
        )

        print(
            f"  Combined similarity: "
            f"{different_similarity:.4f}"
        )

    else:

        print(
            "Not enough categories for "
            "different-category testing."
        )

    # -----------------------------------------------------------------
    # Comparison
    # -----------------------------------------------------------------

    if (
        same_similarity is not None
        and different_similarity is not None
    ):

        print(
            "\nComparison:"
        )

        print(
            f"Same-category similarity: "
            f"{same_similarity:.4f}"
        )

        print(
            f"Different-category similarity: "
            f"{different_similarity:.4f}"
        )

        if (
            same_similarity
            > different_similarity
        ):

            print(
                "The selected same-category pair "
                "has higher similarity in this "
                "sanity check."
            )

        else:

            print(
                "The selected different-category "
                "pair scored as high or higher. "
                "Inspect the selected article pairs "
                "before drawing conclusions."
            )


# =====================================================================
# MAIN
# =====================================================================

def main():
    """
    Build the complete article feature catalogue from the LIVE
    database.

    Database -> features -> index:

        Article.id
            |
            +--> feature row
            |
            +--> articles_index.json id
    """

    app = create_app()

    with app.app_context():

        print(
            "=================================================="
        )

        print(
            "DATABASE-ALIGNED FEATURE EXTRACTION"
        )

        print(
            "=================================================="
        )

        print(
            "\nLoading articles from SQLite database..."
        )

        articles = load_database_articles()

        print(
            f"Loaded "
            f"{len(articles)} articles from the database."
        )

        if not articles:

            raise ValueError(
                "No articles were found in the database."
            )

        # -------------------------------------------------------------
        # Show database ID range.
        # -------------------------------------------------------------

        article_ids = [
            int(
                article["id"]
            )
            for article in articles
        ]

        print(
            f"Article ID range: "
            f"{min(article_ids)} "
            f"-> "
            f"{max(article_ids)}"
        )

        # -------------------------------------------------------------
        # TF-IDF text
        # -------------------------------------------------------------

        print(
            "\nBuilding TF-IDF features "
            "from database article text..."
        )

        (
            text_vectorizer,
            text_matrix,
        ) = build_text_tfidf(
            articles
        )

        print(
            f"Text TF-IDF matrix: "
            f"{text_matrix.shape}"
        )

        # -------------------------------------------------------------
        # Title + description
        # -------------------------------------------------------------

        print(
            "\nBuilding title + description features..."
        )

        (
            title_vectorizer,
            title_matrix,
        ) = build_title_description_tfidf(
            articles
        )

        print(
            f"Title/description matrix: "
            f"{title_matrix.shape}"
        )

        # -------------------------------------------------------------
        # Categories
        # -------------------------------------------------------------

        print(
            "\nBuilding category features..."
        )

        (
            category_encoder,
            category_matrix,
        ) = build_category_features(
            articles
        )

        print(
            f"Category matrix: "
            f"{category_matrix.shape}"
        )

        # -------------------------------------------------------------
        # Recency
        # -------------------------------------------------------------

        print(
            "\nBuilding recency features..."
        )

        recency_matrix = (
            build_recency_features(
                articles
            )
        )

        print(
            f"Recency matrix: "
            f"{recency_matrix.shape}"
        )

        # -------------------------------------------------------------
        # Combined representation
        # -------------------------------------------------------------

        print(
            "\nCombining article features..."
        )

        combined_matrix = (
            build_article_feature_matrix(
                text_matrix=text_matrix,
                title_matrix=title_matrix,
                category_matrix=category_matrix,
                recency_matrix=recency_matrix,
            )
        )

        print(
            f"Combined feature matrix: "
            f"{combined_matrix.shape}"
        )

        # -------------------------------------------------------------
        # Validate every matrix against DB rows.
        # -------------------------------------------------------------

        validate_alignment(
            articles=articles,
            text_matrix=text_matrix,
            title_matrix=title_matrix,
            category_matrix=category_matrix,
            recency_matrix=recency_matrix,
            combined_matrix=combined_matrix,
        )

        # -------------------------------------------------------------
        # Save everything.
        # -------------------------------------------------------------

        save_artifacts(
            text_vectorizer=text_vectorizer,
            text_matrix=text_matrix,
            title_vectorizer=title_vectorizer,
            title_matrix=title_matrix,
            category_encoder=category_encoder,
            category_matrix=category_matrix,
            recency_matrix=recency_matrix,
            combined_matrix=combined_matrix,
            articles=articles,
        )

        # -------------------------------------------------------------
        # Print summary.
        # -------------------------------------------------------------

        print_feature_summary(
            articles=articles,
            text_matrix=text_matrix,
            title_matrix=title_matrix,
            category_matrix=category_matrix,
            recency_matrix=recency_matrix,
            combined_matrix=combined_matrix,
        )

        # -------------------------------------------------------------
        # Sanity check.
        # -------------------------------------------------------------

        run_sanity_check(
            articles=articles,
            combined_matrix=combined_matrix,
        )

        # -------------------------------------------------------------
        # Final verification.
        # -------------------------------------------------------------

        print(
            "\n--- Final database/index alignment check ---"
        )

        print(
            f"Database articles: "
            f"{len(articles)}"
        )

        print(
            f"Feature rows: "
            f"{combined_matrix.shape[0]}"
        )

        index = build_article_index(
            articles
        )

        print(
            f"Index rows: "
            f"{len(index)}"
        )

        explicit_ids = [
            item.get("id")
            for item in index
            if item.get("id") is not None
        ]

        print(
            f"Explicit IDs in index: "
            f"{len(explicit_ids)}"
        )

        missing_ids = sorted(
            set(article_ids)
            - {
                int(article_id)
                for article_id
                in explicit_ids
            }
        )

        print(
            f"Database IDs missing from index: "
            f"{len(missing_ids)}"
        )

        if missing_ids:

            print(
                f"Example missing IDs: "
                f"{missing_ids[:20]}"
            )

            raise RuntimeError(
                "Some database Article IDs are missing "
                "from articles_index.json."
            )

        print(
            "\nSUCCESS:"
        )

        print(
            "Every database article has a corresponding "
            "feature row and explicit Article ID."
        )

        print(
            "\nFeature extraction/build completed successfully."
        )


# =====================================================================
# SCRIPT ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()