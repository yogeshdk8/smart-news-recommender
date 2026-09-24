import re
import html
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Load NLTK resources
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text):
    """
    Clean and normalize article text.

    Steps:
    1. Convert HTML entities
    2. Remove HTML tags
    3. Remove URLs
    4. Convert to lowercase
    5. Tokenize
    6. Remove punctuation
    7. Remove stopwords
    8. Lemmatize words
    """

    if not text:
        return ""

    # Convert HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Keep alphabetic words and remove stopwords
    tokens = [
        word
        for word in tokens
        if word.isalpha() and word not in STOP_WORDS
    ]

    # Lemmatize
    tokens = [
        LEMMATIZER.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)


def clean_article(article):
    """
    Clean the useful text fields of one news article.
    """

    title = article.get("title", "")
    description = article.get("description", "")
    content = article.get("content", "")

    cleaned_title = clean_text(title)
    cleaned_description = clean_text(description)
    cleaned_content = clean_text(content)

    # Combine cleaned fields
    cleaned_text = " ".join(
        part
        for part in [
            cleaned_title,
            cleaned_description,
            cleaned_content
        ]
        if part
    )

    cleaned_article = article.copy()

    cleaned_article["cleaned_title"] = cleaned_title
    cleaned_article["cleaned_description"] = cleaned_description
    cleaned_article["cleaned_content"] = cleaned_content
    cleaned_article["cleaned_text"] = cleaned_text

    return cleaned_article


if __name__ == "__main__":

    sample = """
    <p>News websites are publishing new stories every day!</p>
    Visit https://example.com for more information.
    The users are reading and sharing these stories.
    """

    print("Original text:")
    print(sample)

    print("\nCleaned text:")
    print(clean_text(sample))