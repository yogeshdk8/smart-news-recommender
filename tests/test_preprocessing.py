from ml.preprocessing import clean_text, clean_article


def test_clean_text_lowercase():
    result = clean_text("HELLO WORLD")

    assert result == "hello world"


def test_clean_text_removes_html():
    result = clean_text("<p>Hello world</p>")

    assert "<p>" not in result
    assert "</p>" not in result


def test_clean_text_removes_url():
    result = clean_text("Visit https://example.com for news")

    assert "https" not in result
    assert "example" not in result


def test_clean_text_removes_stopwords():
    result = clean_text("the news is very interesting")

    assert "the" not in result.split()
    assert "is" not in result.split()


def test_clean_text_lemmatizes():
    result = clean_text("The cats are running")

    assert "cat" in result
    assert "cats" not in result
    assert "running" not in result


def test_clean_article_creates_cleaned_text():
    article = {
        "title": "New Technology News",
        "description": "The technology companies are growing.",
        "content": "Companies are developing new products."
    }

    result = clean_article(article)

    assert "cleaned_title" in result
    assert "cleaned_description" in result
    assert "cleaned_content" in result
    assert "cleaned_text" in result

    assert result["cleaned_text"] != ""