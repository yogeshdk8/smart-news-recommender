from app import db
from app.models import Article


def test_homepage_redirects_when_not_logged_in(client):
    response = client.get("/")

    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_register_page_loads(client):
    response = client.get("/register")

    assert response.status_code == 200


def test_user_can_register(client):
    response = client.post(
        "/register",
        data={
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "Password123",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Login" in response.data or b"login" in response.data


def test_user_can_login(authenticated_client):
    response = authenticated_client.get("/articles")

    assert response.status_code == 200


def test_articles_page_redirects_when_not_logged_in(client):
    response = client.get("/articles")

    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_feed_redirects_when_not_logged_in(client):
    response = client.get("/feed")

    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_analytics_redirects_when_not_logged_in(client):
    response = client.get("/analytics")

    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_authenticated_user_can_access_feed(
    authenticated_client,
):
    response = authenticated_client.get("/feed")

    assert response.status_code == 200


def test_authenticated_user_can_access_analytics(
    authenticated_client,
):
    response = authenticated_client.get("/analytics")

    assert response.status_code == 200