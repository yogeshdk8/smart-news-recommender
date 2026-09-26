import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import pytest

from app import create_app, db
from app.models import User


@pytest.fixture
def app():
    """Create a Flask application using an isolated test database."""

    app = create_app(
        {
            "TESTING": True,
            "WTF_CSRF_ENABLED": False,
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        }
    )

    with app.app_context():
        db.create_all()

    yield app

    with app.app_context():
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    """Return a Flask test client."""

    return app.test_client()


@pytest.fixture
def authenticated_client(app, client):
    """Return a test client with a registered and logged-in user."""

    with app.app_context():
        user = User(
            username="testuser",
            email="test@example.com",
        )

        user.set_password("TestPassword123")

        db.session.add(user)
        db.session.commit()

    response = client.post(
        "/login",
        data={
            "username": "testuser",
            "password": "TestPassword123",
        },
        follow_redirects=False,
    )

    assert response.status_code == 302
    assert "/articles" in response.headers["Location"]

    return client