import os
from types import SimpleNamespace
from uuid import uuid4

# Must run before app modules are imported: security.py refuses to start without a strong secret.
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-at-least-32-characters-long")

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.core.security import get_current_user

PASSWORD = "correct-horse-battery"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest.fixture
def as_user():
    """Authenticate requests as a fake user without touching the database."""
    def login(user_id=999):
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=user_id, email="fake@example.com")
    yield login
    app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
def register(client):
    """Create a real user through the API; returns (user id, auth headers)."""
    async def register_and_login(email=None):
        email = email or f"{uuid4().hex}@example.com"
        response = await client.post("/auth/register", json={"email": email, "password": PASSWORD})
        assert response.status_code == 201, response.text
        token = await client.post("/auth/token", data={"username": email, "password": PASSWORD})
        assert token.status_code == 200, token.text
        return response.json()["id"], {"Authorization": f"Bearer {token.json()['access_token']}"}
    return register_and_login
