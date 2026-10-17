import asyncio
import os
import selectors
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from httpx import AsyncClient, ASGITransport

from tests.fakes import FakeEmbedder, FakeLLM

ROOT = Path(__file__).resolve().parent.parent
PASSWORD = "correct-horse-battery"
PGVECTOR_IMAGE = "pgvector/pgvector:0.8.7-pg16"
UNREACHABLE_DATABASE_URL = "postgresql://unused:unused@127.0.0.1:1/unused"


def pytest_configure(config):
    """Point the app at a test database before any app module is imported.

    App modules read DATABASE_URL at import time, and test modules are only imported after this hook.
    Tests never use DATABASE_URL from the environment or .env: it may be a development database,
    and every db test empties the tables afterwards. They use TEST_DATABASE_URL, or a throwaway
    pgvector container started here.
    """
    os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-at-least-32-characters-long")
    config.db_unavailable_reason = None

    url = os.environ.get("TEST_DATABASE_URL")
    db_tests_selected = "not db" not in (config.option.markexpr or "")
    if not url and db_tests_selected:
        try:
            from testcontainers.community.postgres import PostgresContainer

            container = PostgresContainer(PGVECTOR_IMAGE, driver=None).start()
        except Exception as e:  # Docker missing or not running
            config.db_unavailable_reason = f"no test database: set TEST_DATABASE_URL or start Docker ({type(e).__name__})"
        else:
            config.pgvector_container = container
            url = container.get_connection_url()

    os.environ["DATABASE_URL"] = url or UNREACHABLE_DATABASE_URL
    if url:
        # The real migrations build the schema, so a broken migration fails the whole run.
        from alembic import command
        from alembic.config import Config

        command.upgrade(Config(str(ROOT / "alembic.ini")), "head")


def pytest_unconfigure(config):
    container = getattr(config, "pgvector_container", None)
    if container is not None:
        container.stop()


def pytest_collection_modifyitems(config, items):
    if config.db_unavailable_reason:
        skip = pytest.mark.skip(reason=config.db_unavailable_reason)
        for item in items:
            if "db" in item.keywords:
                item.add_marker(skip)


def pytest_asyncio_loop_factories(config, item):
    if sys.platform == "win32":
        # psycopg's async mode can't run on the Proactor loop Windows uses by default.
        return {"selector": lambda: asyncio.SelectorEventLoop(selectors.SelectSelector())}
    return {"default": asyncio.EventLoop}


@pytest.fixture(autouse=True)
def empty_tables_after_db_tests(request):
    yield
    if request.node.get_closest_marker("db"):
        from sqlalchemy import text
        from app.core.database import engine

        with engine.begin() as conn:
            conn.execute(text("TRUNCATE users, documents, document_chunks RESTART IDENTITY CASCADE"))


@pytest.fixture
async def client():
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    """Save uploads into a per-test temporary folder instead of /app/data."""
    from app import main as main_module

    monkeypatch.setattr(main_module, "SAVE_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture
def as_user():
    """Authenticate requests as a fake user without touching the database."""
    from app.main import app
    from app.core.security import get_current_user

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


@pytest.fixture
def fake_ai(monkeypatch):
    """Replace Ollama in both services: no network, deterministic 1024-dim vectors, recorded prompts."""
    from app.services import ask_service, pdf_service

    ai = SimpleNamespace(embedder=FakeEmbedder(), llm=FakeLLM())
    monkeypatch.setattr(ask_service, "embeddings_model", ai.embedder)
    monkeypatch.setattr(pdf_service, "embeddings_model", ai.embedder)
    monkeypatch.setattr(ask_service, "llm", ai.llm)
    return ai
