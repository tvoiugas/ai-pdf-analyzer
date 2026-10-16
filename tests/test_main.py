import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.main import app
from app.core.database import AsyncSessionLocal, get_db
from app.models.sql_models import Document
from app.services import ask_service


async def test_ask_question_no_auth(client):
    response = await client.post("/ask", json={"question": "Привет"})

    assert response.status_code == 401


@pytest.mark.db
async def test_get_documents_empty(client, register):
    _, headers = await register()
    response = await client.get("/documents", headers=headers)

    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.db
async def test_get_documents_filters_by_filename(client, register):
    owner_id, headers = await register()
    other_id, _ = await register()
    async with AsyncSessionLocal() as db:
        db.add_all([
            Document(filename="report-2025.pdf", user_id=owner_id),
            Document(filename="Q3_REPORT.pdf", user_id=owner_id),
            Document(filename="notes.pdf", user_id=owner_id),
            Document(filename="report-of-someone-else.pdf", user_id=other_id),
        ])
        await db.commit()

    async def filenames(**params):
        response = await client.get("/documents", params=params, headers=headers)
        assert response.status_code == 200
        return sorted(d["filename"] for d in response.json())

    assert await filenames(filename="report") == ["Q3_REPORT.pdf", "report-2025.pdf"]
    assert await filenames(filename="_") == ["Q3_REPORT.pdf"]
    assert await filenames(filename="%") == []
    assert await filenames() == ["Q3_REPORT.pdf", "notes.pdf", "report-2025.pdf"]


async def test_upload_pdf_success(client, as_user, upload_dir):
    as_user(999)
    with patch("app.services.pdf_service.process_pdf_task.delay") as mock_task:
        response = await client.post("/upload", files={"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")})

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    # The owner passed to the worker comes from the token, not from the request.
    assert mock_task.call_args.args[1:] == ("test.pdf", 999)


async def test_upload_wrong_file_type(client, as_user):
    as_user(999)
    response = await client.post("/upload", files={"file": ("test.txt", b"Just some text content", "text/plain")})

    assert response.status_code == 400


@pytest.mark.db
async def test_delete_document_permissions(client, register):
    owner_id, owner_headers = await register()
    _, other_headers = await register()
    async with AsyncSessionLocal() as db:
        doc = Document(filename="owned.pdf", user_id=owner_id)
        db.add(doc)
        await db.commit()

    # Another user gets the same 404 as for a missing id, so ids of others' documents aren't revealed.
    response = await client.delete(f"/documents/{doc.id}", headers=other_headers)
    assert response.status_code == 404

    response = await client.delete(f"/documents/{doc.id}", headers=owner_headers)
    assert response.status_code == 200


async def test_get_answer_success(fake_ai):
    db = AsyncMock()
    db.execute.return_value = [("Context chunk",)]

    answer = await ask_service.get_answer(db, "Test question", user_id=999)

    assert answer == "fake answer"
    assert "Context chunk" in fake_ai.llm.last_user_message


class SlowFakeLLM:
    """Takes `delay` seconds to answer: the sync path blocks the event loop, the async path yields it."""

    def __init__(self, delay):
        self.delay = delay

    def invoke(self, prompt):
        time.sleep(self.delay)
        return SimpleNamespace(content="slow answer")

    async def ainvoke(self, prompt):
        await asyncio.sleep(self.delay)
        return SimpleNamespace(content="slow answer")


async def test_slow_ask_does_not_block_other_requests(client, fake_ai, monkeypatch, as_user):
    as_user(1)
    monkeypatch.setattr(ask_service, "llm", SlowFakeLLM(delay=1.0))
    db = AsyncMock()
    db.execute.return_value = [("Context chunk",)]

    async def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    try:
        started = time.perf_counter()
        ask = asyncio.create_task(client.post("/ask", json={"question": "q"}))
        await asyncio.sleep(0.1)  # let /ask reach the LLM call
        other = await client.get("/openapi.json")
        other_elapsed = time.perf_counter() - started
        ask_response = await ask
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert other.status_code == 200
    assert other_elapsed < 0.5, f"/openapi.json waited {other_elapsed:.2f}s behind /ask"
    assert ask_response.json()["answer"] == "slow answer"
