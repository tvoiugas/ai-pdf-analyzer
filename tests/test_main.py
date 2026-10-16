import asyncio
import time
import pytest
from httpx import AsyncClient, ASGITransport
from langchain_core.documents import Document as LC_Document
from unittest.mock import AsyncMock, MagicMock, patch
from app.main import app
from app.core.database import AsyncSessionLocal, get_db
from app.models.sql_models import Document
from app.services.pdf_service import process_pdf_task, embeddings_model
import io

from langchain_text_splitters import RecursiveCharacterTextSplitter

@pytest.mark.asyncio
async def test_ask_question_no_auth():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        response = await ac.post("/ask", json={"question": "Привет"})

    assert response.status_code == 401

@pytest.mark.db
@pytest.mark.asyncio
async def test_get_documents_empty(client, register):
    _, headers = await register()
    response = await client.get("/documents", headers=headers)
    
    assert response.status_code == 200
    assert response.json() == []

@pytest.mark.db
@pytest.mark.asyncio
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

@pytest.mark.asyncio
async def test_upload_pdf_success(as_user, upload_dir):
    as_user(999)
    file_content = b"%PDF-1.4 test content"
    file = io.BytesIO(file_content)

    with patch("app.services.pdf_service.process_pdf_task.delay") as mock_task:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            response = await ac.post(
                "/upload",
                files={"file": ("test.pdf", file, "application/pdf")}
            )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    # The owner passed to the worker comes from the token, not from the request.
    assert mock_task.call_args.args[1:] == ("test.pdf", 999)

@pytest.mark.asyncio
async def test_upload_wrong_file_type(as_user):
    as_user(999)
    file_content = b"Just some text content"
    file = io.BytesIO(file_content)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        response = await ac.post(
            "/upload",
            files={"file": ("test.txt", file, "text/plain")}
        )

    assert response.status_code == 400

@pytest.mark.asyncio
async def test_ask_question_no_user_id():
    text = "Test string " * 200
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    chunks = splitter.split_text(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= 500 for chunk in chunks)

@pytest.mark.db
@pytest.mark.asyncio
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

@pytest.mark.asyncio
async def test_process_pdf_task_logic():
    with patch("app.services.pdf_service.PyPDFLoader") as mock_loader, \
         patch("app.services.pdf_service.SessionLocal") as mock_db_session, \
         patch("app.services.pdf_service.os.path.exists") as mock_exists, \
         patch("app.services.pdf_service.os.remove") as mock_remove:

        mock_exists.return_value = True
        
        mock_loader.return_value.load.return_value = [
            LC_Document(page_content="Test content for embeddings", metadata={"source": "test.pdf"})
        ]
        
        from app.services import pdf_service
        original_embeddings = pdf_service.embeddings_model
        mock_emb_inst = MagicMock()
        mock_emb_inst.embed_query.return_value = [0.1, 0.2, 0.3]
        pdf_service.embeddings_model = mock_emb_inst

        try:
            mock_db = MagicMock()
            mock_db_session.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = None
            
            def mock_add(obj):
                if hasattr(obj, 'filename'):
                    obj.id = 1
            mock_db.add.side_effect = mock_add

            result = pdf_service.process_pdf_task("fake.pdf", "test.pdf", user_id=1)

            assert result is not None
            assert result["status"] == "completed"
            assert result["document_id"] == 1
            
        finally:
            pdf_service.embeddings_model = original_embeddings

@pytest.mark.asyncio
async def test_get_answer_success():
    mock_db = AsyncMock()
    mock_db.execute.return_value = [("Context chunk",)]

    from app.services import ask_service
    
    original_embeddings = ask_service.embeddings_model
    original_llm = ask_service.llm

    mock_emb_inst = AsyncMock()
    mock_emb_inst.aembed_query.return_value = [0.1, 0.2, 0.3]
    
    mock_llm_inst = AsyncMock()
    mock_llm_inst.ainvoke.return_value = MagicMock(content="Answer based on context")

    ask_service.embeddings_model = mock_emb_inst
    ask_service.llm = mock_llm_inst

    try:
        answer = await ask_service.get_answer(mock_db, "Test question", user_id=999)

        assert "Answer based on context" in answer
        assert mock_llm_inst.ainvoke.called
        
    finally:
        ask_service.embeddings_model = original_embeddings
        ask_service.llm = original_llm


class SlowFakeLLM:
    """Takes `delay` seconds to answer: the sync path blocks the event loop, the async path yields it."""

    def __init__(self, delay):
        self.delay = delay

    def invoke(self, prompt):
        time.sleep(self.delay)
        return MagicMock(content="slow answer")

    async def ainvoke(self, prompt):
        await asyncio.sleep(self.delay)
        return MagicMock(content="slow answer")


@pytest.mark.asyncio
async def test_slow_ask_does_not_block_other_requests(monkeypatch, as_user):
    as_user(1)
    from app.services import ask_service

    mock_emb_inst = AsyncMock()
    mock_emb_inst.aembed_query.return_value = [0.1] * 1024
    monkeypatch.setattr(ask_service, "embeddings_model", mock_emb_inst)
    monkeypatch.setattr(ask_service, "llm", SlowFakeLLM(delay=1.0))

    mock_db = AsyncMock()
    mock_db.execute.return_value = [("Context chunk",)]

    async def override_get_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            started = time.perf_counter()
            ask = asyncio.create_task(ac.post("/ask", json={"question": "q"}))
            await asyncio.sleep(0.1)  # let /ask reach the LLM call
            other = await ac.get("/openapi.json")
            other_elapsed = time.perf_counter() - started
            ask_response = await ask
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert other.status_code == 200
    assert other_elapsed < 0.5, f"/openapi.json waited {other_elapsed:.2f}s behind /ask"
    assert ask_response.json()["answer"] == "slow answer"
