import pytest
from httpx import AsyncClient, ASGITransport
from langchain_core.documents import Document as LC_Document
from unittest.mock import MagicMock, patch
from app.main import app
from app.services.pdf_service import process_pdf_task, embeddings_model
import io

from langchain_text_splitters import RecursiveCharacterTextSplitter

@pytest.mark.asyncio
async def test_ask_question_no_auth():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        response = await ac.post("/ask", json={"question": "Привет"})

    assert response.status_code == 422

@pytest.mark.asyncio
async def test_get_documents_empty():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        response = await ac.get("/documents", params={"user_id": "999"})
    
    assert response.status_code == 200
    assert response.json() == []

@pytest.mark.asyncio
async def test_upload_pdf_success():
    file_content = b"%PDF-1.4 test content"
    file = io.BytesIO(file_content)

    with patch("app.services.pdf_service.process_pdf_task.delay") as mock_task:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            response = await ac.post(
                "/upload",
                params={"user_id": "999"},
                files={"file": ("test.pdf", file, "application/pdf")}
            )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert mock_task.called

@pytest.mark.asyncio
async def test_upload_wrong_file_type():
    file_content = b"Just some text content"
    file = io.BytesIO(file_content)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        response = await ac.post(
            "/upload",
            params={"user_id": "test_user"},
            files={"file": ("test.txt", file, "text/plain")}
        )

    assert response.status_code != 200

@pytest.mark.asyncio
async def test_ask_question_no_user_id():
    text = "Test string " * 200
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    chunks = splitter.split_text(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= 500 for chunk in chunks)

@pytest.mark.asyncio
async def test_delete_document_permissions():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        response = await ac.delete("/documents/30", params={"user_id": "999"})
    
    assert response.status_code == 404

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

            result = pdf_service.process_pdf_task("fake.pdf", "test.pdf", user_id="guest")

            assert result is not None
            assert result["status"] == "completed"
            assert result["document_id"] == 1
            
        finally:
            pdf_service.embeddings_model = original_embeddings

@pytest.mark.asyncio
async def test_get_answer_success():
    mock_db = MagicMock()
    mock_db.execute.return_value = [("Context chunk",)]

    from app.services import ask_service
    
    original_embeddings = ask_service.embeddings_model
    original_llm = ask_service.llm

    mock_emb_inst = MagicMock()
    mock_emb_inst.embed_query.return_value = [0.1, 0.2, 0.3]
    
    mock_llm_inst = MagicMock()
    mock_llm_inst.invoke.return_value = MagicMock(content="Answer based on context")

    ask_service.embeddings_model = mock_emb_inst
    ask_service.llm = mock_llm_inst

    try:
        answer = await ask_service.get_answer(mock_db, "Test question", user_id="999")

        assert "Answer based on context" in answer
        assert mock_llm_inst.invoke.called
        
    finally:
        ask_service.embeddings_model = original_embeddings
        ask_service.llm = original_llm
