from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from langchain_core.documents import Document as LC_Document
from sqlalchemy import func, select

from app.core.database import SessionLocal
from app.models.sql_models import Document, DocumentChunk, User
from app.services import pdf_service


def vector_for(text):
    """A 1024-dim vector that encodes which chunk it belongs to, so pairing mistakes are visible."""
    return [float(text.split()[-1])] + [0.0] * 1023


class FakeEmbedder:
    def __init__(self, drop_last=False, fail_on_call=None):
        self.calls = []
        self.drop_last = drop_last
        self.fail_on_call = fail_on_call

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        if len(self.calls) == self.fail_on_call:
            raise ConnectionError("Ollama went away")
        vectors = [vector_for(text) for text in texts]
        return vectors[:-1] if self.drop_last else vectors

    def embed_query(self, text):
        raise AssertionError("ingestion must not embed chunks one at a time")


@pytest.fixture
def pdf_pages(tmp_path):
    """A placeholder file plus PyPDFLoader patched to return one short page (= one chunk) per text."""
    def make(texts):
        path = tmp_path / "upload.pdf"
        path.write_bytes(b"%PDF-1.4")
        mock_loader = patch.object(pdf_service, "PyPDFLoader").start()
        mock_loader.return_value.load.return_value = [LC_Document(page_content=t) for t in texts]
        return str(path)
    yield make
    patch.stopall()


@pytest.fixture
def fake_db(monkeypatch):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None
    db.commit.side_effect = lambda: setattr(db.add.call_args.args[0], "id", 1)
    monkeypatch.setattr(pdf_service, "SessionLocal", lambda: db)
    return db


@pytest.fixture
def real_user():
    with SessionLocal() as db:
        user = User(email=f"{uuid4().hex}@example.com", hashed_password="x")
        db.add(user)
        db.commit()
        return user.id


def document_count(user_id):
    with SessionLocal() as db:
        return db.scalar(select(func.count()).select_from(Document).where(Document.user_id == user_id))


def test_chunks_are_embedded_in_batches(pdf_pages, fake_db, monkeypatch):
    path = pdf_pages([f"chunk {i}" for i in range(5)])
    embedder = FakeEmbedder()
    monkeypatch.setattr(pdf_service, "embeddings_model", embedder)
    monkeypatch.setattr(pdf_service, "EMBED_BATCH_SIZE", 2)

    result = pdf_service.process_pdf_task(path, "test.pdf", user_id=1)

    assert result == {"document_id": 1, "status": "completed"}
    assert [len(call) for call in embedder.calls] == [2, 2, 1]
    stored = fake_db.add.call_args.args[0].chunks
    assert [(c.content, c.embedding[0]) for c in stored] == [(f"chunk {i}", float(i)) for i in range(5)]


def test_document_and_chunks_are_committed_once_after_embedding(pdf_pages, fake_db, monkeypatch):
    path = pdf_pages([f"chunk {i}" for i in range(5)])
    events = []
    embedder = FakeEmbedder()
    original = embedder.embed_documents
    embedder.embed_documents = lambda texts: events.append("embed") or original(texts)
    fake_db.add.side_effect = lambda doc: events.append("add")
    fake_db.commit.side_effect = lambda: events.append("commit") or setattr(fake_db.add.call_args.args[0], "id", 1)
    monkeypatch.setattr(pdf_service, "embeddings_model", embedder)
    monkeypatch.setattr(pdf_service, "EMBED_BATCH_SIZE", 2)

    pdf_service.process_pdf_task(path, "test.pdf", user_id=1)

    assert events == ["embed", "embed", "embed", "add", "commit"]


def test_short_embedding_reply_fails_instead_of_mispairing(pdf_pages, fake_db, monkeypatch):
    path = pdf_pages(["chunk 0", "chunk 1"])
    monkeypatch.setattr(pdf_service, "embeddings_model", FakeEmbedder(drop_last=True))

    with pytest.raises(ValueError):
        pdf_service.process_pdf_task(path, "test.pdf", user_id=1)

    assert not fake_db.add.called and not fake_db.commit.called


def test_pdf_without_text_creates_no_document(pdf_pages, fake_db, monkeypatch):
    path = pdf_pages([])
    embedder = FakeEmbedder()
    monkeypatch.setattr(pdf_service, "embeddings_model", embedder)

    result = pdf_service.process_pdf_task(path, "scan.pdf", user_id=1)

    assert result == {"document_id": None, "status": "no_text"}
    assert embedder.calls == []
    assert not fake_db.add.called and not fake_db.commit.called


@pytest.mark.db
def test_batched_chunks_are_stored_in_order(pdf_pages, real_user, monkeypatch):
    path = pdf_pages([f"chunk {i}" for i in range(7)])
    monkeypatch.setattr(pdf_service, "embeddings_model", FakeEmbedder())
    monkeypatch.setattr(pdf_service, "EMBED_BATCH_SIZE", 3)

    result = pdf_service.process_pdf_task(path, "seven.pdf", user_id=real_user)

    with SessionLocal() as db:
        rows = db.execute(
            select(DocumentChunk.content, DocumentChunk.embedding)
            .where(DocumentChunk.document_id == result["document_id"])
            .order_by(DocumentChunk.id)
        ).all()
        assert db.get(Document, result["document_id"]).user_id == real_user
    assert [(content, embedding[0]) for content, embedding in rows] == [(f"chunk {i}", float(i)) for i in range(7)]


@pytest.mark.db
def test_embedding_failure_midway_leaves_no_document(pdf_pages, real_user, monkeypatch):
    path = pdf_pages([f"chunk {i}" for i in range(7)])
    monkeypatch.setattr(pdf_service, "embeddings_model", FakeEmbedder(fail_on_call=2))
    monkeypatch.setattr(pdf_service, "EMBED_BATCH_SIZE", 3)

    with pytest.raises(ConnectionError):
        pdf_service.process_pdf_task(path, "half.pdf", user_id=real_user)

    assert document_count(real_user) == 0


@pytest.mark.db
def test_reupload_after_a_failure_is_processed(pdf_pages, real_user, monkeypatch):
    monkeypatch.setattr(pdf_service, "EMBED_BATCH_SIZE", 3)
    monkeypatch.setattr(pdf_service, "embeddings_model", FakeEmbedder(fail_on_call=2))
    with pytest.raises(ConnectionError):
        pdf_service.process_pdf_task(pdf_pages([f"chunk {i}" for i in range(7)]), "retry.pdf", user_id=real_user)

    # Previously the empty document left by the failure made the duplicate check skip this for good.
    monkeypatch.setattr(pdf_service, "embeddings_model", FakeEmbedder())
    result = pdf_service.process_pdf_task(pdf_pages([f"chunk {i}" for i in range(7)]), "retry.pdf", user_id=real_user)

    assert result["status"] == "completed"
    assert document_count(real_user) == 1
