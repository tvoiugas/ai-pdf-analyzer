import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from app.core.database import AsyncSessionLocal
from app.models.sql_models import Document, DocumentChunk, User
from app.services import pdf_service
from tests.fakes import fake_vector
from tests.pdf_factory import make_pdf

ROOT = Path(__file__).resolve().parent.parent


async def chunk_count(**filters):
    async with AsyncSessionLocal() as db:
        query = select(func.count()).select_from(DocumentChunk)
        for column, value in filters.items():
            query = query.where(getattr(DocumentChunk, column) == value)
        return await db.scalar(query)


def run_worker_inline():
    """Make /upload run the Celery task in-process instead of sending it to Redis."""
    return patch.object(pdf_service.process_pdf_task, "delay", side_effect=pdf_service.process_pdf_task)


# --- The whole pipeline through the API ---

@pytest.mark.db
async def test_upload_ingest_list_ask_delete(client, register, fake_ai, upload_dir):
    _, headers = await register()
    pdf = make_pdf([
        "Either party may terminate this agreement with ninety days written notice.",
        "Invoices are payable within thirty days of receipt.",
    ])

    with run_worker_inline():
        upload = await client.post("/upload", files={"file": ("Supply Agreement.pdf", pdf, "application/pdf")}, headers=headers)
    assert upload.status_code == 200

    documents = (await client.get("/documents", headers=headers)).json()
    assert [d["filename"] for d in documents] == ["Supply Agreement.pdf"]
    assert set(documents[0]) == {"id", "filename", "upload_date"}
    doc_id = documents[0]["id"]
    # The real loader and splitter ran: both pages were embedded in one batch and stored.
    assert len(fake_ai.embedder.documents) == 1
    assert await chunk_count(document_id=doc_id) == 2
    assert list(upload_dir.iterdir()) == []

    answer = await client.post("/ask", json={"question": "How much notice is needed?"}, headers=headers)
    assert answer.json() == {"question": "How much notice is needed?", "answer": "fake answer"}
    assert "ninety days written notice" in fake_ai.llm.last_user_message
    assert fake_ai.embedder.queries == ["Represent this sentence for searching relevant passages: How much notice is needed?"]

    deleted = await client.delete(f"/documents/{doc_id}", headers=headers)
    assert deleted.status_code == 200
    assert (await client.get("/documents", headers=headers)).json() == []
    assert await chunk_count(document_id=doc_id) == 0


@pytest.mark.db
async def test_uploading_the_same_filename_again_keeps_one_document(client, register, fake_ai, upload_dir):
    _, headers = await register()
    pdf = make_pdf(["Version one of the contract."])

    with run_worker_inline():
        for _ in range(2):
            response = await client.post("/upload", files={"file": ("contract.pdf", pdf, "application/pdf")}, headers=headers)
            assert response.status_code == 200

    assert len((await client.get("/documents", headers=headers)).json()) == 1
    assert len(fake_ai.embedder.documents) == 1


@pytest.mark.db
async def test_ask_can_be_limited_to_one_of_the_users_documents(client, register, fake_ai):
    user_id, headers = await register()
    async with AsyncSessionLocal() as db:
        lease = Document(filename="lease.pdf", user_id=user_id)
        lease.chunks = [DocumentChunk(content="Rent is due on the fifth.", embedding=fake_vector("lease"))]
        invoice = Document(filename="invoice.pdf", user_id=user_id)
        invoice.chunks = [DocumentChunk(content="Total due: 1200 KZT.", embedding=fake_vector("invoice"))]
        db.add_all([lease, invoice])
        await db.commit()

    await client.post("/ask", json={"question": "When is rent due?", "document_id": lease.id}, headers=headers)
    assert "Rent is due" in fake_ai.llm.last_user_message
    assert "1200 KZT" not in fake_ai.llm.last_user_message

    await client.post("/ask", json={"question": "When is rent due?"}, headers=headers)
    assert "Rent is due" in fake_ai.llm.last_user_message
    assert "1200 KZT" in fake_ai.llm.last_user_message


# --- Database rules, enforced by Postgres itself ---

async def add_user(db):
    user = User(email=f"{os.urandom(6).hex()}@example.com", hashed_password="x")
    db.add(user)
    await db.flush()
    return user


@pytest.mark.db
async def test_deleting_a_user_removes_their_documents_and_chunks():
    async with AsyncSessionLocal() as db:
        user = await add_user(db)
        doc = Document(filename="a.pdf", user_id=user.id)
        doc.chunks = [DocumentChunk(content="x", embedding=fake_vector("x"))]
        db.add(doc)
        await db.commit()
        # Plain SQL, so the ON DELETE CASCADE in the schema is tested, not the ORM's own cascade.
        await db.execute(text("DELETE FROM users WHERE id = :id"), {"id": user.id})
        await db.commit()

        assert await db.scalar(select(func.count()).select_from(Document).where(Document.user_id == user.id)) == 0
    assert await chunk_count(document_id=doc.id) == 0


@pytest.mark.db
async def test_a_user_cannot_have_two_documents_with_the_same_filename():
    async with AsyncSessionLocal() as db:
        user = await add_user(db)
        other = await add_user(db)
        db.add_all([Document(filename="a.pdf", user_id=user.id), Document(filename="a.pdf", user_id=other.id)])
        await db.commit()

        db.add(Document(filename="a.pdf", user_id=user.id))
        with pytest.raises(IntegrityError, match="uq_documents_user_id_filename"):
            await db.commit()


@pytest.mark.db
async def test_a_document_needs_an_existing_owner():
    async with AsyncSessionLocal() as db:
        db.add(Document(filename="orphan.pdf", user_id=987654))
        with pytest.raises(IntegrityError, match="documents_user_id_fkey"):
            await db.commit()


@pytest.mark.db
async def test_registering_the_same_email_twice_is_rejected_by_the_database():
    async with AsyncSessionLocal() as db:
        db.add_all([User(email="dup@example.com", hashed_password="x"), User(email="dup@example.com", hashed_password="y")])
        with pytest.raises(IntegrityError, match="ix_users_email"):
            await db.commit()


# --- Worker paths not reached through the API ---

@pytest.mark.db
async def test_worker_skips_a_file_the_user_already_has(fake_ai, tmp_path):
    async with AsyncSessionLocal() as db:
        user = await add_user(db)
        existing = Document(filename="known.pdf", user_id=user.id)
        db.add(existing)
        await db.commit()
    path = tmp_path / "known.pdf"
    path.write_bytes(make_pdf(["anything"]))

    result = pdf_service.process_pdf_task(str(path), "known.pdf", user_id=user.id)

    assert result == {"document_id": existing.id, "status": "already_exists"}
    assert fake_ai.embedder.documents == []
    assert not path.exists()


@pytest.mark.db
async def test_worker_with_a_missing_file_stores_nothing(fake_ai, tmp_path):
    async with AsyncSessionLocal() as db:
        user = await add_user(db)
        await db.commit()

    result = pdf_service.process_pdf_task(str(tmp_path / "gone.pdf"), "gone.pdf", user_id=user.id)

    assert result is None
    assert fake_ai.embedder.documents == []


# --- Startup configuration ---

@pytest.mark.parametrize("secret", ["", "too-short"])
def test_app_refuses_to_start_without_a_strong_jwt_secret(secret):
    env = {**os.environ, "JWT_SECRET_KEY": secret, "PYTHONPATH": str(ROOT)}
    result = subprocess.run([sys.executable, "-c", "import app.main"], cwd=ROOT, env=env, capture_output=True, text=True)

    assert result.returncode != 0
    assert "JWT_SECRET_KEY must be set and at least 32 characters long." in result.stderr
