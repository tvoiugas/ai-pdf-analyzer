from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from sqlalchemy import text

from app.core.database import AsyncSessionLocal
from app.models.sql_models import Document, DocumentChunk, User
from app.services import ask_service

QUERY = [1.0] + [0.0] * 1023


def near_query(i):
    """Very close to QUERY, each slightly different."""
    v = list(QUERY)
    v[1 + i % 1000] = 0.01
    return v


def opposite_query(i):
    """As far from QUERY as cosine distance goes."""
    v = [-x for x in near_query(i)]
    return v


@pytest.fixture
def fake_models(monkeypatch):
    embedder = AsyncMock()
    embedder.aembed_query.return_value = QUERY
    llm = AsyncMock()
    llm.ainvoke.return_value = MagicMock(content="ok")
    monkeypatch.setattr(ask_service, "embeddings_model", embedder)
    monkeypatch.setattr(ask_service, "llm", llm)
    return embedder, llm


async def test_query_is_embedded_with_the_retrieval_prefix(fake_models):
    embedder, _ = fake_models
    db = AsyncMock()
    db.execute.return_value = [("chunk",)]

    await ask_service.get_answer(db, "What is the notice period?", user_id=1)

    embedder.aembed_query.assert_awaited_once_with(
        "Represent this sentence for searching relevant passages: What is the notice period?"
    )


async def test_iterative_scan_is_enabled_for_the_search_transaction(fake_models):
    db = AsyncMock()
    db.execute.return_value = [("chunk",)]

    await ask_service.get_answer(db, "q", user_id=1)

    statements = [str(call.args[0]) for call in db.execute.call_args_list]
    assert statements[0] == "SET LOCAL hnsw.iterative_scan = strict_order"
    assert "ORDER BY embedding <=>" in statements[1]


@pytest.mark.db
async def test_user_with_few_chunks_still_gets_results_among_many(fake_models):
    """The HNSW index returns its 40 nearest candidates before the user filter; without iterative
    scan, a user whose chunks are all outside those 40 would get nothing back."""
    _, llm = fake_models
    async with AsyncSessionLocal() as db:
        crowd = User(email=f"{uuid4().hex}@example.com", hashed_password="x")
        asker = User(email=f"{uuid4().hex}@example.com", hashed_password="x")
        db.add_all([crowd, asker])
        await db.flush()
        crowd_doc = Document(filename="crowd.pdf", user_id=crowd.id)
        crowd_doc.chunks = [DocumentChunk(content=f"crowd {i}", embedding=near_query(i)) for i in range(100)]
        asker_doc = Document(filename="mine.pdf", user_id=asker.id)
        asker_doc.chunks = [DocumentChunk(content=f"mine {i}", embedding=opposite_query(i)) for i in range(5)]
        db.add_all([crowd_doc, asker_doc])
        await db.commit()
        asker_id = asker.id

    async with AsyncSessionLocal() as db:
        # A table this small would be sorted exactly; force the HNSW index as a large table would use it.
        # SET LOCAL ends with this transaction, which get_answer shares.
        await db.execute(text("SET LOCAL enable_seqscan = off"))
        answer = await ask_service.get_answer(db, "q", user_id=asker_id)

    assert answer == "ok"
    prompt = llm.ainvoke.call_args.args[0][1].content
    assert all(f"mine {i}" in prompt for i in range(5))
    assert "crowd" not in prompt
