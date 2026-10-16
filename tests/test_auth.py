from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import jwt
import pytest

from app.core.database import AsyncSessionLocal
from app.core.security import JWT_SECRET_KEY, create_access_token, decode_access_token
from app.models.sql_models import Document, DocumentChunk, User

VALID_PASSWORD = "long-enough-password"

PROTECTED_ENDPOINTS = [
    ("post", "/upload"),
    ("post", "/ask"),
    ("get", "/documents"),
    ("delete", "/documents/1"),
]


def make_token(claims, key=JWT_SECRET_KEY, algorithm="HS256"):
    return jwt.encode(claims, key, algorithm=algorithm)


def in_minutes(minutes):
    return datetime.now(timezone.utc) + timedelta(minutes=minutes)


# --- Tokens (no database) ---

def test_token_round_trip():
    assert decode_access_token(create_access_token(42)) == 42


@pytest.mark.parametrize("token", [
    pytest.param(make_token({"sub": "42", "exp": in_minutes(-1)}), id="expired"),
    pytest.param(make_token({"sub": "42", "exp": in_minutes(5)}, key="x" * 40), id="wrong-key"),
    pytest.param(make_token({"sub": "42", "exp": in_minutes(5)}, key=None, algorithm="none"), id="unsigned"),
    pytest.param(make_token({"sub": "42"}), id="no-exp"),
    pytest.param(make_token({"exp": in_minutes(5)}), id="no-sub"),
    pytest.param(make_token({"sub": "admin", "exp": in_minutes(5)}), id="non-numeric-sub"),
    pytest.param("not-a-jwt", id="garbage"),
])
def test_bad_tokens_are_rejected(token):
    with pytest.raises(jwt.InvalidTokenError):
        decode_access_token(token)


@pytest.mark.parametrize("method,path", PROTECTED_ENDPOINTS)
async def test_endpoints_require_token(client, method, path):
    response = await client.request(method, path)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("method,path", PROTECTED_ENDPOINTS)
async def test_endpoints_reject_forged_token(client, method, path):
    forged = make_token({"sub": "1", "exp": in_minutes(5)}, key="attacker-key-" * 4)
    response = await client.request(method, path, headers={"Authorization": f"Bearer {forged}"})

    assert response.status_code == 401


@pytest.mark.parametrize("body", [
    pytest.param({"email": "not-an-email", "password": VALID_PASSWORD}, id="bad-email"),
    pytest.param({"email": "a@example.com", "password": "short"}, id="short-password"),
    pytest.param({"email": "a@example.com", "password": "x" * 129}, id="long-password"),
])
async def test_register_validates_input(client, body):
    response = await client.post("/auth/register", json=body)

    assert response.status_code == 422


# --- Register / login flow (real database) ---

@pytest.mark.db
async def test_register_rejects_duplicate_email_case_insensitively(client, register):
    name = uuid4().hex
    await register(f"{name.upper()}@example.com")
    response = await client.post("/auth/register", json={"email": f"{name}@EXAMPLE.com", "password": VALID_PASSWORD})

    assert response.status_code == 409


@pytest.mark.db
async def test_password_is_stored_hashed(register):
    user_id, _ = await register()
    async with AsyncSessionLocal() as db:
        user = await db.get(User, user_id)

    assert user.hashed_password.startswith("$argon2id$")


@pytest.mark.db
@pytest.mark.parametrize("who", ["wrong-password", "unknown-email"])
async def test_login_failures_look_the_same(client, register, who):
    email = f"{uuid4().hex}@example.com"
    if who == "wrong-password":
        await register(email)
    response = await client.post("/auth/token", data={"username": email, "password": "wrong-password-123"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect email or password."


@pytest.mark.db
async def test_token_of_deleted_user_stops_working(client, register):
    user_id, headers = await register()
    async with AsyncSessionLocal() as db:
        await db.delete(await db.get(User, user_id))
        await db.commit()

    response = await client.get("/documents", headers=headers)

    assert response.status_code == 401


# --- Ownership (real database) ---

@pytest.mark.db
async def test_users_only_see_and_query_their_own_documents(client, register, monkeypatch):
    from app.services import ask_service

    alice_id, alice = await register()
    bob_id, bob = await register()
    async with AsyncSessionLocal() as db:
        alice_doc = Document(filename="alice.pdf", user_id=alice_id)
        bob_doc = Document(filename="bob.pdf", user_id=bob_id)
        db.add_all([alice_doc, bob_doc])
        await db.flush()
        db.add_all([
            DocumentChunk(document_id=alice_doc.id, content="alice secret", embedding=[0.1] * 1024),
            DocumentChunk(document_id=bob_doc.id, content="bob secret", embedding=[0.1] * 1024),
        ])
        await db.commit()

    listing = await client.get("/documents", headers=bob)
    assert [d["filename"] for d in listing.json()] == ["bob.pdf"]

    mock_emb_inst = AsyncMock()
    mock_emb_inst.aembed_query.return_value = [0.1] * 1024
    mock_llm_inst = AsyncMock()
    mock_llm_inst.ainvoke.return_value = MagicMock(content="ok")
    monkeypatch.setattr(ask_service, "embeddings_model", mock_emb_inst)
    monkeypatch.setattr(ask_service, "llm", mock_llm_inst)

    # Pointing /ask at someone else's document finds nothing rather than leaking its text.
    response = await client.post("/ask", json={"question": "q", "document_id": alice_doc.id}, headers=bob)
    assert response.json()["answer"] == "No relevant information found in the documents."
    assert not mock_llm_inst.ainvoke.called

    await client.post("/ask", json={"question": "q"}, headers=bob)
    prompt = mock_llm_inst.ainvoke.call_args.args[0]
    assert "bob secret" in prompt
    assert "alice secret" not in prompt
