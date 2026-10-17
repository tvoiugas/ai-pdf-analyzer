# AI PDF Analyzer

[![Tests](https://github.com/tvoiugas/ai-pdf-analyzer/actions/workflows/test.yml/badge.svg)](https://github.com/tvoiugas/ai-pdf-analyzer/actions/workflows/test.yml)
![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen)
![Python](https://img.shields.io/badge/python-3.14-blue)

A question-answering API over your own PDF documents (RAG). Upload a PDF, and a background worker splits it into chunks, embeds them with a local Ollama model and stores them in PostgreSQL with pgvector. Questions are answered by an LLM using only the most relevant chunks of your documents, in English or Russian to match the question.

**Stack:** FastAPI · Celery + Redis · PostgreSQL + pgvector · LangChain + Ollama · Alembic · pytest

## How it works

1. **Upload:** `/upload` validates the PDF, stores it under a server-generated name and queues a Celery task.
2. **Ingestion:** the worker splits the text into ~500-character chunks, embeds them in batches (`mxbai-embed-large`) and saves the document together with its chunks in a single transaction.
3. **Retrieval:** a question is embedded and the 15 closest chunks are found with an HNSW cosine index, limited to the user's own documents.
4. **Answer:** `llama3.1` answers from those chunks only. If they don't contain the answer, the API says so instead of guessing.

Every user sees only their own documents; all document endpoints require a JWT.

## Quick start

Requirements: Docker with Docker Compose, and [Ollama](https://ollama.com) running on the host.

```bash
ollama pull mxbai-embed-large
ollama pull llama3.1
cp .env.example .env
```

Fill in `.env` (see [Configuration](#configuration)). For the default Docker setup:

```env
DATABASE_URL=postgresql://myuser:mypassword@postgres:5432/rag_db
REDIS_URL=redis://redis:6379/0
OLLAMA_BASE_URL=http://host.docker.internal:11434
JWT_SECRET_KEY=<output of: python -c "import secrets; print(secrets.token_urlsafe(48))">
```

Start everything:

```bash
docker compose up --build -d
```

The API runs at http://localhost:8000, with interactive docs at http://localhost:8000/docs. Database migrations are applied automatically when the API container starts.

## Using the API

Register, then exchange your credentials for a token:

```bash
curl -X POST http://localhost:8000/auth/register -H "Content-Type: application/json" -d '{"email": "me@example.com", "password": "long-enough-password"}'
curl -X POST http://localhost:8000/auth/token -d "username=me@example.com&password=long-enough-password"
```

Send the token as `Authorization: Bearer <access_token>`, or use the **Authorize** button in `/docs`.

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/auth/register` | Create an account (`email`, `password` of 8–128 characters) |
| `POST` | `/auth/token` | Log in with form fields `username` (the email) and `password` |
| `POST` | `/upload` | Upload a PDF (multipart field `file`); processed in the background |
| `GET` | `/documents` | List your documents; optional `filename` filter (case-insensitive) |
| `DELETE` | `/documents/{id}` | Delete a document and its chunks |
| `POST` | `/ask` | Ask a question: `{"question": "...", "document_id": null}`; set `document_id` to search one document |

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `DATABASE_URL` | — | PostgreSQL connection URL (pgvector 0.8.0 or newer) |
| `REDIS_URL` | `redis://localhost:6379/0` | Celery broker and result backend |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama server |
| `JWT_SECRET_KEY` | — | Token signing key, at least 32 characters; the API won't start without it |
| `JWT_EXPIRE_MINUTES` | `30` | Token lifetime |
| `MAX_UPLOAD_MB` | `50` | Largest accepted PDF |
| `EMBED_BATCH_SIZE` | `64` | Chunks per embedding request |
| `UPLOAD_DIR` | `data/` | Where uploads wait for the worker; must be shared by the API and the worker |

## Development

**Migrations:** the schema is managed by Alembic. Outside Docker, apply it with:

```bash
alembic upgrade head
```

After changing a model, add a migration (`alembic revision --autogenerate -m "..."`). The test suite fails if models and migrations differ.

**Tests:** tests marked `db` run against a real PostgreSQL with pgvector. With Docker running, a temporary database container is started and removed automatically:

```bash
pytest --cov
```

Without Docker, or to skip database tests:

```bash
pytest -m "not db"
```

To use an existing database instead, set `TEST_DATABASE_URL`. Tests empty the tables after each run and never read `DATABASE_URL`, so never point `TEST_DATABASE_URL` at a database with data you need.

Coverage is measured by branch, and the run fails below 90%. GitHub Actions runs the same suite on every push and pull request, along with a migration upgrade, downgrade and re-upgrade check.
