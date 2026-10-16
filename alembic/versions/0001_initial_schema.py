"""Initial schema: users, documents, document chunks with an HNSW vector index

Revision ID: 0001
Revises:
Create Date: 2026-10-16

Replaces the three earlier migrations, which never created the tables they altered.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# hnsw.iterative_scan arrived in 0.8.0. On older versions Postgres still accepts the SET as an
# unknown custom setting, so retrieval would silently lose it instead of failing.
MIN_PGVECTOR_VERSION = (0, 8, 0)


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # A database created under an older image keeps the old extension version until updated.
    op.execute("ALTER EXTENSION vector UPDATE")
    version = op.get_bind().execute(sa.text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")).scalar_one()
    if tuple(int(part) for part in version.split(".")) < MIN_PGVECTOR_VERSION:
        raise RuntimeError(
            f"pgvector {version} is installed; 0.8.0+ is required. Use the image pinned in docker-compose.yml."
        )

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("hashed_password", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("upload_date", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("user_id", "filename", name="uq_documents_user_id_filename"),
    )
    op.create_index("ix_documents_user_id", "documents", ["user_id"])

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("document_id", sa.Integer(), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1024)),
    )
    op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"])
    # vector_cosine_ops matches the <=> operator retrieval orders by.
    op.create_index(
        "ix_document_chunks_embedding_hnsw",
        "document_chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    """Downgrade schema."""
    # The vector extension is left in place: other schemas in the database may rely on it.
    op.drop_table("document_chunks")
    op.drop_table("documents")
    op.drop_table("users")
