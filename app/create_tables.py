from sqlalchemy import text
from app.core.database import engine, Base
from app.models.sql_models import Document, DocumentChunk
from app.core.logger import logger

# hnsw.iterative_scan arrived in 0.8.0. On older versions Postgres still accepts the SET as an
# unknown custom setting, so the feature would be silently off instead of failing.
MIN_PGVECTOR_VERSION = (0, 8, 0)


def _version(text_version):
    return tuple(int(part) for part in text_version.split("."))


def init_db():
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            # A database created under an older image keeps the old extension version until updated.
            conn.execute(text("ALTER EXTENSION vector UPDATE;"))
            version = conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")).scalar_one()
            conn.commit()
            if _version(version) < MIN_PGVECTOR_VERSION:
                raise RuntimeError(
                    f"pgvector {version} is installed; 0.8.0+ is required. Use the image pinned in docker-compose.yml."
                )
            logger.info(f"Extension 'vector' {version} ensured in PostgreSQL.")
        
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

if __name__ == "__main__":
    init_db()
