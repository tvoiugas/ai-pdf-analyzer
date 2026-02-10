from sqlalchemy import text
from app.core.database import engine, Base
from app.models.sql_models import Document, DocumentChunk
from app.core.logger import logger

def init_db():
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()
            logger.info("Extension 'vector' ensured in PostgreSQL.")
        
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

if __name__ == "__main__":
    init_db()
