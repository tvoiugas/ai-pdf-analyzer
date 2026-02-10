from sqlalchemy.orm import Session
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from app.models.sql_models import Document, DocumentChunk
from app.core.logger import logger
from app.core.celery_app import celery_app
from app.core.database import SessionLocal
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
logger.info(f"Connecting to Ollama at: {OLLAMA_URL}")

embeddings_model = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url=OLLAMA_URL,
    client_kwargs={"timeout": 300.0}
)

@celery_app.task(name="app.services.pdf_service.process_pdf_task")
def process_pdf_task(file_path: str, filename: str, user_id: str = "guest"):
    db = SessionLocal()
    try:
        logger.info(f"Processing PDF: {filename} for user: {user_id}")

        existing_doc = db.query(Document).filter(
            Document.filename == filename,
            Document.user_id == user_id
        ).first()
        
        if existing_doc:
            logger.info(f"Document {filename} already exists for user {user_id}. Skipping.")
            return {"document_id": existing_doc.id, "status": "already_exists"}

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return

        loader = PyPDFLoader(file_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        chunks = text_splitter.split_documents(pages)

        db_doc = Document(
            filename=filename,
            user_id=user_id
        )
        db.add(db_doc)
        db.commit()
        db.refresh(db_doc)

        logger.info(f"Created Document ID: {db_doc.id}. Generating embeddings.")

        for chunk in chunks:
            vector = embeddings_model.embed_query(chunk.page_content)

            db_chunk = DocumentChunk(
                document_id=db_doc.id,
                content=chunk.page_content,
                embedding=vector
            )
            db.add(db_chunk)
        
        db.commit()
        logger.info(f"Successfully ingested {filename}")

        return {"document_id": db_doc.id, "status": "completed"}
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}", exc_info=True)
        db.rollback()
        raise e
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Temporary file removed: {file_path}")
        db.close()
