from sqlalchemy.exc import IntegrityError
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

# embed_documents() sends its whole input as one Ollama request, so a large PDF is embedded in bounded
# batches: each request stays well inside the client timeout, and progress shows up in the logs.
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))

@celery_app.task(name="app.services.pdf_service.process_pdf_task")
def process_pdf_task(file_path: str, filename: str, user_id: int):
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
        texts = [chunk.page_content for chunk in chunks]

        if not texts:
            # e.g. a scanned PDF with no text layer: there is nothing to search, so no document is created.
            logger.warning(f"No extractable text in {filename}; nothing stored.")
            return {"document_id": None, "status": "no_text"}

        # Embed everything before touching the database, so a failure part-way leaves no trace
        # and no transaction sits open during the slow Ollama calls.
        vectors = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start:start + EMBED_BATCH_SIZE]
            batch_vectors = embeddings_model.embed_documents(batch)
            # A short reply from the embedder must fail loudly, not pair text with the wrong vector.
            if len(batch_vectors) != len(batch):
                raise ValueError(f"Expected {len(batch)} embeddings, got {len(batch_vectors)}")
            vectors.extend(batch_vectors)
            logger.info(f"Embedded {len(vectors)}/{len(texts)} chunks of {filename}")

        # The document and its chunks are committed together: it only becomes visible once complete.
        db_doc = Document(filename=filename, user_id=user_id)
        db_doc.chunks = [
            DocumentChunk(content=text, embedding=vector)
            for text, vector in zip(texts, vectors, strict=True)
        ]
        db.add(db_doc)
        try:
            db.commit()
        except IntegrityError as e:
            # Another task stored the same filename for this user while this one was embedding.
            if getattr(e.orig.diag, "constraint_name", None) != "uq_documents_user_id_filename":
                raise
            db.rollback()
            existing_id = db.query(Document.id).filter(
                Document.filename == filename,
                Document.user_id == user_id
            ).scalar()
            logger.info(f"Document {filename} was stored by a parallel task for user {user_id}. Skipping.")
            return {"document_id": existing_id, "status": "already_exists"}
        logger.info(f"Successfully ingested {filename} as document {db_doc.id} with {len(texts)} chunks")

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
