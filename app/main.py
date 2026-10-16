from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import anyio
import os
from uuid import uuid4

from app.models.sql_models import Document, User
from app.core.database import get_db
from app.core.security import get_current_user
from app.routers import auth
from app.services.pdf_service import process_pdf_task

from app.services.ask_service import get_answer
from app.core.logger import logger
from pydantic import BaseModel
from typing import Optional


class QuestionRequest(BaseModel):
    question: str
    document_id: Optional[int] = None


app = FastAPI(title="AI Document Analyzer") 
app.include_router(auth.router)

SAVE_PATH = "/app/data"
os.makedirs(SAVE_PATH, exist_ok=True)

UPLOAD_CHUNK_SIZE = 1024 * 1024
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024
MAX_FILENAME_LENGTH = 255
# PDF readers accept the header anywhere in the first 1024 bytes.
PDF_HEADER = b"%PDF-"
PDF_HEADER_WINDOW = 1024


def display_filename(raw_name: Optional[str]) -> str:
    """Reduce the client's filename to a label for the DB and responses. It is never used to build a path."""
    # Keep only the last path component, whichever separator the client used.
    name = (raw_name or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(ch for ch in name if ch.isprintable()).strip()
    # Trim from the front so the extension survives.
    return name[-MAX_FILENAME_LENGTH:]


async def save_upload(file: UploadFile, path: str) -> None:
    first_chunk = await file.read(UPLOAD_CHUNK_SIZE)
    if PDF_HEADER not in first_chunk[:PDF_HEADER_WINDOW]:
        raise HTTPException(status_code=400, detail="File is not a valid PDF.")

    size = 0
    chunk = first_chunk
    # "x" mode refuses to overwrite, so two uploads can never share a file.
    async with await anyio.open_file(path, "xb") as buffer:
        while chunk:
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                )
            await buffer.write(chunk)
            chunk = await file.read(UPLOAD_CHUNK_SIZE)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    filename = display_filename(file.filename)
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Generated server-side: nothing the client sends ends up in the path.
    full_path = os.path.join(SAVE_PATH, f"{uuid4().hex}.pdf")
    try:
        await save_upload(file, full_path)
        logger.info(f"Saved upload {filename!r} for user {current_user.id} at {full_path}")
        # delay() is a blocking round-trip to the Redis broker.
        await run_in_threadpool(process_pdf_task.delay, full_path, filename, current_user.id)
    except BaseException:
        # Covers rejected content, client disconnects and a broker failure: no file is left without a task.
        if os.path.exists(full_path):
            os.remove(full_path)
        raise

    return {"status": "success", "filename": filename, "message": "File processing in background."}

@app.post("/ask")
async def ask_question(
    request: QuestionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    answer = await get_answer(
        db=db, 
        question=request.question, 
        user_id=current_user.id, 
        document_id=request.document_id
    )
    return {"question": request.question, "answer": answer}

@app.get("/documents")
async def list_documents(
    filename: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = select(Document).where(Document.user_id == current_user.id)
    if filename:
        # autoescape makes "%" and "_" in the search text match literally instead of acting as wildcards.
        query = query.where(Document.filename.icontains(filename, autoescape=True))

    result = await db.execute(query)
    docs = result.scalars().all()
    return [{"id": d.id, "filename": d.filename, "upload_date": d.upload_date} for d in docs]

@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(Document).where(
        Document.id == document_id,
        Document.user_id == current_user.id
    ))
    db_doc = result.scalars().first()

    if not db_doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    
    await db.delete(db_doc)
    await db.commit()

    return {"status": "success", "message": f"Document {document_id} deleted."}
