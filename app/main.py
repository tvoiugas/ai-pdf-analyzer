from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import anyio
import os

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

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    filename = f"{current_user.id}_{file.filename}"
    full_path = os.path.join(SAVE_PATH, filename)

    async with await anyio.open_file(full_path, "wb") as buffer:
        while chunk := await file.read(UPLOAD_CHUNK_SIZE):
            await buffer.write(chunk)

    if os.path.exists(full_path):
        logger.info(f"File saved successfully at {full_path}")
        # delay() is a blocking round-trip to the Redis broker.
        task = await run_in_threadpool(process_pdf_task.delay, full_path, file.filename, current_user.id)

        return {"status": "success", "filename": file.filename, "message": "File processing in background."}
    else:
        logger.error(f"File was not saved: {full_path}")
        return {"error": "Failed to save file"}

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
