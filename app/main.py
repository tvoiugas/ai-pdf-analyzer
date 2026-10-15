from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import anyio
import os

from app.models.sql_models import Document
from app.core.database import get_db
from app.services.pdf_service import process_pdf_task

from app.services.ask_service import get_answer
from app.core.logger import logger
from pydantic import BaseModel
from typing import Optional


class QuestionRequest(BaseModel):
    question: str
    user_id: int
    document_id: Optional[int] = None


app = FastAPI(title="AI Document Analyzer") 

SAVE_PATH = "/app/data"
os.makedirs(SAVE_PATH, exist_ok=True)

UPLOAD_CHUNK_SIZE = 1024 * 1024

@app.post("/upload")
async def upload_pdf(user_id: int = 0, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    filename = f"{user_id}_{file.filename}"
    full_path = os.path.join(SAVE_PATH, filename)

    async with await anyio.open_file(full_path, "wb") as buffer:
        while chunk := await file.read(UPLOAD_CHUNK_SIZE):
            await buffer.write(chunk)

    if os.path.exists(full_path):
        logger.info(f"File saved successfully at {full_path}")
        # delay() is a blocking round-trip to the Redis broker.
        task = await run_in_threadpool(process_pdf_task.delay, full_path, file.filename, user_id)

        return {"status": "success", "filename": file.filename, "message": "File processing in background."}
    else:
        logger.error(f"File was not saved: {full_path}")
        return {"error": "Failed to save file"}

@app.post("/ask")
async def ask_question(request: QuestionRequest, db: AsyncSession = Depends(get_db)):
    answer = await get_answer(
        db=db, 
        question=request.question, 
        user_id=request.user_id, 
        document_id=request.document_id
    )
    return {"question": request.question, "answer": answer}

@app.get("/documents")
async def list_documents(user_id: int, filename: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    query = select(Document)
    if filename:
        query = query.where(Document.filename.ilike(f"%{filename}%"))

    result = await db.execute(select(Document).where(Document.user_id == user_id))
    docs = result.scalars().all()
    return [{"id": d.id, "filename": d.filename, "upload_date": d.upload_date} for d in docs]

@app.delete("/documents/{document_id}")
async def delete_document(document_id: int, user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).where(
        Document.id == document_id,
        Document.user_id == user_id
    ))
    db_doc = result.scalars().first()

    if not db_doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    
    await db.delete(db_doc)
    await db.commit()

    return {"status": "success", "message": f"Document {document_id} deleted."}
