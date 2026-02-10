from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
import shutil
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
    user_id: str
    document_id: Optional[int] = None


app = FastAPI(title="AI Document Analyzer") 

SAVE_PATH = "/app/data"
os.makedirs(SAVE_PATH, exist_ok=True)

@app.post("/upload")
async def upload_pdf(user_id: str = "0", file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    filename = f"{user_id}_{file.filename}"
    full_path = os.path.join(SAVE_PATH, filename)

    with open(full_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    if os.path.exists(full_path):
        logger.info(f"File saved successfully at {full_path}")
        task = process_pdf_task.delay(full_path, file.filename, user_id)

        return {"status": "success", "filename": file.filename, "message": "File processing in background."}
    else:
        logger.error(f"File was not saved: {full_path}")
        return {"error": "Failed to save file"}

@app.post("/ask")
async def ask_question(request: QuestionRequest, db: Session = Depends(get_db)):
    answer = await get_answer(
        db=db, 
        question=request.question, 
        user_id=request.user_id, 
        document_id=request.document_id
    )
    return {"question": request.question, "answer": answer}

@app.get("/documents")
async def list_documents(user_id: str, filename: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(Document)
    if filename:
        query = query.filter(Document.filename.ilike(f"%{filename}%"))
    
    docs = db.query(Document).filter(Document.user_id == user_id).all()
    return [{"id": d.id, "filename": d.filename, "upload_date": d.upload_date} for d in docs]

@app.delete("/documents/{document_id}")
async def delete_document(document_id: int, user_id: str, db: Session = Depends(get_db)):
    db_doc = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == user_id
    ).first()

    if not db_doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    
    db.delete(db_doc)
    db.commit()

    return {"status": "success", "message": f"Document {document_id} deleted."}
