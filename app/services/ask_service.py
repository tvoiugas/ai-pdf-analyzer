from sqlalchemy.orm import Session
from sqlalchemy import text
from langchain_ollama import OllamaEmbeddings, ChatOllama
from app.core.logger import logger
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

embeddings_model = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url=OLLAMA_URL,
    client_kwargs={"timeout": 300.0}
)
llm = ChatOllama(
    model="llama3.1",
    temperature=0,
    base_url=OLLAMA_URL
)

async def get_answer(db: Session, question: str, user_id: str, document_id: Optional[int] = None):
    try:
        question_vector = embeddings_model.embed_query(question)
        vector_str = f"[{','.join(map(str, question_vector))}]"

        query = """
            SELECT dc.content 
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE d.user_id = :user_id  
        """
        params = {"vector": vector_str, "user_id": user_id}

        if document_id:
            query += " AND d.id = :doc_id " 
            params["doc_id"] = document_id

        query += "ORDER BY embedding <=> :vector LIMIT 15"

        result = db.execute(text(query), params)
        chunks = [row[0] for row in result]

        if not chunks:
            return "No relevant information found in the documents."
        
        context = "\n---\n".join(chunks)
        logger.info(f"Context retrieved. Generating answer...")

        prompt = prompt = f"""
            Ты — профессиональный аналитик документов. Твоя задача — давать точные ответы, основываясь исключительно на предоставленном контексте.

            ### ИНСТРУКЦИИ:
            1. В ответах используй ТОЛЬКО русский язык.
            2. Используй ТОЛЬКО предоставленный ниже контекст. Не используй внешние знания.
            3. Если ответ есть, пиши кратко, структурировано и по существу.
            4. Если в контексте упоминаются даты, цифры или специфические термины, приводи их в ответе без изменений.
            5. Соблюдай нейтральный и деловой тон.

            ### КОНТЕКСТ:
            {context}

            ### ВОПРОС:
            {question}

            ### ОТВЕТ:
        """

        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error in ask_service: {e}", exc_info=True)
        return "An error occurred while processing your request. Please try again later."
    