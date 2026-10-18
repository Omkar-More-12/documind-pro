import os
import shutil
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.ingestion.loader import load_pdf, chunk_pages
from app.retrieval.retriever import Retriever
from app.generation.llm import generate_answer

app = FastAPI(title="DocuMind Pro")
retriever = Retriever()
os.makedirs("data/uploads", exist_ok=True)


class Question(BaseModel):
    question: str


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    path = f"data/uploads/{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    chunks = chunk_pages(load_pdf(path))
    retriever.add(chunks, source=file.filename)
    return {"status": "ok", "chunks": len(chunks)}


@app.post("/ask")
async def ask(q: Question):
    chunks = retriever.search(q.question)
    answer = generate_answer(q.question, chunks)
    return {"answer": answer, "sources": chunks}