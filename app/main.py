# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import shutil
import uuid
from pathlib import Path
from typing import List
import requests
import json
from app.pdf_utils import extract_text_from_pdf
from app.text_chunker import chunk_text
from app.vector_utils import (
    embed_and_index_chunks,
    search_similar_chunks,
    save_index_and_chunks,
    load_index_and_chunks,
    get_stored_chunks
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.on_event("startup")
async def on_startup():
    load_index_and_chunks()
    
    
def build_prompt(query: str, top_chunks: List[str]) -> str:
    return f"""Answer the question based on the following context:

{chr(10).join(top_chunks)}

Question: {query}

Answer:"""


def query_ollama(prompt: str, model: str = "llama3.2:latest") -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()

@app.get("/")
async def root():
    return {"message": "🚀 FastAPI server is running!"}

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    temp_filename = f"{uuid.uuid4()}_{file.filename}"
    temp_path = UPLOAD_DIR / temp_filename

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pages = extract_text_from_pdf(str(temp_path))
    chunks = chunk_text(pages)
    embed_and_index_chunks(chunks)
    save_index_and_chunks()

    return {
        "filename": file.filename,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "sample_chunk": chunks[0] if chunks else "No chunks found",
        "status": "Chunks embedded and indexed"
    }

@app.get("/search/")
async def search(query: str):
    stored_chunks = get_stored_chunks()
    print(stored_chunks)
    if not stored_chunks:
        raise HTTPException(status_code=400, detail="No chunks found. Please upload a document first.")

    results = search_similar_chunks(query)
    return {
        "query": query,
        "matches": [{"chunk": chunk, "score": float(score)} for chunk, score in results]
    }



def stream_from_ollama(prompt: str, model: str = "llama3"):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "stream": True
    }

    response = requests.post(url, json=payload, stream=True)
    response.raise_for_status()

    def event_stream():
        for line in response.iter_lines():
            if line:
                try:
                    json_data = json.loads(line.decode("utf-8"))
                    content = json_data.get("message", {}).get("content", "")
                    if content:
                        yield f"data: {content}\n\n"
                except Exception as e:
                    print("Stream error:", e)

    return event_stream()


@app.get("/ask-stream/")
async def ask_stream(query: str = Query(..., min_length=3), model: str = "llama3"):
    chunks = get_stored_chunks()
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks found. Please upload a document first.")

    results = search_similar_chunks(query, top_k=5)
    top_chunks = [chunk for chunk, _ in results]
    prompt = build_prompt(query, top_chunks)

    return StreamingResponse(
        stream_from_ollama(prompt, model=model),
        media_type="text/event-stream"
    )