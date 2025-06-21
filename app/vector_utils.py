# app/vector_utils.py

import faiss
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS index setup
dimension = 384
index = faiss.IndexFlatL2(dimension)
stored_chunks: List[str] = []

# Persistence paths
CHUNKS_PATH = "persist/chunks.json"
INDEX_PATH = "persist/faiss.index"


def embed_and_index_chunks(chunks: List[str]):
    global stored_chunks, index
    index.reset()
    embeddings = embedding_model.encode(chunks)
    index.add(np.array(embeddings).astype('float32'))
    stored_chunks = chunks


def search_similar_chunks(query: str, top_k: int = 5) -> List[Tuple[str, float]]:
    query_embedding = embedding_model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    results = [(stored_chunks[i], distances[0][idx])
               for idx, i in enumerate(indices[0]) if i < len(stored_chunks)]
    return results


def save_index_and_chunks():
    os.makedirs("persist", exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(stored_chunks, f, ensure_ascii=False, indent=2)
    faiss.write_index(index, INDEX_PATH)
    print("✅ FAISS index and chunks saved.")


def load_index_and_chunks():
    global stored_chunks, index
    if not os.path.exists(CHUNKS_PATH) or not os.path.exists(INDEX_PATH):
        print("⚠️ No saved index/chunks found.")
        return
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        stored_chunks = json.load(f)
    index = faiss.read_index(INDEX_PATH)
    print(f"✅ Loaded {len(stored_chunks)} chunks and FAISS index.")


def get_stored_chunks() -> List[str]:
    return stored_chunks


def get_index() -> faiss.Index:
    return index


def clear_index_and_chunks():
    global stored_chunks, index
    stored_chunks = []
    index.reset()
    print("🧹 Cleared all chunks and index.")
