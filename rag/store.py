"""
Simplified vector store using a local JSON file and numpy for similarity search.
This avoids ChromaDB compatibility issues with Python 3.14.
"""
import os
import json
import uuid
from typing import List, Optional
import numpy as np
from .config import CHROMA_PERSIST_DIR

STORE_FILE = os.path.join(CHROMA_PERSIST_DIR, "vector_store.json")

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def load_store() -> dict:
    """Load the vector store from disk."""
    if os.path.exists(STORE_FILE):
        with open(STORE_FILE, 'r') as f:
            return json.load(f)
    return {"documents": [], "embeddings": [], "metadatas": [], "ids": []}

def save_store(store: dict) -> None:
    """Save the vector store to disk."""
    os.makedirs(os.path.dirname(STORE_FILE), exist_ok=True)
    with open(STORE_FILE, 'w') as f:
        json.dump(store, f)

def store_chunks(chunks: List[dict], embeddings: List[List[float]]) -> None:
    """Store chunks and embeddings in the local vector store."""
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    
    # Create new store (overwrite existing)
    store = {
        "documents": [c["text"] for c in chunks],
        "embeddings": embeddings,
        "metadatas": [c["metadata"] for c in chunks],
        "ids": [str(uuid.uuid4()) for _ in chunks]
    }
    
    save_store(store)
    print(f"  ✓ Stored {len(chunks)} chunks in local vector store")
    print(f"  ✓ Persisted to: {STORE_FILE}")

def query_store(query_embedding: List[float], k: int = 3) -> List[dict]:
    """Query the vector store for similar documents."""
    store = load_store()
    
    if not store["documents"]:
        return []
    
    # Calculate similarities
    similarities = []
    for i, emb in enumerate(store["embeddings"]):
        sim = cosine_similarity(query_embedding, emb)
        similarities.append((i, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Get top k results
    results = []
    for i, sim in similarities[:k]:
        results.append({
            "text": store["documents"][i],
            "metadata": store["metadatas"][i],
            "distance": 1 - sim  # Convert similarity to distance
        })
    
    return results

def get_collection():
    """Get the store (for compatibility with existing code)."""
    return load_store()

def get_chroma_client():
    """Stub for compatibility."""
    return None
