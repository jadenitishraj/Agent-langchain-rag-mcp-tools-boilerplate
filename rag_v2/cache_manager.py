"""
Semantic Cache Manager
======================
Handles storage and retrieval of query-response pairs in Qdrant.
This enables "Instant Answers" for repeated or semantically similar questions.
"""
import time
from typing import Optional, Dict, List
import json
from qdrant_client.models import Distance, VectorParams, PointStruct

from .vector_store import get_client, USE_LOCAL_QDRANT, QDRANT_PATH
from .embedder import embed_text

CACHE_COLLECTION = "semantic_cache"
CACHE_THRESHOLD = 0.70  # Adjusted for text-embedding-3-large variability

def ensure_cache_collection():
    """Ensure the cache collection exists."""
    client = get_client()
    try:
        collections = client.get_collections().collections
        exists = any(c.name == CACHE_COLLECTION for c in collections)
        
        if not exists:
            print(f"Creating cache collection: {CACHE_COLLECTION}")
            client.create_collection(
                collection_name=CACHE_COLLECTION,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE) # text-embedding-3-large default
            )
    except Exception as e:
        print(f"⚠️ Error checking/creating cache collection: {e}")

def search_cache(query: str) -> Optional[str]:
    """
    Search for a semantically similar query in the cache.
    Returns the cached response if found (similarity > threshold).
    """
    try:
        ensure_cache_collection()
        client = get_client()
        
        # Embed query
        vector = embed_text(query)
        
        # Search
        results = client.query_points(
            collection_name=CACHE_COLLECTION,
            query=vector,
            limit=1,
            with_payload=True
        ).points
        
        if not results:
            return None
            
        hit = results[0]
        if hit.score >= CACHE_THRESHOLD:
            print(f"⚡ CACHE HIT! Score: {hit.score:.4f}")
            return hit.payload.get("response")
            
        return None
        
    except Exception as e:
        print(f"⚠️ Cache search error: {e}")
        return None

def save_to_cache(query: str, response: str):
    """Save a query-response pair to the cache."""
    try:
        ensure_cache_collection()
        client = get_client()
        
        vector = embed_text(query)
        
        # Use simple int ID based on timestamp (collision unlikely for single user local)
        # or UUID. Let's use timestamp micros
        point_id = int(time.time() * 1000000)
        
        client.upsert(
            collection_name=CACHE_COLLECTION,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "query": query,
                        "response": response,
                        "timestamp": time.time()
                    }
                )
            ]
        )
        print(f"💾 Saved to cache.")
        
    except Exception as e:
        print(f"⚠️ Cache save error: {e}")
