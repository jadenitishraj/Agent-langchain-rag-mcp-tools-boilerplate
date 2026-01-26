"""
Vector Store - Qdrant-based vector database
Supports local file-based storage (no server required)
"""
import os
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, 
    Distance, 
    PointStruct,
    ScalarQuantizationConfig,
    ScalarType,
    QuantizationConfig,
    Filter,
    FieldCondition,
    MatchValue
)

from .config import (
    QDRANT_PATH,
    COLLECTION_NAME,
    EMBED_DIMENSION,
    ENABLE_QUANTIZATION,
    USE_LOCAL_QDRANT,
    QDRANT_HOST,
    QDRANT_PORT
)

# Global client instance
_client: Optional[QdrantClient] = None

def get_client() -> QdrantClient:
    """Get or create Qdrant client."""
    global _client
    
    if _client is None:
        if USE_LOCAL_QDRANT:
            # Local file-based storage (no server needed)
            os.makedirs(QDRANT_PATH, exist_ok=True)
            _client = QdrantClient(path=QDRANT_PATH)
            print(f"   📦 Using local Qdrant at: {QDRANT_PATH}")
        else:
            # Connect to Qdrant server
            _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            print(f"   🌐 Connected to Qdrant server at {QDRANT_HOST}:{QDRANT_PORT}")
    
    return _client

def collection_exists() -> bool:
    """Check if collection exists."""
    client = get_client()
    try:
        collections = client.get_collections().collections
        return any(c.name == COLLECTION_NAME for c in collections)
    except:
        return False

def create_collection(recreate: bool = False):
    """
    Create the vector collection.
    
    Args:
        recreate: If True, delete existing collection first
    """
    client = get_client()
    
    if recreate and collection_exists():
        client.delete_collection(COLLECTION_NAME)
        print(f"   🗑️  Deleted existing collection: {COLLECTION_NAME}")
    
    # Configure quantization for storage optimization
    quantization_config = None
    if ENABLE_QUANTIZATION:
        quantization_config = ScalarQuantizationConfig(
            type=ScalarType.INT8,
            always_ram=True
        )
    
    # Create collection with vector configuration
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBED_DIMENSION,
            distance=Distance.COSINE
        ),
        quantization_config=quantization_config
    )
    
    quant_status = "with INT8 quantization" if ENABLE_QUANTIZATION else "without quantization"
    print(f"   ✓ Created collection: {COLLECTION_NAME} ({quant_status})")

def store_vectors(
    chunks: List,
    embeddings: List[List[float]],
    recreate: bool = True
):
    """
    Store chunks and their embeddings in Qdrant.
    
    Args:
        chunks: List of Chunk objects
        embeddings: List of embedding vectors
        recreate: If True, recreate collection
    """
    client = get_client()
    
    # Create collection
    create_collection(recreate=recreate)
    
    # Prepare points
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(PointStruct(
            id=i,
            vector=embedding,
            payload={
                "text": chunk.text,
                "source": chunk.metadata.get("source", ""),
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata
            }
        ))
    
    # Upsert in batches
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )
    
    print(f"   ✓ Stored {len(points)} vectors in Qdrant")
    
    # Get collection info
    info = client.get_collection(COLLECTION_NAME)
    print(f"   📊 Collection stats: {info.points_count} points")

def search_vectors(
    query_embedding: List[float],
    limit: int = 5,
    score_threshold: float = 0.0,
    source_filter: Optional[str] = None
) -> List[Dict]:
    """
    Search for similar vectors.
    """
    client = get_client()
    
    # Use query_points for newer Qdrant API
    try:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=limit,
        )
        
        # Format results
        formatted = []
        for point in results.points:
            formatted.append({
                "text": point.payload.get("text", ""),
                "score": point.score,
                "source": point.payload.get("source", ""),
                "metadata": point.payload.get("metadata", {}),
                "id": point.id
            })
        
        return formatted
    except Exception as e:
        print(f"   ⚠️  Search error: {e}")
        return []

def get_all_texts() -> List[str]:
    """Get all texts from the collection (for BM25 index)."""
    client = get_client()
    
    if not collection_exists():
        return []
    
    # Scroll through all points
    all_texts = []
    offset = None
    
    while True:
        results, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        
        for point in results:
            all_texts.append(point.payload.get("text", ""))
        
        if offset is None:
            break
    
    return all_texts

def get_collection_info() -> Dict:
    """Get information about the collection."""
    client = get_client()
    
    if not collection_exists():
        return {"exists": False}
    
    info = client.get_collection(COLLECTION_NAME)
    return {
        "exists": True,
        "name": COLLECTION_NAME,
        "points_count": info.points_count,
        "status": info.status.name
    }
