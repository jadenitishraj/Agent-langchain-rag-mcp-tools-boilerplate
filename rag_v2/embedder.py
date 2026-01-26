"""
Embedding Service - Generates vector embeddings
Uses OpenAI text-embedding-3-large for high quality embeddings
"""
import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI

from .config import EMBED_MODEL, EMBED_DIMENSION

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

def embed_text(text: str) -> List[float]:
    """
    Embed a single text string.
    
    Args:
        text: The text to embed
    
    Returns:
        List of floats representing the embedding vector
    """
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding

def embed_texts(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    """
    Embed multiple texts with batching for efficiency.
    
    Args:
        texts: List of texts to embed
        batch_size: Number of texts per API call
    
    Returns:
        List of embedding vectors
    """
    all_embeddings = []
    total = len(texts)
    
    print(f"   Embedding {total} texts...")
    
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch
        )
        
        # Extract embeddings in order
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        
        # Progress
        processed = min(i + batch_size, total)
        print(f"   ✓ {processed}/{total} embedded")
    
    return all_embeddings

def embed_chunks(chunks: List) -> List[List[float]]:
    """
    Embed a list of Chunk objects.
    
    Args:
        chunks: List of Chunk objects
    
    Returns:
        List of embedding vectors
    """
    texts = [chunk.text for chunk in chunks]
    return embed_texts(texts)

def get_embedding_dimension() -> int:
    """Return the dimension of the embedding model."""
    return EMBED_DIMENSION
