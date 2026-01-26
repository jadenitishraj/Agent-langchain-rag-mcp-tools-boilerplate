import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from .config import EMBED_MODEL

# Load environment variables
load_dotenv()

client = OpenAI()

def embed_text(text: str) -> List[float]:
    """Embed a single text string."""
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding

def embed_chunks(chunks: List[dict]) -> List[List[float]]:
    """Embed multiple chunks and return list of embeddings."""
    print(f"  Embedding {len(chunks)} chunks...")
    
    embeddings = []
    for i, chunk in enumerate(chunks):
        emb = embed_text(chunk["text"])
        embeddings.append(emb)
        
        # Progress indicator every 10 chunks
        if (i + 1) % 10 == 0:
            print(f"    → {i + 1}/{len(chunks)} embedded")
    
    print(f"  ✓ All {len(chunks)} chunks embedded")
    return embeddings
