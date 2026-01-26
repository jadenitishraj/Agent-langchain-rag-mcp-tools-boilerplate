from typing import List
from .store import query_store
from .embedder import embed_text
from .config import DEFAULT_TOP_K

def query_memory(question: str, k: int = DEFAULT_TOP_K) -> List[dict]:
    """Query the Osho memory and return relevant chunks."""
    # Embed the question
    query_embedding = embed_text(question)
    
    # Query the vector store
    results = query_store(query_embedding, k)
    
    return results

def get_context(question: str, k: int = DEFAULT_TOP_K) -> str:
    """Get context string for RAG prompt."""
    results = query_memory(question, k)
    context = "\n\n---\n\n".join([r["text"] for r in results])
    return context
