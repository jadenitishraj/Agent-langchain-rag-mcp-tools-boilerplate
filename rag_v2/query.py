"""
Query Engine - Hybrid search with reranking
Combines vector search (semantic) with BM25 (keyword) for best results
"""
from typing import List, Dict, Optional

from .config import (
    VECTOR_WEIGHT,
    BM25_WEIGHT,
    DEFAULT_TOP_K,
    RERANK_TOP_K
)
from .embedder import embed_text
from .vector_store import search_vectors, get_all_texts
from .bm25_index import search_bm25, get_text_by_index, load_bm25_index
from .reranker import rerank_results, diversity_rerank

def normalize_scores(scores: List[float]) -> List[float]:
    """Normalize scores to 0-1 range."""
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [1.0] * len(scores)
    
    return [(s - min_score) / (max_score - min_score) for s in scores]

def hybrid_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    vector_weight: float = VECTOR_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
    use_reranker: bool = True,
    use_diversity: bool = True
) -> List[Dict]:
    """
    Perform hybrid search combining vector and BM25 results.
    
    Args:
        query: Search query
        top_k: Number of results to return
        vector_weight: Weight for vector search (0-1)
        bm25_weight: Weight for BM25 search (0-1)
        use_reranker: Whether to apply reranking
        use_diversity: Whether to apply diversity filter
    
    Returns:
        List of search results
    """
    # Get more results initially for reranking
    initial_k = top_k * 3 if use_reranker else top_k
    
    # 1. Vector search
    print(f"\n🔍 Hybrid Search for: '{query}'")
    print("-" * 50)
    
    query_embedding = embed_text(query)
    vector_results = search_vectors(query_embedding, limit=initial_k)
    print(f"   Vector search: {len(vector_results)} results")
    
    # 2. BM25 search
    bm25_results = search_bm25(query, top_k=initial_k)
    print(f"   BM25 search: {len(bm25_results)} results")
    
    # 3. Combine results using Reciprocal Rank Fusion (RRF)
    text_scores = {}  # text -> (vector_score, bm25_score, combined_score)
    text_metadata = {}  # text -> metadata
    
    # Process vector results
    vector_scores = [r["score"] for r in vector_results]
    normalized_vector = normalize_scores(vector_scores)
    
    for i, result in enumerate(vector_results):
        text = result["text"]
        score = normalized_vector[i] if normalized_vector else 0
        text_scores[text] = {"vector": score, "bm25": 0, "combined": 0}
        text_metadata[text] = {
            "source": result.get("source", ""),
            "metadata": result.get("metadata", {})
        }
    
    # Process BM25 results
    bm25_scores_raw = [score for _, score in bm25_results]
    normalized_bm25 = normalize_scores(bm25_scores_raw)
    
    for i, (idx, _) in enumerate(bm25_results):
        text = get_text_by_index(idx)
        if text:
            score = normalized_bm25[i] if normalized_bm25 else 0
            if text in text_scores:
                text_scores[text]["bm25"] = score
            else:
                text_scores[text] = {"vector": 0, "bm25": score, "combined": 0}
                text_metadata[text] = {"source": "", "metadata": {}}
    
    # Calculate combined scores
    for text in text_scores:
        vs = text_scores[text]["vector"]
        bs = text_scores[text]["bm25"]
        text_scores[text]["combined"] = (vs * vector_weight) + (bs * bm25_weight)
    
    # Sort by combined score
    sorted_texts = sorted(
        text_scores.keys(),
        key=lambda t: text_scores[t]["combined"],
        reverse=True
    )
    
    # Build results
    results = []
    for text in sorted_texts[:initial_k]:
        results.append({
            "text": text,
            "score": text_scores[text]["combined"],
            "vector_score": text_scores[text]["vector"],
            "bm25_score": text_scores[text]["bm25"],
            **text_metadata.get(text, {})
        })
    
    print(f"   Hybrid combined: {len(results)} results")
    
    # 4. Rerank
    if use_reranker and results:
        results = rerank_results(query, results, top_k=top_k * 2)
        print(f"   After reranking: {len(results)} results")
    
    # 5. Diversity filter
    if use_diversity and results:
        results = diversity_rerank(results, top_k=top_k)
        print(f"   After diversity filter: {len(results)} results")
    
    print("-" * 50)
    
    return results[:top_k]

def query_memory(
    question: str,
    k: int = RERANK_TOP_K
) -> List[Dict]:
    """
    Query the knowledge base.
    
    Args:
        question: The question to search for
        k: Number of results
    
    Returns:
        List of relevant chunks
    """
    return hybrid_search(question, top_k=k)

def get_context(
    question: str,
    k: int = RERANK_TOP_K
) -> str:
    """
    Get context string for RAG prompt.
    
    Args:
        question: The question
        k: Number of chunks to include
    
    Returns:
        Formatted context string
    """
    results = query_memory(question, k=k)
    
    if not results:
        return ""
    
    # Format context with source attribution
    context_parts = []
    for i, result in enumerate(results, 1):
        source = result.get("source", "Unknown")
        text = result.get("text", "")
        score = result.get("score", 0)
        
        context_parts.append(f"[Source {i}: {source} (relevance: {score:.2f})]\n{text}")
    
    return "\n\n---\n\n".join(context_parts)

def simple_search(query: str, k: int = 5) -> List[Dict]:
    """
    Simple vector-only search (for comparison/debugging).
    """
    query_embedding = embed_text(query)
    return search_vectors(query_embedding, limit=k)
