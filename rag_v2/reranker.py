"""
Reranker - Reranks search results for better precision
Uses a simple cross-encoder style approach without external APIs
"""
from typing import List, Dict, Tuple
import re

def calculate_relevance_score(query: str, text: str) -> float:
    """
    Calculate a relevance score between query and text.
    Uses multiple heuristics for a simple but effective reranking.
    
    Args:
        query: The search query
        text: The candidate text
    
    Returns:
        Relevance score (higher is more relevant)
    """
    query_lower = query.lower()
    text_lower = text.lower()
    
    # Extract query terms
    query_terms = set(re.findall(r'\b\w+\b', query_lower))
    query_terms = {t for t in query_terms if len(t) > 2}
    
    # Calculate different relevance signals
    scores = []
    
    # 1. Term overlap score
    text_terms = set(re.findall(r'\b\w+\b', text_lower))
    if query_terms:
        overlap = len(query_terms & text_terms) / len(query_terms)
        scores.append(overlap * 0.3)
    
    # 2. Exact phrase match bonus
    if query_lower in text_lower:
        scores.append(0.3)
    
    # 3. Term proximity score (how close query terms appear in text)
    if len(query_terms) > 1:
        # Check if terms appear near each other
        window_size = 100  # characters
        for term in query_terms:
            if term in text_lower:
                pos = text_lower.find(term)
                window = text_lower[max(0, pos-window_size):pos+window_size]
                nearby_terms = sum(1 for t in query_terms if t in window)
                if nearby_terms > 1:
                    scores.append(0.2 * (nearby_terms / len(query_terms)))
                    break
    
    # 4. Position score (terms appearing earlier are better)
    avg_position = 0
    found_terms = 0
    for term in query_terms:
        pos = text_lower.find(term)
        if pos >= 0:
            avg_position += pos
            found_terms += 1
    
    if found_terms > 0:
        avg_position /= found_terms
        # Normalize: earlier = higher score
        position_score = max(0, 1 - (avg_position / len(text))) * 0.2
        scores.append(position_score)
    
    return sum(scores)

def rerank_results(
    query: str,
    results: List[Dict],
    top_k: int = 3
) -> List[Dict]:
    """
    Rerank search results based on relevance to query.
    
    Args:
        query: The original search query
        results: List of search results with 'text' and 'score' keys
        top_k: Number of results to return after reranking
    
    Returns:
        Reranked list of results
    """
    if not results:
        return []
    
    # Calculate rerank scores
    scored_results = []
    for result in results:
        text = result.get("text", "")
        original_score = result.get("score", 0)
        
        # Calculate relevance score
        relevance_score = calculate_relevance_score(query, text)
        
        # Combine original score with relevance score
        # Original score is usually similarity (0-1), relevance is our heuristic
        combined_score = (original_score * 0.6) + (relevance_score * 0.4)
        
        scored_results.append({
            **result,
            "original_score": original_score,
            "relevance_score": relevance_score,
            "final_score": combined_score
        })
    
    # Sort by combined score
    scored_results.sort(key=lambda x: x["final_score"], reverse=True)
    
    return scored_results[:top_k]

def diversity_rerank(
    results: List[Dict],
    top_k: int = 3,
    diversity_threshold: float = 0.7
) -> List[Dict]:
    """
    Rerank to maximize diversity (MMR-style).
    Reduces redundancy in results.
    
    Args:
        results: List of search results
        top_k: Number of results
        diversity_threshold: Similarity threshold for diversity
    
    Returns:
        Diverse set of results
    """
    if len(results) <= top_k:
        return results
    
    selected = [results[0]]  # Always include top result
    
    for result in results[1:]:
        if len(selected) >= top_k:
            break
        
        # Check if this result is too similar to already selected ones
        is_diverse = True
        result_text = result.get("text", "").lower()
        
        for selected_result in selected:
            selected_text = selected_result.get("text", "").lower()
            
            # Simple overlap check
            result_words = set(result_text.split())
            selected_words = set(selected_text.split())
            
            if result_words and selected_words:
                overlap = len(result_words & selected_words) / len(result_words | selected_words)
                if overlap > diversity_threshold:
                    is_diverse = False
                    break
        
        if is_diverse:
            selected.append(result)
    
    return selected
