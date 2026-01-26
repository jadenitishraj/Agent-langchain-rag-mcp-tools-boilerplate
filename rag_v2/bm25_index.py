"""
BM25 Index - Keyword-based search for hybrid retrieval
"""
import os
import json
import pickle
from typing import List, Tuple, Optional
from rank_bm25 import BM25Okapi

from .config import QDRANT_PATH

BM25_INDEX_PATH = os.path.join(QDRANT_PATH, "bm25_index.pkl")
BM25_CORPUS_PATH = os.path.join(QDRANT_PATH, "bm25_corpus.json")

# Global BM25 index
_bm25_index: Optional[BM25Okapi] = None
_corpus: Optional[List[str]] = None

def tokenize(text: str) -> List[str]:
    """Simple tokenization for BM25."""
    # Lowercase and split on non-alphanumeric
    import re
    tokens = re.findall(r'\b\w+\b', text.lower())
    
    # Remove very short tokens
    tokens = [t for t in tokens if len(t) > 2]
    
    return tokens

def build_bm25_index(texts: List[str]) -> BM25Okapi:
    """
    Build a BM25 index from texts.
    
    Args:
        texts: List of document texts
    
    Returns:
        BM25Okapi index
    """
    global _bm25_index, _corpus
    
    print("   Building BM25 index...")
    
    # Tokenize all texts
    tokenized_corpus = [tokenize(text) for text in texts]
    
    # Create BM25 index
    _bm25_index = BM25Okapi(tokenized_corpus)
    _corpus = texts
    
    # Save index and corpus
    os.makedirs(QDRANT_PATH, exist_ok=True)
    
    with open(BM25_INDEX_PATH, 'wb') as f:
        pickle.dump(_bm25_index, f)
    
    with open(BM25_CORPUS_PATH, 'w') as f:
        json.dump(texts, f)
    
    print(f"   ✓ BM25 index built with {len(texts)} documents")
    
    return _bm25_index

def load_bm25_index() -> Tuple[Optional[BM25Okapi], Optional[List[str]]]:
    """
    Load BM25 index from disk.
    
    Returns:
        Tuple of (BM25 index, corpus)
    """
    global _bm25_index, _corpus
    
    if _bm25_index is not None:
        return _bm25_index, _corpus
    
    if not os.path.exists(BM25_INDEX_PATH):
        return None, None
    
    try:
        with open(BM25_INDEX_PATH, 'rb') as f:
            _bm25_index = pickle.load(f)
        
        with open(BM25_CORPUS_PATH, 'r') as f:
            _corpus = json.load(f)
        
        return _bm25_index, _corpus
    except Exception as e:
        print(f"   ⚠️  Error loading BM25 index: {e}")
        return None, None

def search_bm25(query: str, top_k: int = 10) -> List[Tuple[int, float]]:
    """
    Search using BM25.
    
    Args:
        query: Search query
        top_k: Number of results
    
    Returns:
        List of (index, score) tuples
    """
    index, corpus = load_bm25_index()
    
    if index is None:
        return []
    
    # Tokenize query
    query_tokens = tokenize(query)
    
    if not query_tokens:
        return []
    
    # Get scores for all documents
    scores = index.get_scores(query_tokens)
    
    # Get top-k indices
    import numpy as np
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # Only include if there's some match
            results.append((int(idx), float(scores[idx])))
    
    return results

def get_text_by_index(index: int) -> Optional[str]:
    """Get text by corpus index."""
    _, corpus = load_bm25_index()
    
    if corpus and 0 <= index < len(corpus):
        return corpus[index]
    
    return None
