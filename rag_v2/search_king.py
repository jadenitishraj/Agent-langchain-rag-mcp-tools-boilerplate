# =============================================================================
# 🔥 ADVANCED HYBRID SEARCH ENGINE - THE SEARCH KING
# =============================================================================
"""
Advanced search engine with multiple techniques:
- Query expansion
- HyDE (Hypothetical Document Embeddings)
- Vector search (semantic)
- BM25 search (keyword)
- Reciprocal Rank Fusion
- Cross-encoder reranking
- Neighbor chunk retrieval

Usage:
    from rag_v2.search_king import SearchKing
    
    search_engine = SearchKing(
        embed_fn=embed_text,
        qdrant_client=client,
        collection_name="rag_collection",
        bm25_index=bm25_index,
        corpus_texts=corpus_texts,
        all_chunks=all_chunks
    )
    
    results = search_engine.search("What is meditation?")
"""

import os
import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any
from collections import defaultdict
from dataclasses import dataclass


# Try to import reranker (optional)
try:
    from sentence_transformers import CrossEncoder
    HAS_RERANKER = True
except ImportError:
    HAS_RERANKER = False


@dataclass
class SearchResult:
    """Represents a search result with metadata."""
    text: str
    score: float
    source: str
    metadata: Dict
    chunk_index: int
    is_neighbor: bool = False
    rerank_score: Optional[float] = None


class SearchKing:
    """
    🔥 Advanced Hybrid Search Engine
    
    Combines multiple search techniques for best results.
    """
    
    def __init__(
        self,
        embed_fn: Callable[[str], List[float]],
        qdrant_client: Any,
        collection_name: str,
        bm25_index: Any,
        corpus_texts: List[str],
        all_chunks: List[Any],
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        use_reranker: bool = True,
        reranker_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    ):
        """
        Initialize SearchKing.
        
        Args:
            embed_fn: Function to embed text
            qdrant_client: Qdrant client instance
            collection_name: Name of the Qdrant collection
            bm25_index: BM25Okapi index
            corpus_texts: List of corpus texts for BM25
            all_chunks: List of all chunk objects
            vector_weight: Weight for vector search (0-1)
            bm25_weight: Weight for BM25 search (0-1)
            use_reranker: Whether to use cross-encoder reranking
            reranker_model: Model name for reranker
        """
        self.embed_fn = embed_fn
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.bm25_index = bm25_index
        self.corpus_texts = corpus_texts
        self.all_chunks = all_chunks
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        
        # Initialize reranker
        self.reranker = None
        if use_reranker and HAS_RERANKER:
            try:
                self.reranker = CrossEncoder(reranker_model)
                print("✅ Reranker loaded!")
            except Exception as e:
                print(f"⚠️ Reranker not available: {e}")
    
    # =========================================================================
    # UTILITY FUNCTIONS
    # =========================================================================
    
    @staticmethod
    def normalize_scores(scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range."""
        if not scores:
            return []
        min_score, max_score = min(scores), max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        v1, v2 = np.array(vec1), np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple tokenization for BM25."""
        return [t for t in re.findall(r'\b\w+\b', text.lower()) if len(t) > 2]
    
    # =========================================================================
    # QUERY ENHANCEMENT
    # =========================================================================
    
    def expand_query(self, query: str) -> List[str]:
        """Simple query expansion using common patterns."""
        expansions = [query]
        
        if "what is" in query.lower():
            expansions.append(query.lower().replace("what is", "explain"))
            expansions.append(query.lower().replace("what is", "describe"))
        
        if "how" in query.lower():
            expansions.append(query.lower().replace("how", "what is the way"))
        
        keywords = ["Krishnamurti", "meditation", "awareness", "thought", "observer"]
        for kw in keywords:
            if kw.lower() in query.lower():
                expansions.append(kw)
        
        return list(set(expansions))
    
    def hyde_search(self, query: str) -> List[float]:
        """
        HyDE: Hypothetical Document Embeddings
        Instead of embedding the question, embed a hypothetical answer.
        """
        hypothetical = f"""
        Krishnamurti addresses this question about {query.lower().replace('?', '')}.
        He speaks about the nature of awareness and the observation of the mind.
        The key insight is that true understanding comes not from analysis or method,
        but from direct perception without the interference of thought.
        """
        return self.embed_fn(hypothetical)
    
    # =========================================================================
    # SEARCH FUNCTIONS
    # =========================================================================
    
    def search_vectors(self, query_embedding: List[float], limit: int = 10) -> List[Dict]:
        """Search vectors in Qdrant."""
        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
        )
        
        formatted = []
        for point in results.points:
            formatted.append({
                "text": point.payload.get("text", ""),
                "score": point.score,
                "source": point.payload.get("source", ""),
                "metadata": point.payload.get("metadata", {}),
                "chunk_index": point.payload.get("metadata", {}).get("chunk_index", 0)
            })
        return formatted
    
    def search_bm25(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search using BM25."""
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []
        
        scores = self.bm25_index.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((int(idx), float(scores[idx])))
        return results
    
    # =========================================================================
    # FUSION & RANKING
    # =========================================================================
    
    @staticmethod
    def reciprocal_rank_fusion(results_lists: List[List[Dict]], k: int = 60) -> Dict[str, float]:
        """Reciprocal Rank Fusion - combines multiple result lists."""
        fused_scores = defaultdict(float)
        
        for results in results_lists:
            for rank, doc in enumerate(results):
                doc_id = doc["text"]
                fused_scores[doc_id] += 1.0 / (k + rank + 1)
        
        return dict(fused_scores)
    
    def rerank_results(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """Rerank results using cross-encoder."""
        if not self.reranker or not results:
            return results[:top_k]
        
        pairs = [[query, r["text"]] for r in results]
        scores = self.reranker.predict(pairs)
        
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])
            result["final_score"] = 0.6 * float(scores[i]) + 0.4 * result.get("score", 0)
        
        ranked = sorted(results, key=lambda x: x["final_score"], reverse=True)
        return ranked[:top_k]
    
    def get_neighbor_chunks(self, chunk_indices: List[int], window: int = 1) -> List[int]:
        """Get neighboring chunk indices for context."""
        neighbors = set()
        for idx in chunk_indices:
            for offset in range(-window, window + 1):
                neighbor_idx = idx + offset
                if 0 <= neighbor_idx < len(self.all_chunks):
                    neighbors.add(neighbor_idx)
        return sorted(neighbors)
    
    # =========================================================================
    # 🔥 MAIN SEARCH FUNCTION
    # =========================================================================
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        use_hyde: bool = True,
        use_query_expansion: bool = True,
        use_reranking: bool = True,
        use_neighbors: bool = True,
        verbose: bool = True
    ) -> List[Dict]:
        """
        🔥 Advanced Hybrid Search with all the bells and whistles!
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_hyde: Use hypothetical document embeddings
            use_query_expansion: Expand query with variations
            use_reranking: Use cross-encoder reranking
            use_neighbors: Include neighboring chunks
            verbose: Print progress
        
        Returns:
            List of search results
        """
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔍 ADVANCED SEARCH: '{query}'")
            print(f"{'='*60}")
        
        all_results = []
        text_metadata = {}
        
        # STEP 1: Query Expansion
        queries = [query]
        if use_query_expansion:
            queries = self.expand_query(query)
            if verbose:
                print(f"📝 Query variations: {len(queries)}")
        
        # STEP 2: Multi-Query Vector Search
        for q in queries:
            q_embedding = self.embed_fn(q)
            vector_results = self.search_vectors(q_embedding, limit=top_k * 2)
            all_results.append(vector_results)
            
            for r in vector_results:
                text_metadata[r["text"]] = {
                    "source": r.get("source", ""),
                    "metadata": r.get("metadata", {}),
                    "chunk_index": r.get("chunk_index", 0)
                }
        
        if verbose:
            print(f"🧠 Vector search: {sum(len(r) for r in all_results)} results")
        
        # STEP 3: HyDE Search
        if use_hyde:
            hyde_embedding = self.hyde_search(query)
            hyde_results = self.search_vectors(hyde_embedding, limit=top_k * 2)
            all_results.append(hyde_results)
            
            for r in hyde_results:
                if r["text"] not in text_metadata:
                    text_metadata[r["text"]] = {
                        "source": r.get("source", ""),
                        "metadata": r.get("metadata", {}),
                        "chunk_index": r.get("chunk_index", 0)
                    }
            
            if verbose:
                print(f"🎯 HyDE search: {len(hyde_results)} results")
        
        # STEP 4: BM25 Search
        bm25_formatted = []
        for q in queries:
            bm25_results = self.search_bm25(q, top_k=top_k * 2)
            for idx, score in bm25_results:
                text = self.corpus_texts[idx]
                bm25_formatted.append({
                    "text": text,
                    "score": score,
                    "chunk_index": idx
                })
                if text not in text_metadata:
                    text_metadata[text] = {
                        "source": self.all_chunks[idx].metadata.get("source", "") if idx < len(self.all_chunks) else "",
                        "metadata": self.all_chunks[idx].metadata if idx < len(self.all_chunks) else {},
                        "chunk_index": idx
                    }
        
        all_results.append(bm25_formatted)
        
        if verbose:
            print(f"🔤 BM25 search: {len(bm25_formatted)} results")
        
        # STEP 5: Reciprocal Rank Fusion
        fused_scores = self.reciprocal_rank_fusion(all_results)
        
        combined_results = []
        for text, rrf_score in fused_scores.items():
            combined_results.append({
                "text": text,
                "score": rrf_score,
                **text_metadata.get(text, {})
            })
        
        combined_results = sorted(combined_results, key=lambda x: x["score"], reverse=True)
        
        if verbose:
            print(f"🔗 Fusion: {len(combined_results)} unique results")
        
        # STEP 6: Neighbor Chunks
        if use_neighbors and combined_results:
            top_indices = [r.get("chunk_index", 0) for r in combined_results[:top_k]]
            neighbor_indices = self.get_neighbor_chunks(top_indices, window=1)
            
            existing_texts = {r["text"] for r in combined_results}
            for idx in neighbor_indices:
                if idx < len(self.all_chunks):
                    chunk = self.all_chunks[idx]
                    if chunk.text not in existing_texts:
                        combined_results.append({
                            "text": chunk.text,
                            "score": 0.1,
                            "source": chunk.metadata.get("source", ""),
                            "metadata": chunk.metadata,
                            "chunk_index": idx,
                            "is_neighbor": True
                        })
            
            if verbose:
                print(f"📍 Added {len(neighbor_indices) - len(top_indices)} neighbor chunks")
        
        # STEP 7: Reranking
        if use_reranking and self.reranker:
            combined_results = self.rerank_results(query, combined_results, top_k=top_k * 2)
            if verbose:
                print(f"⚡ Reranked with cross-encoder")
        
        final_results = combined_results[:top_k]
        
        if verbose:
            print(f"{'='*60}")
            print(f"✅ Returning top {len(final_results)} results")
            print(f"{'='*60}")
        
        return final_results
    
    def get_context(self, question: str, k: int = 3, verbose: bool = True) -> str:
        """Get formatted context for RAG."""
        results = self.search(
            question, 
            top_k=k,
            use_hyde=True,
            use_query_expansion=True,
            use_reranking=self.reranker is not None,
            use_neighbors=True,
            verbose=verbose
        )
        
        if not results:
            return "No relevant context found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            source = os.path.basename(result.get("source", "Unknown"))
            text = result.get("text", "")
            score = result.get("score", 0)
            neighbor_tag = " [NEIGHBOR]" if result.get("is_neighbor") else ""
            context_parts.append(f"[Source {i}: {source} | Score: {score:.3f}{neighbor_tag}]\n{text}")
        
        return "\n\n---\n\n".join(context_parts)


# =============================================================================
# STANDALONE FUNCTIONS (for backward compatibility)
# =============================================================================

def normalize_scores(scores: List[float]) -> List[float]:
    """Normalize scores to 0-1 range."""
    return SearchKing.normalize_scores(scores)


def reciprocal_rank_fusion(results_lists: List[List[Dict]], k: int = 60) -> Dict[str, float]:
    """Reciprocal Rank Fusion - combines multiple result lists."""
    return SearchKing.reciprocal_rank_fusion(results_lists, k)


if __name__ == "__main__":
    print("🔥 SEARCH KING is ready!")
    print("   ✅ Query expansion: ON")
    print("   ✅ HyDE search: ON")
    print("   ✅ Hybrid (Vector + BM25): ON")
    print("   ✅ Reciprocal Rank Fusion: ON")
    print("   ✅ Neighbor chunks: ON")
    print(f"   {'✅' if HAS_RERANKER else '⚠️'} Reranking: {'Available' if HAS_RERANKER else 'Not available (install sentence-transformers)'}")
