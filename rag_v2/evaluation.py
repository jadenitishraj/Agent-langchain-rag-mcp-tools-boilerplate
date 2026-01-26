"""
RAG Evaluation using RAGAS-inspired metrics
(Lightweight version that doesn't require the full RAGAS package due to Python 3.14 issues)

Metrics:
1. Context Relevancy - How relevant is the retrieved context to the question?
2. Answer Faithfulness - Is the answer grounded in the context?
3. Answer Relevancy - Does the answer actually address the question?
"""
import os
import sys
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_v2.query import get_context, hybrid_search
from rag_v2.embedder import embed_text
import numpy as np

@dataclass
class EvalResult:
    """Holds evaluation results for a single test case."""
    question: str
    context: str
    answer: str
    expected_answer: str
    context_relevancy: float
    answer_relevancy: float
    faithfulness: float
    overall_score: float

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def calculate_context_relevancy(question: str, context: str) -> float:
    """
    Measure how relevant the retrieved context is to the question.
    Uses embedding similarity.
    """
    if not context:
        return 0.0
    
    q_emb = embed_text(question)
    c_emb = embed_text(context[:2000])  # Limit context length
    
    return max(0, cosine_similarity(q_emb, c_emb))

def calculate_answer_relevancy(question: str, answer: str) -> float:
    """
    Measure how well the answer addresses the question.
    Uses embedding similarity.
    """
    if not answer:
        return 0.0
    
    q_emb = embed_text(question)
    a_emb = embed_text(answer[:2000])
    
    return max(0, cosine_similarity(q_emb, a_emb))

def calculate_faithfulness(context: str, answer: str) -> float:
    """
    Measure if the answer is grounded in the context.
    Uses embedding similarity and keyword overlap.
    """
    if not context or not answer:
        return 0.0
    
    # Embedding similarity
    c_emb = embed_text(context[:2000])
    a_emb = embed_text(answer[:2000])
    emb_sim = max(0, cosine_similarity(c_emb, a_emb))
    
    # Keyword overlap
    import re
    context_words = set(re.findall(r'\b\w+\b', context.lower()))
    answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
    
    # Remove common words
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                 'it', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
                 'she', 'we', 'they', 'and', 'or', 'but', 'if', 'then'}
    
    context_words -= stopwords
    answer_words -= stopwords
    
    if not answer_words:
        return emb_sim
    
    overlap = len(context_words & answer_words) / len(answer_words)
    
    # Combine embedding similarity and keyword overlap
    return (emb_sim * 0.7) + (overlap * 0.3)

def evaluate_rag(
    question: str,
    answer: str,
    expected_answer: str = None
) -> EvalResult:
    """
    Evaluate a single RAG response.
    
    Args:
        question: The input question
        answer: The generated answer
        expected_answer: Optional ground truth answer
    
    Returns:
        EvalResult with all metrics
    """
    # Get context
    context = get_context(question, k=3)
    
    # Calculate metrics
    context_relevancy = calculate_context_relevancy(question, context)
    answer_relevancy = calculate_answer_relevancy(question, answer)
    faithfulness = calculate_faithfulness(context, answer)
    
    # Overall score (weighted average)
    overall = (
        context_relevancy * 0.3 +
        answer_relevancy * 0.3 +
        faithfulness * 0.4
    )
    
    return EvalResult(
        question=question,
        context=context[:500] + "..." if len(context) > 500 else context,
        answer=answer[:500] + "..." if len(answer) > 500 else answer,
        expected_answer=expected_answer or "",
        context_relevancy=context_relevancy,
        answer_relevancy=answer_relevancy,
        faithfulness=faithfulness,
        overall_score=overall
    )

# Test cases for evaluation
TEST_CASES = [
    {
        "question": "What does Osho say about fixed ideas?",
        "expected_keywords": ["fixed", "idea", "change", "mind"]
    },
    {
        "question": "Tell me about the Picasso story",
        "expected_keywords": ["picasso", "portrait", "painting", "dead"]
    },
    {
        "question": "What is the story about bhindi?",
        "expected_keywords": ["bhindi", "mullah", "king", "vegetable"]
    },
    {
        "question": "What does Heraclitus say about the river?",
        "expected_keywords": ["river", "step", "twice", "change"]
    },
    {
        "question": "Why do people create fixed ideas about others?",
        "expected_keywords": ["fixed", "idea", "photograph", "change"]
    }
]

def run_evaluation(test_cases: List[Dict] = None, verbose: bool = True) -> Dict:
    """
    Run evaluation on test cases.
    
    Returns:
        Summary of evaluation results
    """
    if test_cases is None:
        test_cases = TEST_CASES
    
    print("\n" + "=" * 60)
    print("📊 RAG EVALUATION")
    print("=" * 60 + "\n")
    
    results = []
    
    for i, test in enumerate(test_cases, 1):
        question = test["question"]
        
        if verbose:
            print(f"🧪 Test {i}/{len(test_cases)}: {question}")
        
        # Get context and check if it contains expected keywords
        context = get_context(question, k=3)
        
        # Simulate an answer (in real eval, you'd use the actual agent)
        # For now, we just check retrieval quality
        expected_keywords = test.get("expected_keywords", [])
        context_lower = context.lower()
        
        keyword_hits = sum(1 for kw in expected_keywords if kw in context_lower)
        keyword_score = keyword_hits / len(expected_keywords) if expected_keywords else 1.0
        
        # Calculate retrieval metrics
        context_relevancy = calculate_context_relevancy(question, context)
        
        results.append({
            "question": question,
            "context_relevancy": context_relevancy,
            "keyword_score": keyword_score,
            "keywords_found": keyword_hits,
            "keywords_expected": len(expected_keywords)
        })
        
        if verbose:
            print(f"   Context Relevancy: {context_relevancy:.3f}")
            print(f"   Keyword Score: {keyword_score:.3f} ({keyword_hits}/{len(expected_keywords)})")
            print()
    
    # Calculate averages
    avg_relevancy = sum(r["context_relevancy"] for r in results) / len(results)
    avg_keyword = sum(r["keyword_score"] for r in results) / len(results)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_tests": len(results),
        "avg_context_relevancy": avg_relevancy,
        "avg_keyword_score": avg_keyword,
        "results": results
    }
    
    print("=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    print(f"   Tests run: {len(results)}")
    print(f"   Avg Context Relevancy: {avg_relevancy:.3f}")
    print(f"   Avg Keyword Score: {avg_keyword:.3f}")
    print("=" * 60 + "\n")
    
    return summary

def save_evaluation_results(results: Dict, filepath: str = None):
    """Save evaluation results to JSON."""
    if filepath is None:
        filepath = os.path.join(
            os.path.dirname(__file__),
            "evaluation_results.json"
        )
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {filepath}")

if __name__ == "__main__":
    results = run_evaluation(verbose=True)
    save_evaluation_results(results)
