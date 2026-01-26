#!/usr/bin/env python3
"""
RAG v2 Pipeline - Production Grade
Orchestrates the full indexing process
"""
import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_v2.config import PDF_FOLDER, COLLECTION_NAME
from rag_v2.loader import load_documents
from rag_v2.chunker import chunk_documents
from rag_v2.embedder import embed_chunks
from rag_v2.vector_store import store_vectors, get_collection_info
from rag_v2.bm25_index import build_bm25_index
from rag_v2.query import get_context, hybrid_search

def build_index(recreate: bool = True) -> bool:
    """
    Build the complete RAG index.
    
    Args:
        recreate: If True, recreate the entire index
    
    Returns:
        True if successful
    """
    start_time = time.time()
    
    print("\n" + "=" * 60)
    print("🚀 RAG v2 PIPELINE - Production Grade")
    print("=" * 60)
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Documents folder: {PDF_FOLDER}")
    print("=" * 60 + "\n")
    
    # Step 1: Load documents
    print("📄 STEP 1: Loading Documents")
    print("-" * 40)
    documents = load_documents(PDF_FOLDER)
    
    if not documents:
        print("\n⚠️  No documents to process!")
        print(f"   Add files to: {PDF_FOLDER}")
        return False
    
    print(f"\n   ✓ Loaded {len(documents)} document(s)\n")
    
    # Step 2: Chunk documents (semantic chunking)
    print("✂️  STEP 2: Semantic Chunking")
    print("-" * 40)
    chunks = chunk_documents(documents)
    
    if not chunks:
        print("\n⚠️  No chunks created!")
        return False
    
    print()
    
    # Step 3: Generate embeddings
    print("🧠 STEP 3: Generating Embeddings")
    print("-" * 40)
    embeddings = embed_chunks(chunks)
    print()
    
    # Step 4: Store in Qdrant
    print("💾 STEP 4: Storing in Qdrant Vector DB")
    print("-" * 40)
    store_vectors(chunks, embeddings, recreate=recreate)
    print()
    
    # Step 5: Build BM25 index for hybrid search
    print("📚 STEP 5: Building BM25 Index")
    print("-" * 40)
    texts = [chunk.text for chunk in chunks]
    build_bm25_index(texts)
    print()
    
    # Summary
    elapsed = time.time() - start_time
    print("=" * 60)
    print("✅ INDEXING COMPLETE!")
    print("=" * 60)
    print(f"   Documents processed: {len(documents)}")
    print(f"   Chunks created: {len(chunks)}")
    print(f"   Vectors stored: {len(embeddings)}")
    print(f"   Time elapsed: {elapsed:.2f}s")
    print()
    
    # Show collection info
    info = get_collection_info()
    print("📊 Collection Info:")
    print(f"   Name: {info.get('name', 'N/A')}")
    print(f"   Points: {info.get('points_count', 'N/A')}")
    print(f"   Status: {info.get('status', 'N/A')}")
    print("=" * 60 + "\n")
    
    return True

def test_search(query: str = "What is meditation?"):
    """Test the search functionality."""
    print("\n" + "=" * 60)
    print("🧪 TEST SEARCH")
    print("=" * 60)
    
    results = hybrid_search(query, top_k=3)
    
    print(f"\n📝 Query: '{query}'")
    print(f"   Found {len(results)} results\n")
    
    for i, result in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Score: {result.get('score', 0):.4f}")
        print(f"Source: {result.get('source', 'Unknown')}")
        print(f"Text: {result.get('text', '')[:200]}...")
        print()

def interactive_mode():
    """Interactive query mode."""
    print("\n" + "=" * 60)
    print("🔮 INTERACTIVE MODE")
    print("=" * 60)
    print("Type your questions (or 'exit' to quit)\n")
    
    while True:
        try:
            query = input("You → ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not query:
                continue
            
            context = get_context(query, k=3)
            
            print("\n--- Retrieved Context ---")
            print(context if context else "No relevant context found.")
            print("-" * 40 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG v2 Pipeline")
    parser.add_argument("--build", action="store_true", help="Build the index")
    parser.add_argument("--test", action="store_true", help="Run test search")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--query", "-q", type=str, help="Single query")
    
    args = parser.parse_args()
    
    if args.build:
        build_index()
    elif args.test:
        test_search()
    elif args.interactive:
        interactive_mode()
    elif args.query:
        context = get_context(args.query)
        print(context)
    else:
        # Default: build and test
        if build_index():
            test_search()
