#!/usr/bin/env python3
"""
RAG Pipeline: Build Index from PDFs

This script loads all PDFs from the 'documents' folder,
processes them, and stores them in ChromaDB.

Usage:
    python -m rag.pipeline
    
Or run directly:
    python rag/pipeline.py
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.config import PDF_FOLDER
from rag.loader import load_all_pdfs
from rag.processor import process_documents
from rag.embedder import embed_chunks
from rag.store import store_chunks
from rag.query import get_context

def build_index():
    """Build the RAG index from all PDFs in the documents folder."""
    print("\n" + "="*50)
    print("🧘 OSHO RAG PIPELINE")
    print("="*50 + "\n")
    
    # Step 1: Load PDFs
    print("📄 Step 1: Loading PDFs...")
    documents = load_all_pdfs(PDF_FOLDER)
    
    if not documents:
        print("\n⚠️  No documents to process.")
        print(f"   Add PDF files to: {PDF_FOLDER}")
        return False
    
    print(f"   Loaded {len(documents)} PDF(s)\n")
    
    # Step 2: Process documents
    print("🔧 Step 2: Processing documents...")
    chunks = process_documents(documents)
    print(f"   Total chunks: {len(chunks)}\n")
    
    # Step 3: Embed chunks
    print("🧠 Step 3: Generating embeddings...")
    embeddings = embed_chunks(chunks)
    print()
    
    # Step 4: Store in ChromaDB
    print("💾 Step 4: Storing in ChromaDB...")
    store_chunks(chunks, embeddings)
    
    print("\n" + "="*50)
    print("✅ Memory ready!")
    print("="*50 + "\n")
    
    return True

def interactive_query():
    """Interactive query loop."""
    print("\n🔮 Ask Osho anything (type 'exit' to quit)\n")
    
    while True:
        try:
            question = input("You → ")
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("\n🙏 Namaste!")
                break
            
            if not question.strip():
                continue
            
            context = get_context(question)
            print("\n--- Osho Memory ---\n")
            print(context)
            print("\n" + "-"*40 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n🙏 Namaste!")
            break

if __name__ == "__main__":
    success = build_index()
    
    if success:
        interactive_query()
