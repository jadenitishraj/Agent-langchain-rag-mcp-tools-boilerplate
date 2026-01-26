"""
Semantic Chunking - Groups semantically related sentences together
Falls back to token-based chunking if needed
"""
import re
from typing import List, Dict
from dataclasses import dataclass

from .config import (
    CHUNK_MIN_SIZE, 
    CHUNK_MAX_SIZE, 
    SIMILARITY_THRESHOLD,
    FALLBACK_CHUNK_SIZE,
    FALLBACK_CHUNK_OVERLAP
)

@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    metadata: Dict
    chunk_index: int

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex."""
    # Clean the text first
    text = re.sub(r'\s+', ' ', text)
    
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove timestamps
    text = re.sub(r'\b\d+:\d+\b', '', text)
    # Remove page numbers (standalone numbers on lines)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def simple_semantic_chunk(sentences: List[str], embedder_fn=None) -> List[str]:
    """
    Create semantic chunks by grouping text.
    Handles both proper sentences and transcripts without punctuation.
    """
    if not sentences:
        return []
    
    # Join all sentences back together
    full_text = ' '.join(sentences)
    
    # If no proper sentence endings, just split by size
    has_punctuation = any(full_text.count(p) > 3 for p in ['.', '!', '?'])
    
    if not has_punctuation:
        # Split by newlines/paragraphs first, then by size
        paragraphs = full_text.split('\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            if current_size + len(para) > CHUNK_MAX_SIZE and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(para)
            current_size += len(para)
            
            # Force split at 60% to create more chunks
            if current_size >= CHUNK_MAX_SIZE * 0.6:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    # Original logic for properly punctuated text
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_len = len(sentence)
        
        if current_size + sentence_len > CHUNK_MAX_SIZE and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
        
        current_chunk.append(sentence)
        current_size += sentence_len + 1
        
        if current_size >= CHUNK_MAX_SIZE * 0.6 and sentence.endswith(('.', '!', '?')):
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def merge_small_chunks(chunks: List[str]) -> List[str]:
    """Merge chunks that are too small."""
    if not chunks:
        return []
    
    merged = []
    current = chunks[0]
    
    for chunk in chunks[1:]:
        if len(current) < CHUNK_MIN_SIZE:
            current += ' ' + chunk
        else:
            merged.append(current)
            current = chunk
    
    merged.append(current)
    return merged

def chunk_document(content: str, source: str, embedder_fn=None) -> List[Chunk]:
    """
    Chunk a document using size-based strategy.
    For transcripts without punctuation, splits by character count.
    """
    # Remove timestamps first
    content = re.sub(r'\b\d+:\d+\b', '', content)
    
    # Check if text has proper punctuation
    has_punctuation = any(content.count(p) > 5 for p in ['.', '!', '?'])
    
    if has_punctuation:
        # Use sentence-based chunking
        cleaned = clean_text(content)
        sentences = split_into_sentences(cleaned)
        raw_chunks = simple_semantic_chunk(sentences, embedder_fn)
    else:
        # For transcripts: split by lines, then group by size
        lines = content.split('\n')
        lines = [l.strip() for l in lines if l.strip()]
        
        raw_chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_len = len(line)
            
            if current_size + line_len > CHUNK_MAX_SIZE and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk_text = re.sub(r'\s+', ' ', chunk_text).strip()  # Clean
                raw_chunks.append(chunk_text)
                current_chunk = []
                current_size = 0
            
            current_chunk.append(line)
            current_size += line_len
            
            # Force split at target size
            if current_size >= CHUNK_MAX_SIZE * 0.6:
                chunk_text = ' '.join(current_chunk)
                chunk_text = re.sub(r'\s+', ' ', chunk_text).strip()
                raw_chunks.append(chunk_text)
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_text = re.sub(r'\s+', ' ', chunk_text).strip()
            raw_chunks.append(chunk_text)
    
    # Merge small chunks
    final_chunks = merge_small_chunks(raw_chunks)
    
    # Convert to Chunk objects
    chunks = []
    for i, text in enumerate(final_chunks):
        if text and len(text) > 50:  # Skip tiny chunks
            chunks.append(Chunk(
                text=text,
                metadata={
                    "source": source,
                    "chunk_index": i,
                    "total_chunks": len(final_chunks),
                    "char_count": len(text)
                },
                chunk_index=i
            ))
    
    return chunks

def chunk_documents(documents: List, embedder_fn=None) -> List[Chunk]:
    """
    Chunk multiple documents.
    
    Args:
        documents: List of Document objects
        embedder_fn: Optional embedding function
    
    Returns:
        List of all chunks from all documents
    """
    all_chunks = []
    
    for doc in documents:
        print(f"   Chunking: {doc.metadata.get('filename', 'unknown')}")
        chunks = chunk_document(doc.content, doc.source, embedder_fn)
        all_chunks.extend(chunks)
        print(f"   ✓ Created {len(chunks)} chunks")
    
    print(f"\n   Total chunks: {len(all_chunks)}")
    return all_chunks
