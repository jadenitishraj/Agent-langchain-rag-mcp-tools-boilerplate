# =============================================================================
# 🔥 ADVANCED CHUNKING ENGINE - THE CHUNKING KING
# =============================================================================
"""
Advanced document chunking with multiple strategies:
- Semantic chunking (embedding-based)
- Speaker-based (for Q&A/dialogues)
- Sentence-based
- Recursive
- Fixed with overlap
- Auto-detection

Usage:
    from rag_v2.chunker_king import advanced_chunk_document, chunk_all_documents
    
    chunks = advanced_chunk_document(content, source, strategy="auto")
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from collections import defaultdict


@dataclass
class Chunk:
    """Represents a text chunk with rich metadata."""
    text: str
    metadata: Dict
    chunk_index: int
    parent_text: Optional[str] = None
    neighbors: List[int] = field(default_factory=list)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    text = re.sub(r'\b\d+:\d+\b', '', text)  # Remove timestamps
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Remove page numbers
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    text = re.sub(r'\s+', ' ', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    v1, v2 = np.array(vec1), np.array(vec2)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


# =============================================================================
# CHUNKING STRATEGIES
# =============================================================================

def chunk_fixed_size(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Strategy 1: Fixed-size chunking with overlap.
    Simple but fast. Good for uniform content.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.5:
                end = start + last_period + 1
                chunk = text[start:end]
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return [c for c in chunks if len(c) > 50]


def chunk_sentence_based(text: str, min_size: int = 200, max_size: int = 800) -> List[str]:
    """
    Strategy 2: Sentence-based chunking.
    Respects sentence boundaries. Good for essays/books.
    """
    sentences = split_into_sentences(text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_len = len(sentence)
        
        if current_size + sentence_len > max_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
        
        current_chunk.append(sentence)
        current_size += sentence_len + 1
        
        if current_size >= min_size and sentence.endswith(('.', '!', '?')):
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return [c for c in chunks if len(c) > 50]


def chunk_by_speaker(text: str) -> List[str]:
    """
    Strategy 3: Speaker-based chunking for dialogues.
    Keeps each speaker's complete thought together.
    Best for Krishnamurti Q&A format!
    """
    speaker_pattern = r'(Questioner\s*:|Krishnamurti\s*:|Q\s*:|K\s*:)'
    parts = re.split(speaker_pattern, text)
    
    chunks = []
    current_speaker = ""
    
    for i, part in enumerate(parts):
        if re.match(speaker_pattern, part):
            current_speaker = part
        elif part.strip():
            chunk = f"{current_speaker} {part.strip()}"
            chunks.append(chunk)
    
    return [c for c in chunks if len(c) > 50]


def chunk_by_paragraph(text: str, max_size: int = 1000) -> List[str]:
    """
    Strategy 4: Paragraph-based chunking.
    Good for structured documents with clear paragraphs.
    """
    paragraphs = text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para_len = len(para)
        
        if current_size + para_len > max_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = []
            current_size = 0
        
        current_chunk.append(para)
        current_size += para_len
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return [c for c in chunks if len(c) > 50]


def chunk_semantic(
    text: str, 
    embed_fn: Callable[[str], List[float]],
    threshold: float = 0.75, 
    min_size: int = 200, 
    max_size: int = 1000
) -> List[str]:
    """
    Strategy 5: Semantic chunking using embeddings.
    Splits when topic changes significantly.
    Most intelligent but slowest!
    
    Args:
        text: Document text
        embed_fn: Function to embed text (e.g., OpenAI embeddings)
        threshold: Similarity threshold for splitting
        min_size: Minimum chunk size
        max_size: Maximum chunk size
    """
    sentences = split_into_sentences(text)
    
    if len(sentences) < 2:
        return [text] if len(text) > 50 else []
    
    print("   🧠 Computing sentence embeddings for semantic chunking...")
    
    embeddings = []
    batch_size = 20
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        batch_embeddings = [embed_fn(s) for s in batch]
        embeddings.extend(batch_embeddings)
    
    chunks = []
    current_chunk = [sentences[0]]
    current_size = len(sentences[0])
    
    for i in range(1, len(sentences)):
        similarity = cosine_similarity(embeddings[i-1], embeddings[i])
        sentence_len = len(sentences[i])
        
        should_split = (
            similarity < threshold and 
            current_size >= min_size
        ) or (current_size + sentence_len > max_size)
        
        if should_split and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentences[i]]
            current_size = sentence_len
        else:
            current_chunk.append(sentences[i])
            current_size += sentence_len
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return [c for c in chunks if len(c) > 50]


def chunk_recursive(text: str, max_size: int = 800) -> List[str]:
    """
    Strategy 6: Recursive chunking.
    Tries multiple delimiters in order: \\n\\n → \\n → . → space
    Good for mixed content.
    """
    separators = ['\n\n', '\n', '. ', ' ']
    
    def split_recursive(text: str, separators: List[str]) -> List[str]:
        if not separators or len(text) <= max_size:
            return [text] if text.strip() else []
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        parts = text.split(separator)
        
        chunks = []
        current = []
        current_len = 0
        
        for part in parts:
            part_len = len(part) + len(separator)
            
            if current_len + part_len > max_size and current:
                chunk_text = separator.join(current)
                if len(chunk_text) > max_size:
                    chunks.extend(split_recursive(chunk_text, remaining_separators))
                else:
                    chunks.append(chunk_text)
                current = []
                current_len = 0
            
            current.append(part)
            current_len += part_len
        
        if current:
            chunk_text = separator.join(current)
            if len(chunk_text) > max_size:
                chunks.extend(split_recursive(chunk_text, remaining_separators))
            else:
                chunks.append(chunk_text)
        
        return chunks
    
    return [c.strip() for c in split_recursive(text, separators) if len(c.strip()) > 50]


# =============================================================================
# OVERLAP & MERGING
# =============================================================================

def add_overlap(chunks: List[str], overlap_chars: int = 100) -> List[str]:
    """Add overlap between consecutive chunks."""
    if len(chunks) < 2:
        return chunks
    
    overlapped = [chunks[0]]
    
    for i in range(1, len(chunks)):
        prev_end = chunks[i-1][-overlap_chars:] if len(chunks[i-1]) > overlap_chars else chunks[i-1]
        overlapped.append(f"{prev_end}... {chunks[i]}")
    
    return overlapped


def merge_small_chunks(chunks: List[str], min_size: int = 200) -> List[str]:
    """Merge chunks that are too small with neighbors."""
    if not chunks:
        return []
    
    merged = []
    current = chunks[0]
    
    for chunk in chunks[1:]:
        if len(current) < min_size:
            current += ' ' + chunk
        else:
            merged.append(current)
            current = chunk
    
    merged.append(current)
    return merged


# =============================================================================
# PARENT-CHILD CHUNKING
# =============================================================================

def chunk_parent_child(
    text: str, 
    parent_size: int = 2000, 
    child_size: int = 500
) -> Tuple[List[str], List[str], Dict[int, int]]:
    """
    Strategy 7: Parent-Child chunking.
    Creates large parent chunks and small child chunks.
    Search on children, return parent for context!
    """
    parents = chunk_fixed_size(text, chunk_size=parent_size, overlap=200)
    
    all_children = []
    child_to_parent = {}
    
    for parent_idx, parent in enumerate(parents):
        children = chunk_sentence_based(parent, min_size=200, max_size=child_size)
        for child in children:
            child_idx = len(all_children)
            all_children.append(child)
            child_to_parent[child_idx] = parent_idx
    
    return parents, all_children, child_to_parent


# =============================================================================
# 🔥 THE CHUNKING KING - MAIN FUNCTION
# =============================================================================

def advanced_chunk_document(
    content: str, 
    source: str,
    strategy: str = "auto",
    embed_fn: Optional[Callable[[str], List[float]]] = None,
    use_overlap: bool = True,
    overlap_chars: int = 100,
    min_chunk_size: int = 200,
    max_chunk_size: int = 800,
    semantic_threshold: float = 0.75,
    verbose: bool = True
) -> List[Chunk]:
    """
    🔥 Advanced Document Chunking with multiple strategies!
    
    Args:
        content: Document text
        source: Source file path
        strategy: "auto", "semantic", "speaker", "sentence", "recursive", "fixed", "paragraph"
        embed_fn: Embedding function for semantic chunking
        use_overlap: Whether to add overlap between chunks
        overlap_chars: Number of overlap characters
        min_chunk_size: Minimum chunk size
        max_chunk_size: Maximum chunk size
        semantic_threshold: Similarity threshold for semantic chunking
        verbose: Print progress
    
    Returns:
        List of Chunk objects
    """
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"✂️ ADVANCED CHUNKING: {source}")
        print(f"{'='*60}")
    
    content = re.sub(r'\b\d+:\d+\b', '', content)
    cleaned = clean_text(content)
    
    # AUTO-DETECT BEST STRATEGY
    if strategy == "auto":
        has_speakers = bool(re.search(r'(Questioner\s*:|Krishnamurti\s*:|Q\s*:|K\s*:)', content))
        has_punctuation = sum(content.count(p) for p in ['.', '!', '?']) > 10
        has_paragraphs = content.count('\n\n') > 5
        
        if has_speakers:
            strategy = "speaker"
            if verbose:
                print(f"📝 Auto-detected: Speaker-based (Q&A format)")
        elif has_punctuation and len(cleaned) > 5000 and embed_fn:
            strategy = "semantic"
            if verbose:
                print(f"📝 Auto-detected: Semantic (long well-formatted text)")
        elif has_paragraphs:
            strategy = "recursive"
            if verbose:
                print(f"📝 Auto-detected: Recursive (structured paragraphs)")
        else:
            strategy = "sentence"
            if verbose:
                print(f"📝 Auto-detected: Sentence-based (default)")
    
    if verbose:
        print(f"🔧 Using strategy: {strategy}")
    
    # APPLY STRATEGY
    if strategy == "semantic":
        if embed_fn is None:
            raise ValueError("embed_fn required for semantic chunking")
        raw_chunks = chunk_semantic(cleaned, embed_fn, semantic_threshold, min_chunk_size, max_chunk_size)
    elif strategy == "speaker":
        raw_chunks = chunk_by_speaker(content)
        split_chunks = []
        for chunk in raw_chunks:
            if len(chunk) > max_chunk_size:
                split_chunks.extend(chunk_sentence_based(chunk, min_chunk_size, max_chunk_size))
            else:
                split_chunks.append(chunk)
        raw_chunks = split_chunks
    elif strategy == "sentence":
        raw_chunks = chunk_sentence_based(cleaned, min_chunk_size, max_chunk_size)
    elif strategy == "recursive":
        raw_chunks = chunk_recursive(cleaned, max_chunk_size)
    elif strategy == "fixed":
        raw_chunks = chunk_fixed_size(cleaned, max_chunk_size, overlap_chars)
    elif strategy == "paragraph":
        raw_chunks = chunk_by_paragraph(cleaned, max_chunk_size)
    else:
        raw_chunks = chunk_sentence_based(cleaned, min_chunk_size, max_chunk_size)
    
    if verbose:
        print(f"   ✓ Created {len(raw_chunks)} raw chunks")
    
    # MERGE SMALL CHUNKS
    merged_chunks = merge_small_chunks(raw_chunks, min_chunk_size)
    
    if verbose and len(merged_chunks) != len(raw_chunks):
        print(f"   ✓ Merged to {len(merged_chunks)} chunks")
    
    # ADD OVERLAP
    if use_overlap and strategy != "fixed":
        final_texts = add_overlap(merged_chunks, overlap_chars)
        if verbose:
            print(f"   ✓ Added {overlap_chars} char overlap")
    else:
        final_texts = merged_chunks
    
    # CREATE CHUNK OBJECTS
    chunks = []
    for i, text in enumerate(final_texts):
        if text and len(text) > 50:
            neighbors = []
            if i > 0:
                neighbors.append(i - 1)
            if i < len(final_texts) - 1:
                neighbors.append(i + 1)
            
            chunks.append(Chunk(
                text=text,
                metadata={
                    "source": source,
                    "chunk_index": i,
                    "total_chunks": len(final_texts),
                    "char_count": len(text),
                    "strategy": strategy,
                    "has_overlap": use_overlap
                },
                chunk_index=i,
                neighbors=neighbors
            ))
    
    if verbose:
        avg_size = sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0
        print(f"   ✓ Final: {len(chunks)} chunks (avg {avg_size:.0f} chars)")
        print(f"{'='*60}")
    
    return chunks


def chunk_all_documents(
    documents: List,
    strategy: str = "auto",
    embed_fn: Optional[Callable[[str], List[float]]] = None,
    use_overlap: bool = True,
    verbose: bool = True
) -> List[Chunk]:
    """Process multiple documents with advanced chunking."""
    
    print("\n" + "🔥" * 20)
    print("CHUNKING KING - Processing Documents")
    print("🔥" * 20)
    
    all_chunks = []
    
    for doc in documents:
        chunks = advanced_chunk_document(
            content=doc.content,
            source=doc.source,
            strategy=strategy,
            embed_fn=embed_fn,
            use_overlap=use_overlap,
            verbose=verbose
        )
        all_chunks.extend(chunks)
        print(f"   ✓ {doc.metadata.get('filename', 'Unknown')}: {len(chunks)} chunks")
    
    print(f"\n{'='*60}")
    print(f"✅ TOTAL: {len(all_chunks)} chunks from {len(documents)} documents")
    print(f"{'='*60}")
    
    return all_chunks


if __name__ == "__main__":
    print("🔥 CHUNKING KING is ready!")
    print("   ✅ Semantic chunking (embedding-based)")
    print("   ✅ Speaker-based (for Q&A)")
    print("   ✅ Sentence-based")
    print("   ✅ Recursive")
    print("   ✅ Fixed with overlap")
    print("   ✅ Auto-detection")
