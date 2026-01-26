import re
from typing import List
from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter
from .config import CHUNK_SIZE, CHUNK_OVERLAP

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove timestamps like 12:34
    text = re.sub(r'\b\d+:\d+\b', '', text)
    # Remove page numbers
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str, source: str = "") -> List[dict]:
    """Split text into chunks using LlamaIndex TokenTextSplitter."""
    docs = [Document(text=text, metadata={"source": source})]
    
    splitter = TokenTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    nodes = splitter.get_nodes_from_documents(docs)
    
    chunks = []
    for i, node in enumerate(nodes):
        chunks.append({
            "text": node.text,
            "metadata": {
                "source": source,
                "chunk_index": i
            }
        })
    
    return chunks

def process_documents(documents: List[dict]) -> List[dict]:
    """Process multiple documents: clean and chunk each one."""
    all_chunks = []
    
    for doc in documents:
        print(f"  Processing: {doc['filename']}")
        cleaned = clean_text(doc["text"])
        chunks = chunk_text(cleaned, source=doc["filename"])
        all_chunks.extend(chunks)
        print(f"    → {len(chunks)} chunks created")
    
    return all_chunks
