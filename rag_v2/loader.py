"""
Document Loader - Supports multiple file formats
Uses pdfplumber for PDFs (unstructured has Python 3.14 issues)
"""
import os
import pdfplumber
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Document:
    """Represents a loaded document with metadata."""
    content: str
    metadata: Dict
    source: str
    doc_type: str

def load_pdf(filepath: str) -> Document:
    """Load a single PDF file."""
    text = ""
    page_count = 0
    
    try:
        with pdfplumber.open(filepath) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
    except Exception as e:
        print(f"  ⚠️  Error loading {filepath}: {e}")
        return None
    
    return Document(
        content=text.strip(),
        metadata={
            "page_count": page_count,
            "file_size": os.path.getsize(filepath),
            "filename": os.path.basename(filepath)
        },
        source=filepath,
        doc_type="pdf"
    )

def load_text_file(filepath: str) -> Document:
    """Load a plain text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  ⚠️  Error loading {filepath}: {e}")
        return None
    
    return Document(
        content=content,
        metadata={
            "file_size": os.path.getsize(filepath),
            "filename": os.path.basename(filepath)
        },
        source=filepath,
        doc_type="txt"
    )

def load_documents(folder_path: str) -> List[Document]:
    """
    Load all supported documents from a folder.
    Supports: PDF, TXT, MD
    """
    documents = []
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"📁 Created folder: {folder_path}")
        print("   Please add your documents and run again.")
        return documents
    
    # Supported file extensions and their loaders
    loaders = {
        '.pdf': load_pdf,
        '.txt': load_text_file,
        '.md': load_text_file,
    }
    
    files = os.listdir(folder_path)
    supported_files = [
        f for f in files 
        if any(f.lower().endswith(ext) for ext in loaders.keys())
    ]
    
    if not supported_files:
        print(f"⚠️  No supported documents found in {folder_path}")
        print(f"   Supported formats: {', '.join(loaders.keys())}")
        return documents
    
    print(f"📄 Found {len(supported_files)} document(s)")
    
    for filename in supported_files:
        filepath = os.path.join(folder_path, filename)
        ext = os.path.splitext(filename)[1].lower()
        
        print(f"   Loading: {filename}")
        loader = loaders.get(ext)
        
        if loader:
            doc = loader(filepath)
            if doc:
                documents.append(doc)
                print(f"   ✓ Loaded ({len(doc.content)} chars)")
    
    return documents
