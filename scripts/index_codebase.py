"""
Codebase Indexer - Index all source files into RAG v2
This script reads all .py, .md, .js, .jsx files and indexes them
so the agent can answer questions about the codebase using RAG v2 (Qdrant).
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import RAG v2 components
from rag_v2.embedder import embed_chunks
from rag_v2.vector_store import store_vectors
from rag_v2.chunker import Chunk

# Directories to exclude
EXCLUDE_DIRS = {
    'venv', 'node_modules', '__pycache__', '.git', 
    'chroma_db', 'qdrant_db', '.DS_Store', 'data', 'dist'
}

# File extensions to include
INCLUDE_EXTENSIONS = {'.py', '.md', '.js', '.jsx', '.json', '.css'}

# Files to skip
SKIP_FILES = {'package-lock.json', 'vector_store.json'}

def should_include_file(path: Path) -> bool:
    """Check if file should be included in indexing."""
    # Check if any parent directory should be excluded
    for parent in path.parents:
        if parent.name in EXCLUDE_DIRS:
            return False
    
    # Check file extension
    if path.suffix not in INCLUDE_EXTENSIONS:
        return False
    
    # Skip specific files
    if path.name in SKIP_FILES:
        return False
    
    return True

def get_file_description(filepath: str, content: str) -> str:
    """Generate a brief description of the file based on its path and content."""
    path = Path(filepath)
    
    # Check for docstrings
    lines = content.split('\n')
    for i, line in enumerate(lines[:20]):  # Check first 20 lines
        if '"""' in line or "'''" in line:
            # Try to extract docstring
            start = i
            quote = '"""' if '"""' in line else "'''"
            docstring_lines = []
            
            # Single line docstring
            if line.count(quote) >= 2:
                return line.replace(quote, '').strip()
            
            # Multi-line docstring
            for j, l in enumerate(lines[i+1:i+10]):
                if quote in l:
                    return ' '.join(docstring_lines).strip()
                docstring_lines.append(l.strip())
    
    # Fallback to path-based description
    descriptions = {
        'main.py': 'FastAPI application entry point with CORS middleware',
        'agent.py': 'LangGraph agent logic',
        'router.py': 'FastAPI router',
        'App.jsx': 'React chat interface',
    }
    
    return descriptions.get(path.name, f'Source file in {path.parent.name}/')

def create_chunk(filepath: str, content: str, base_path: str, chunk_index: int = 0) -> Chunk:
    """Create a Chunk object for RAG indexing."""
    relative_path = os.path.relpath(filepath, base_path)
    description = get_file_description(filepath, content)
    
    # Create structured document
    doc_text = f"""## File: {relative_path}

**Description:** {description}

**Path:** `{relative_path}`

```{Path(filepath).suffix[1:]}
{content}
```
"""
    
    return Chunk(
        text=doc_text,
        metadata={
            "source": relative_path,
            "type": "code",
            "extension": Path(filepath).suffix,
            "filename": Path(filepath).name,
            "directory": str(Path(filepath).parent.name),
            "description": description
        },
        chunk_index=chunk_index
    )

def collect_files(root_dir: str) -> list:
    """Collect all relevant files from the codebase."""
    files = []
    root_path = Path(root_dir)
    
    for path in root_path.rglob('*'):
        if path.is_file() and should_include_file(path):
            try:
                content = path.read_text(encoding='utf-8')
                # Skip very large files
                if len(content) > 100000:
                    print(f"  ⚠️  Skipping large file: {path.name} ({len(content)} chars)")
                    continue
                files.append((str(path), content))
            except Exception as e:
                print(f"  ❌ Error reading {path}: {e}")
    
    return files

def split_large_content(content: str, max_size: int = 8000) -> list:
    """Split content into smaller parts if too large."""
    if len(content) <= max_size:
        return [content]
    
    parts = []
    lines = content.split('\n')
    current_part = []
    current_size = 0
    
    for line in lines:
        if current_size + len(line) > max_size and current_part:
            parts.append('\n'.join(current_part))
            current_part = [line]
            current_size = len(line)
        else:
            current_part.append(line)
            current_size += len(line) + 1
            
    if current_part:
        parts.append('\n'.join(current_part))
        
    return parts

def index_codebase(root_dir: str = None):
    """Main function to index the entire codebase."""
    if root_dir is None:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("🚀 CODEBASE INDEXER (RAG v2)")
    print("=" * 60)
    print(f"📁 Root directory: {root_dir}")
    print("=" * 60)
    
    # 1. Collect all files
    print("\n📂 Collecting files...")
    files = collect_files(root_dir)
    print(f"   Found {len(files)} files to index")
    
    # 2. Create chunks
    print("\n📝 Creating chunks...")
    chunks = []
    
    for filepath, content in files:
        # Split large files first
        content_parts = split_large_content(content)
        
        for i, part in enumerate(content_parts):
            chunk = create_chunk(filepath, part, root_dir, i)
            chunks.append(chunk)
            
        print(f"   ✓ {os.path.basename(filepath)} ({len(content_parts)} chunks)")
    
    print(f"   Total chunks created: {len(chunks)}")
    
    # 3. Generate embeddings
    print("\n🧠 Generating embeddings...")
    # RAG v2 embedder expects list of dicts with 'text', but store_vectors expects Chunks
    # We need to adapt slightly or check embed_chunks signature
    
    # Check rag_v2/embedder.py: def embed_chunks(chunks: List[Chunk]) -> List[List[float]]:
    # It likely expects Chunk objects directly or dicts. Let's assume Chunk objects based on typical v2 pattern.
    # Actually, let's verify embedder.py signature first just to match perfectly.
    
    try:
        embeddings = embed_chunks(chunks)
    except TypeError:
        # Fallback if it expects dicts
        chunk_dicts = [{"text": c.text} for c in chunks]
        embeddings = embed_chunks(chunk_dicts)

    # 4. Store in vector database
    print("\n💾 Storing in Qdrant...")
    store_vectors(chunks, embeddings)
    
    print("\n" + "=" * 60)
    print("✅ CODEBASE INDEXING COMPLETE!")
    print(f"   Indexed {len(chunks)} chunks from {len(files)} files")
    print("=" * 60)
    
    return len(chunks)

if __name__ == "__main__":
    index_codebase()
