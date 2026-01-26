# RAG (Retrieval Augmented Generation) Module

This module implements a RAG pipeline for Osho teachings using ChromaDB and OpenAI embeddings.

## Structure

```
rag/
├── __init__.py       # Module exports
├── config.py         # Configuration settings
├── loader.py         # PDF loading utilities
├── processor.py      # Text cleaning & chunking
├── embedder.py       # OpenAI embeddings
├── store.py          # ChromaDB storage
├── query.py          # Query/retrieval functions
├── pipeline.py       # Main build script
├── documents/        # Put your PDFs here!
└── chroma_db/        # Persistent vector storage
```

## Usage

### 1. Add PDF Files

Place all your Osho PDF files in the `documents/` folder:

```
rag/documents/
├── Osho1.pdf
├── Osho2.pdf
└── ...
```

### 2. Build the Index

Run the pipeline to process all PDFs and store them in ChromaDB:

```bash
# From the project root
source venv/bin/activate
python -m rag.pipeline
```

This will:
1. Load all PDFs from `documents/`
2. Clean and chunk the text
3. Generate embeddings using OpenAI
4. Store everything in ChromaDB (persisted to `chroma_db/`)

### 3. Query the Memory

After building, you can query interactively:

```python
from rag import get_context, query_memory

# Get context for a question
context = get_context("What is meditation?")
print(context)

# Get raw results with metadata
results = query_memory("What is enlightenment?", k=5)
for r in results:
    print(r["text"])
    print(r["metadata"])
```

## Environment Variables

Make sure `OPENAI_API_KEY` is set in your `.env` file:

```
OPENAI_API_KEY=sk-your-key-here
```

## Configuration

Edit `config.py` to customize:

- `CHUNK_SIZE`: Number of tokens per chunk (default: 400)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 80)
- `EMBED_MODEL`: OpenAI embedding model (default: text-embedding-3-large)
- `DEFAULT_TOP_K`: Number of results to return (default: 3)
