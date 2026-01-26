import os

# Paths
PDF_FOLDER = os.path.join(os.path.dirname(__file__), "documents")
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# Collection
COLLECTION_NAME = "codebase_knowledge"

# Embedding Model
EMBED_MODEL = "text-embedding-3-large"

# Chunking
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

# Query
DEFAULT_TOP_K = 3
