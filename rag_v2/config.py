"""
RAG v2 Configuration - Production Grade Settings
"""
import os

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = os.path.dirname(__file__)
PDF_FOLDER = os.path.join(BASE_DIR, "documents")
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_db")

# =============================================================================
# COLLECTION SETTINGS
# =============================================================================
COLLECTION_NAME = "codebase_knowledge"

# =============================================================================
# EMBEDDING SETTINGS
# =============================================================================
# Using OpenAI embeddings (production-ready, high quality)
# For fully local: switch to sentence-transformers when Python 3.14 support improves
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIMENSION = 3072  # Dimension of text-embedding-3-large

# =============================================================================
# CHUNKING SETTINGS (Semantic Chunking)
# =============================================================================
# Semantic chunking groups semantically related sentences together
CHUNK_MIN_SIZE = 200        # Minimum characters per chunk
CHUNK_MAX_SIZE = 800        # Maximum characters per chunk
SIMILARITY_THRESHOLD = 0.75  # Threshold for semantic grouping (0-1)

# Fallback to token-based if semantic fails
FALLBACK_CHUNK_SIZE = 400
FALLBACK_CHUNK_OVERLAP = 80

# =============================================================================
# SEARCH SETTINGS
# =============================================================================
# Hybrid search weights
VECTOR_WEIGHT = 0.7         # Weight for vector similarity
BM25_WEIGHT = 0.3           # Weight for keyword (BM25) similarity

# Retrieval settings
DEFAULT_TOP_K = 5           # Initial retrieval count
RERANK_TOP_K = 3            # Final count after reranking

# =============================================================================
# QDRANT SETTINGS
# =============================================================================
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
USE_LOCAL_QDRANT = True # Use local file-based Qdrant (no server needed)

# =============================================================================
# QUANTIZATION (Storage optimization)
# =============================================================================
ENABLE_QUANTIZATION = True  # Enable scalar quantization for 4x smaller storage
