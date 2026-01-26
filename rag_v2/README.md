# RAG v2 - Production Grade

A production-ready Retrieval-Augmented Generation system with enterprise features.

## 🚀 Features

| Feature | Implementation |
|---------|----------------|
| **Vector Database** | Qdrant (local file-based, no server needed) |
| **Chunking** | Semantic chunking with smart sentence grouping |
| **Embeddings** | OpenAI text-embedding-3-large (3072 dims) |
| **Search** | Hybrid (Vector + BM25 keyword search) |
| **Reranking** | Relevance scoring + diversity (MMR-style) |
| **Storage** | INT8 quantization (4x smaller) |
| **Evaluation** | RAGAS-inspired metrics |

## 📁 Structure

```
rag_v2/
├── __init__.py       # Module exports
├── config.py         # All configuration settings
├── loader.py         # Document loading (PDF, TXT, MD)
├── chunker.py        # Semantic chunking
├── embedder.py       # OpenAI embeddings with batching
├── vector_store.py   # Qdrant vector database
├── bm25_index.py     # BM25 keyword index
├── reranker.py       # Result reranking + diversity
├── query.py          # Hybrid search engine
├── pipeline.py       # Indexing orchestrator
├── evaluation.py     # RAGAS-style evaluation
└── documents/        # Put your documents here
```

## 🛠️ Usage

### 1. Build the Index

```bash
# From project root
./venv/bin/python -m rag_v2.pipeline --build
```

### 2. Test Search

```bash
./venv/bin/python -m rag_v2.pipeline --test
```

### 3. Interactive Mode

```bash
./venv/bin/python -m rag_v2.pipeline --interactive
```

### 4. Single Query

```bash
./venv/bin/python -m rag_v2.pipeline --query "What is meditation?"
```

### 5. Run Evaluation

```bash
./venv/bin/python -m rag_v2.evaluation
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Chunking
CHUNK_MIN_SIZE = 100        # Min chars per chunk
CHUNK_MAX_SIZE = 1500       # Max chars per chunk

# Search
VECTOR_WEIGHT = 0.7         # Vector similarity weight
BM25_WEIGHT = 0.3           # Keyword match weight
DEFAULT_TOP_K = 5           # Initial retrieval count
RERANK_TOP_K = 3            # Final count after reranking

# Storage
ENABLE_QUANTIZATION = True  # INT8 quantization
```

## 🔄 How Hybrid Search Works

```
User Query
    │
    ├─→ Vector Search (semantic similarity)
    │   └─→ Embeds query, finds similar vectors in Qdrant
    │
    ├─→ BM25 Search (keyword matching)
    │   └─→ Tokenizes query, matches against BM25 index
    │
    └─→ Combine Results (Reciprocal Rank Fusion)
        │
        └─→ Rerank (relevance scoring)
            │
            └─→ Diversity Filter (remove redundancy)
                │
                └─→ Final Top-K Results
```

## 📊 Evaluation Metrics

1. **Context Relevancy**: How relevant is retrieved context to the question?
2. **Answer Relevancy**: Does the answer address the question?
3. **Faithfulness**: Is the answer grounded in the context?
4. **Keyword Score**: Are expected keywords found in context?

## 🆚 v1 vs v2 Comparison

| Feature | RAG v1 | RAG v2 |
|---------|--------|--------|
| Vector DB | JSON file | Qdrant |
| Chunking | Token-based | Semantic |
| Search | Vector only | Hybrid (Vector + BM25) |
| Reranking | None | Yes |
| Diversity | None | MMR-style |
| Quantization | None | INT8 (4x smaller) |
| Evaluation | Manual | RAGAS-style metrics |
| Multi-format | PDF only | PDF, TXT, MD |

## 🎯 Production Checklist

- [x] Persistent vector storage (Qdrant)
- [x] Hybrid search (semantic + keyword)
- [x] Result reranking
- [x] Diversity filtering
- [x] Storage optimization (quantization)
- [x] Evaluation metrics
- [x] CLI interface
- [x] Error handling
- [x] Logging
- [ ] API rate limiting
- [ ] Caching layer
- [ ] Monitoring/observability
