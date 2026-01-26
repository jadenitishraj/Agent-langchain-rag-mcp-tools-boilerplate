# System Architecture 🏗️

This document describes the high-level architecture of the AgentForge boilerplate.

## 🧩 Core Components

### 1. Agent Logic (`langraph/`)
The core of the system is built on **LangGraph**, a stateful orchestration library for LLM agents.
- **State Graph**: Defines the flow (nodes and edges) of the agent.
- **Nodes**:
  - `validate_input`: Checks user input for safety.
  - `retrieve`: Fetches relevant context from RAG.
  - `call_model`: Invokes the LLM (GPT-4o) with tools.
  - `execute_tools`: Runs requested tools.
  - `validate_output`: Sanitizes the final response.
- **Memory**: Persistent conversation history and user memories.

### 2. RAG Pipeline (`rag/` & `rag_v2/`)
Retrieval Augmented Generation allows the agent to access custom knowledge (like this codebase).
- **Ingestion**: `loader.py` reads files (PDFs, code, markdown).
- **Chunking**: `processor.py` splits text into semantic chunks.
- **Embedding**: `embedder.py` converts text to vectors (OpenAI Ada-002/3).
- **Storage**: Local JSON/NumPy vector store (ChromaDB compatible).
- **Retrieval**: Hybrid search (Vector + Keyword) with re-ranking.

### 3. Model Context Protocol (`mcp_servers/`)
Implements Anthropic's [MCP](https://modelcontextprotocol.io/) to standardize tool connections.
- **MCP Client**: Connects the LangGraph agent to MCP servers.
- **MCP Servers**:
  - `duckduckgo_server`: Web search.
  - `sqlite_server`: Database operations.
  - `brave_server`: Alternative search.

### 4. Guardrails (`guardrails/`)
ensure safety and quality at both input and output stages.
- **Input Guards**:
  - Prompt Injection Detection
  - Toxicity/Hate Speech Check
  - Topic Relevance Filter
- **Output Guards**:
  - PII Redaction (Email, Phone, Credit Card)
  - Hallucination Detection
  - Response Quality Check

### 5. Frontend (`frontend/`)
A modern, responsive UI built with:
- **React 19**: Component-based UI.
- **Vite**: Fast build tool.
- **TailwindCSS**: Utility-first styling.
- **Server-Sent Events (SSE)**: For real-time streaming responses.

## 🔄 Data Flow

1. **User Request**: User sends a message via Frontend.
2. **API Layer**: FastAPI receives request at `/agent/ask/stream`.
3. **Agent Start**: LangGraph initializes state with history.
4. **Input Guard**: Validates request. If safe, proceeds.
5. **Retrieval**: RAG system searches for relevant context.
6. **Thinking**: LLM processes context + query + tools.
7. **Tool Execution** (Optional): If tools are called (e.g., search, memory), they execute and return results.
8. **Generation**: LLM generates final response.
9. **Output Guard**: Response is checked for PII/safety.
10. **Streaming**: Response is streamed token-by-token to Frontend.

## 🗄️ Database Schema

We use **SQLite** (via SQLAlchemy) for persistent data:

**Users Table**
- `id`: Integer PK
- `email`: String (Unique)
- `hashed_password`: String
- `is_active`: Boolean

*Note: Agent memories are stored via the SQLite MCP server in a separate structure.*
