# AgentForge Boilerplate 🚀

**The Ultimate Full-Stack AI Agent Starter Kit**

<img width="901" height="866" alt="Screenshot 2026-01-26 at 9 38 18 AM" src="https://github.com/user-attachments/assets/d1191728-6d0a-4841-94a1-31181420d060" />


AgentForge is a production-ready boilerplate for building advanced AI agents. It combines the power of **LangGraph** for orchestration, **RAG** for knowledge retrieval, and **MCP (Model Context Protocol)** for standardized tool integration—all wrapped in a modern **FastAPI** backend and **React** frontend..

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![React](https://img.shields.io/badge/react-18%2B-61DAFB.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688.svg)

## ✨ Key Features

- **🤖 LangGraph Agent**: State-of-the-art agent orchestration with state management.
- **📚 RAG Pipeline v2**: Advanced retrieval with semantic chunking, re-ranking, and hybrid search.
- **🔌 MCP Integration**: Full support for Anthropic's Model Context Protocol (Client & Server).
- **🛡️ Guardrails**: Input/Output validation for safety, privacy (PII redaction), and quality.
- **⚡ Full-Stack**: 
  - **Backend**: FastAPI with async support and streaming responses.
  - **Frontend**: Modern React (Vite) with TailwindCSS and markdown rendering.
- **🧠 Memory**: Persistent user memories using SQLite.
- **🔍 Web Search**: Integrated free web search via DuckDuckGo and Brave.

## 🏗️ Architecture Overview

```mermaid
graph TD
    User[User / Frontend] <-->|Rest API / SSE| API[FastAPI Backend]
    API <-->|Orchestration| Agent[LangGraph Agent]
    
    subgraph "Agent Brain"
        Agent <-->|Safety| Guard[Guardrails]
        Agent <-->|Context| RAG[RAG Pipeline]
        Agent <-->|Tools| MCP[MCP Client]
        Agent <-->|State| Memory[SQLite Memory]
    end
    
    subgraph "External"
        MCP <-->|Protocol| Tools[External Tools]
        RAG <-->|Embeddings| VectorDB[Vector Store]
        Agent <-->|Inference| LLM[OpenAI GPT-4o]
    end
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- OpenAI API Key

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/agentforge.git](https://github.com/jadenitishraj/Agent-langchain-rag-mcp-tools-boilerplate.git
cd agentforge
```

### 2. Backend Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. Index the Codebase (RAG)

Make the agent self-aware by indexing the codebase:

```bash
# From root directory
source venv/bin/activate
python scripts/index_codebase.py
```

### 5. Run Everything

You can run the components separately:

**Backend:**
```bash
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm run dev
```

Visit **http://localhost:5173** to chat with your agent!

## 📂 Project Structure

- **`langraph/`**: Core agent logic, state graph, and router.
- **`rag/`**: Retrieval Augmented Generation pipeline (Loader -> Embedder -> Store).
- **`mcp_servers/`**: Model Context Protocol servers (Search, SQLite).
- **`guardrails/`**: Input/Output safety checks (PII, Toxicity, Hallucination).
- **`frontend/`**: React application with TailwindCSS.
- **`tools/`**: Custom tools (Memory, Contact).

## 🛠️ Customization

### Adding a New Tool

1. Define your tool in `tools/my_tool.py` using `@tool` decorator.
2. Add it to `ALL_TOOLS` in `tools/__init__.py`.
3. The agent will automatically detect and use it!

### Modifying the System Prompt

Edit `langraph/agent.py` and update the `SYSTEM_PROMPT` variable to change the agent's personality and instructions.

## 🤝 Contributing

Contributions are welcome! Please read `CONTRIBUTING.md` for details.

## 📄 License

MIT License - feel free to use this boilerplate for your own projects!
