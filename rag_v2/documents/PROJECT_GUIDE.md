# AgentForge Project Guide & Architecture Explanation 📘

## 🌟 Executive Summary

**AgentForge** is a modern, production-grade boilerplate for building AI Agents. It is not just a chatbot, but a sophisticated **Cognitive Architecture** that combines:

1.  **Reasoning**: Using **LangGraph** to model complex workflows as a graph (nodes & edges).
2.  **Knowledge**: Using **RAG (Retrieval Augmented Generation)** to ground answers in real data.
3.  **Action**: Using **MCP (Model Context Protocol)** to safely interact with the outside world (databases, web, APIs).
4.  **Safety**: Using **Guardrails** to ensure the AI behaves correctly.

---

## 🏗️ System Architecture: The "Brain" Anatomy

Imagine the agent as a digital employee. Here is how its "brain" is structured:

### 1. The Cortex (LangGraph Agent)
*Folder: `langraph/`*

This is the decision-making center. Unlike simple chains (Step A -> Step B), our agent works like a flow chart:
*   **Router (`router.py`)**: The "ears" of the agent. It receives queries from the user via the API.
*   **Graph (`agent.py`)**: The "mind". It decides:
    *   *"Should I search memory?"*
    *   *"Should I look up the code?"*
    *   *"Is this a dangerous question?"*
*   **State**: It remembers the conversation history and what it's currently working on.

### 2. The Library (RAG Pipeline)
*Folder: `rag_v2/`*

The agent doesn't know everything. It uses RAG to "look up" information from your documents/codebase.
*   **Ingestion**: Reading files (PDFs, Code).
*   **Chunking**: Breaking text into small, meaningful pieces.
*   **Embedding**: Converting text into numbers (vectors) so computers can compare concepts.
*   **Vector Database (Qdrant)**: A specialized database that stores these meaning-vectors for fast searching.

**Why v2?** Our v2 pipeline uses "Semantic Chunking" – it keeps related sentences together rather than just cutting text at random spots. This makes retrieval much smarter.

### 3. The Hands (MCP & Tools)
*Folder: `mcp_servers/` and `tools/`*

An AI model is just text-in, text-out. To *do* things, it uses Tools.
*   **MCP (Model Context Protocol)**: A standard way to connect AI to systems. Think of it like "USB for AI".
    *   **DuckDuckGo Server**: Allows the agent to search the web for real-time info.
    *   **SQLite Server**: persistent memory storage (remembering user preferences).
*   **Local Tools**: Simple Python functions (e.g., specific math or logic) the agent can run directly.

### 4. The Filters (Guardrails)
*Folder: `guardrails/`*

These are the safety checks.
*   **Input Guard**: Checks the user's question. *"Is this asking for illegal stuff? Is it prompt injection?"*
*   **Output Guard**: Checks the agent's answer. *"Did I leak a password? Am I hallucinating facts?"*

### 5. The Face (Frontend)
*Folder: `frontend/`*

A modern **React 19** interface.
*   **Streaming**: It shows the answer being typed out in real-time (like ChatGPT).
*   **Markdown Rendering**: It displays code blocks beautifully with syntax highlighting.
*   **TailwindCSS**: Professional styling.

---

## 📂 detailed File Map (English Explanations)

### Root Directory
*   `main.py`: The **Entry Point**. It starts the web server (FastAPI). It connects the Frontend asking questions to the Backend agent.
*   `models.py` & `database.py`: Handles user accounts (login/signup) using a traditional SQL database.

### langraph/
*   `agent.py`: **The Master Controller**. Defines the StateGraph. It says: "First Validate Input -> Then Retrieval -> Then Call LLM -> Then Validate Output".
*   `router.py`: **The API Endpoints**. Defines URLs like `/agent/ask` that the frontend sends data to.

### rag_v2/
*   `pipeline.py`: **The Coordinator**. Runs the full flow: Read File -> Chunk -> Embed -> Store.
*   `chunker.py`: **The Scissors**. Smartly cuts text.
*   `embedder.py`: **The Translator**. Turns text into vector arrays (lists of numbers).
*   `vector_store.py`: **The Librarian**. Manages storing and finding vectors in Qdrant.

### mcp_servers/
*   `mcp_client.py`: The bridge that lets our Python Agent talk to MCP servers.
*   `duckduckgo_server.py`: A mini-server that performs web searches.

### guardrails/
*   `input_guards.py`: Contains logic to detect "Jailbreaks" (users trying to trick the AI).
*   `output_guards.py`: Contains logic to redact emails/phones (PII) from answers.

---

## 🔄 How a Request Flows (The Life of a Message)

1.  **User** types: *"How does RAG work?"* in the Frontend.
2.  **Frontend** sends this text to `main.py` (Backend).
3.  **Backend** passes it to the `Agent` (`langraph/agent.py`).
4.  **Agent Step 1 (Guard)**: Checks if the question is safe.
5.  **Agent Step 2 (Retrieve)**:
    *   Calls `rag_v2` to look up "RAG" in the vector database.
    *   Finds this very file (`PROJECT_GUIDE.md`) and code files.
6.  **Agent Step 3 (Think)**:
    *   Sends the User's question + The Retrieved File Content to GPT-4o.
    *   System Prompt says: *"You are an expert developer. Explain using the context."*
7.  **Agent Step 4 (Generate)**: GPT-4o generates the explanation.
8.  **Agent Step 5 (Guard)**: Checks the answer for secrets/errors.
9.  **Frontend**: Receives the text stream and displays it nicely.

---

## 🔮 Core Concepts Glossary

*   **LangChain**: A library that creates "chains" of logic (A -> B -> C).
*   **LangGraph**: An evolution of LangChain that creates "graphs" (loops, branches, states). Essential for complex agents.
*   **Vector**: A long list of numbers representing meaning. "King" and "Queen" -> vectors are close. "King" and "Banana" -> vectors are far apart.
*   **FastAPI**: A super-fast Python framework for building web APIs (the backend server).
*   **React**: A Javascript library for building user interfaces (the frontend).
