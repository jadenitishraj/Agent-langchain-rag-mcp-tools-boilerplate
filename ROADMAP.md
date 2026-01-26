# Future Roadmap 🗺️

We are constantly improving AgentForge. Here are the planned features and improvements.

## 📅 Phase 1: Production Hardening (Q2 2026)
- [ ] **Authentication**: Implement JWT/OAuth2 for API security.
- [ ] **Rate Limiting**: Add Redis-backed throttling to prevent abuse.
- [ ] **Structured Logging**: Replace print statements with ELK-compatible logging.
- [ ] **Error Handling**: comprehensive try/catch blocks with recovery strategies.
- [ ] **Config**: Externalize all settings to `.env` or YAML config.

## 🚀 Phase 2: Scalability (Q3 2026)
- [ ] **Docker Compose**: One-command deployment for the full stack.
- [ ] **Vector DB**: Support for Pinecone/Qdrant/Weaviate (Cloud).
- [ ] **PostgreSQL**: Migrate from SQLite for production persistence.
- [ ] **Async Queues**: Integrate Celery/Redis for long-running tasks.
- [ ] **Caching**: Cache common queries and embeddings.

## 🛠️ Phase 3: Developer Experience (Q4 2026)
- [ ] **CLI Tool**: `npx create-agent-app` style project generator.
- [ ] **Testing**: Add `pytest` suite and frontend unit tests.
- [ ] **CI/CD**: GitHub Actions workflows for linting and testing.
- [ ] **Documentation**: Auto-generate API docs (Swagger/OpenAPI).

## 🧠 Phase 4: Advanced AI Features (2027)
- [ ] **Multi-Agent**: Orchestrate teams of specialized agents.
- [ ] **Multi-Modal**: Support for Image, Audio, and Video inputs.
- [ ] **Fine-tuning**: Pipeline for training custom models on user data.
- [ ] **Eval Framework**: Automated evaluation of RAG quality (RAGAS).
- [ ] **Self-Correction**: Agent that can critique and fix its own code.

---

*Have a feature request? Open an issue on GitHub!*
