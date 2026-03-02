from dotenv import load_dotenv
load_dotenv()

import os
import sys

# =============================================================================
# BLOCKER FIX #6: Startup API key validation (fail-fast on boot)
# Crashes immediately if OPENAI_API_KEY is missing instead of failing at
# runtime on the first real request, making misconfiguration obvious.
# =============================================================================
if not os.environ.get("OPENAI_API_KEY"):
    print("❌ FATAL: OPENAI_API_KEY is not set. Please configure your .env file.")
    sys.exit(1)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import models, database
from routers import users
from langraph import router as agent_router

models.Base.metadata.create_all(bind=database.engine)

# =============================================================================
# BLOCKER FIX #3: Rate limiting
# Protects all endpoints from abuse that would drain OpenAI credits.
# Default: 20 req/min globally. Agent endpoints override to 10/min.
# =============================================================================
limiter = Limiter(key_func=get_remote_address, default_limits=["20/minute"])

app = FastAPI(
    title="AgentForge API",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# =============================================================================
# BLOCKER FIX #1: CORS — restrict to allowed origins
# Read from environment so production and local dev can differ.
# Set ALLOWED_ORIGINS="https://yourapp.com" in your prod .env
# Defaults to localhost for local development only.
# =============================================================================
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(users.router)
from routers import feedback

app.include_router(agent_router.router)
app.include_router(feedback.router)

@app.get("/")
def read_root():
    return {"status": "ok", "service": "AgentForge API"}
