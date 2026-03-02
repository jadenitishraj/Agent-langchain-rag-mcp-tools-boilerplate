import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# =============================================================================
# BLOCKER FIX #2: Database URL from environment variable.
# In production, set DATABASE_URL to a PostgreSQL connection string.
# SQLite is kept as a fallback for local development only — it cannot handle
# concurrent writes which will cause lock errors under any real load.
#
# Production example:
#   DATABASE_URL=postgresql://user:password@localhost:5432/agentforge
#
# The DATABASE_URL in .env.example shows the correct format.
# =============================================================================
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./sql_app.db")

IS_SQLITE = DATABASE_URL.startswith("sqlite")

if IS_SQLITE:
    # SQLite requires check_same_thread=False for FastAPI's threading model
    connect_args = {"check_same_thread": False}
    print("⚠️  WARNING: Using SQLite. Switch to PostgreSQL (DATABASE_URL) for production.")
else:
    connect_args = {}
    # PostgreSQL: use proper connection pooling
    print(f"✅ Using database: {DATABASE_URL.split('@')[-1]}")  # log host only, not credentials

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    # Pool settings: safe defaults that work for both SQLite and Postgres
    pool_pre_ping=True,   # Detect stale connections before use
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
