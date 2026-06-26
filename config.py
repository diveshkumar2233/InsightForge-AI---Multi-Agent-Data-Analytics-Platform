"""
config.py
─────────
Central configuration for InsightForge AI.
All settings are loaded from environment variables (via .env).
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# ── Load .env ────────────────────────────────────────────────────────────────
load_dotenv()

# ── Base Paths ───────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
UPLOADS_DIR     = Path(os.getenv("UPLOADS_DIR",   BASE_DIR / "data" / "uploads"))
PROCESSED_DIR   = Path(os.getenv("PROCESSED_DIR", BASE_DIR / "data" / "processed"))
REPORTS_DIR     = Path(os.getenv("REPORTS_DIR",   BASE_DIR / "reports"))
CHROMA_DIR      = Path(os.getenv("CHROMA_PERSIST_DIR", BASE_DIR / "vector_store" / "chroma_db"))
SQLITE_DB_PATH  = Path(os.getenv("SQLITE_DB_PATH", BASE_DIR / "data" / "app.db"))

# ── Create dirs eagerly ──────────────────────────────────────────────────────
for _d in [UPLOADS_DIR, PROCESSED_DIR, REPORTS_DIR, CHROMA_DIR, SQLITE_DB_PATH.parent]:
    _d.mkdir(parents=True, exist_ok=True)

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY   : str = os.getenv("GROQ_API_KEY",   "")
GOOGLE_API_KEY : str = os.getenv("GOOGLE_API_KEY", "")

# ── LLM Settings ─────────────────────────────────────────────────────────────
GROQ_MODEL          : str   = "llama3-70b-8192"   # fast + capable
GROQ_TEMPERATURE    : float = 0.1
GROQ_MAX_TOKENS     : int   = 4096
EMBEDDING_MODEL     : str   = "models/embedding-001"  # Google

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_COLLECTION   : str = os.getenv("CHROMA_COLLECTION", "insightforge_docs")

# ── Upload limits ─────────────────────────────────────────────────────────────
MAX_UPLOAD_MB       : int = int(os.getenv("MAX_UPLOAD_MB", 200))
MAX_UPLOAD_BYTES    : int = MAX_UPLOAD_MB * 1024 * 1024

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    level=LOG_LEVEL,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}",
    colorize=True,
)
logger.add(BASE_DIR / "insightforge.log", rotation="10 MB", retention="7 days", level="DEBUG")

# ── Guard ─────────────────────────────────────────────────────────────────────
def validate_env() -> list[str]:
    """Return list of missing critical env vars."""
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")
    return missing
