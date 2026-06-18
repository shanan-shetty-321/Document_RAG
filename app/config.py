"""
Central configuration for the RAG system.

Every "magic number" and design decision lives here so the project is easy to
read and the choices are easy to defend in review. Each value maps to a
decision explained in the README / technical report.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load secrets/config from a local .env file (never committed). See .env.example.
load_dotenv()

# --- Paths -------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PDF_PATH = DATA_DIR / "aws_customer_agreement.pdf"

# Persisted artifacts (gitignored): the Chroma vector store and the SQLite log.
STORAGE_DIR = ROOT_DIR / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma_db"
DB_PATH = STORAGE_DIR / "usage.db"
CHROMA_COLLECTION = "aws_agreement"

# --- Chunking ----------------------------------------------------------------
# We chunk along the document's own section numbering (1, 1.1, 2.3 ...). Most
# legal clauses are short, but a few sections are long, so we cap chunk size and
# split oversized sections, with a small overlap so a clause that spans a
# boundary keeps context on both sides.
MAX_CHUNK_CHARS = 1000      # ~250 tokens -> well under bge-small's 512-token window
CHUNK_OVERLAP_CHARS = 150   # ~15% overlap; preserves context across splits

# --- Embeddings --------------------------------------------------------------
# Local, free, runs on CPU. bge-small-en-v1.5 beats all-MiniLM-L6-v2 on
# retrieval at the same 384-dim footprint and 512-token window.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# --- Retrieval ---------------------------------------------------------------
# Chunks fed to the LLM as context. For this small (~106-chunk) corpus, k=8
# maximizes retrieval recall at negligible cost (the 70b model handles the extra
# context easily); empirically it recovers answers that k=5 missed, e.g. Force
# Majeure (11.3) and End-User support (2.5) where query wording differs from the
# contract's wording.
TOP_K = 8

# Hybrid retrieval: fuse keyword (BM25) + dense (vector) search. Dense is the
# primary signal; BM25 catches exact legal terms / section numbers the vectors
# might drift past. Weights are used by LangChain's EnsembleRetriever (RRF).
DENSE_WEIGHT = 0.7
BM25_WEIGHT = 0.3

# "Answer not in document" gate. Chroma's relevance score is in [0, 1] (it
# converts cosine distance to similarity). If the best chunk scores below this,
# we treat the question as out-of-scope and skip the LLM.
# Calibrated empirically: answerable questions score ~0.78-0.85, clearly
# out-of-scope ones ~0.45. 0.45 is a conservative gate; the LLM prompt (which
# can reply NOT_FOUND) is the second safety net. Re-checked after the 30-query run.
MIN_RELEVANCE = 0.45

# --- LLM (Groq, free hosted API) ---------------------------------------------
# Embeddings run locally; only answer-generation uses a hosted API. Groq is
# free, fast, and needs no credit card. Get a key at https://console.groq.com.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = 0.0  # deterministic, fact-grounded answers (no creativity)

# --- API ---------------------------------------------------------------------
# Base URL the Streamlit frontend uses to reach the FastAPI backend.
API_URL = os.getenv("API_URL", "http://localhost:8000")
