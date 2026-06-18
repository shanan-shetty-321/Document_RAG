"""
SQLite usage logging + analytics.

Every call to /ask is logged as one row in `query_logs`. Because the document
is static, the SQL part is about USAGE analytics (not document content): which
questions are asked most, which got no answer, and how fast we respond.

Schema rationale (one row per question):
  - question / answer        : what was asked and returned
  - answer_found (0/1)        : powers the "no answer found" analytic
  - top_score                 : retrieval confidence (best dense relevance)
  - retrieved_sections (JSON) : which contract sections were used as sources
  - num_chunks                : how many chunks were fed to the LLM
  - latency_ms                : powers the "average response latency" analytic
  - model / created_at        : which LLM, and when (for time-based analysis)
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager

from app import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS query_logs (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    question           TEXT    NOT NULL,
    answer             TEXT,
    answer_found       INTEGER NOT NULL,             -- 1 = answered, 0 = not found
    top_score          REAL,                         -- best dense relevance score
    retrieved_sections TEXT,                          -- JSON list of section labels
    num_chunks         INTEGER,                       -- chunks used as context
    latency_ms         REAL,                          -- end-to-end response time
    model              TEXT,
    created_at         TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""


@contextmanager
def _connect():
    """Connection context manager: commits on success, always closes."""
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create the storage dir and the query_logs table if they don't exist."""
    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript(SCHEMA)


def log_query(
    *,
    question: str,
    answer: str,
    answer_found: bool,
    top_score: float,
    retrieved_sections: list[str],
    num_chunks: int,
    latency_ms: float,
    model: str,
) -> None:
    """Insert one interaction into the log."""
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO query_logs
                (question, answer, answer_found, top_score,
                 retrieved_sections, num_chunks, latency_ms, model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                question,
                answer,
                int(answer_found),
                top_score,
                json.dumps(retrieved_sections),
                num_chunks,
                latency_ms,
                model,
            ),
        )


# --- Analytics (the three required, plus a small overview) -------------------

def most_frequent_questions(limit: int = 10) -> list[dict]:
    """Most frequently asked questions (case/whitespace-insensitive grouping)."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT question, COUNT(*) AS count
            FROM query_logs
            GROUP BY LOWER(TRIM(question))
            ORDER BY count DESC, question ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def no_answer_queries(limit: int = 50) -> list[dict]:
    """Queries where no answer was found in the document context."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT question, top_score, created_at
            FROM query_logs
            WHERE answer_found = 0
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def latency_stats() -> dict:
    """Average (and min/max) response latency in milliseconds."""
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT
                ROUND(AVG(latency_ms), 1) AS avg_latency_ms,
                ROUND(MIN(latency_ms), 1) AS min_latency_ms,
                ROUND(MAX(latency_ms), 1) AS max_latency_ms
            FROM query_logs
            """
        ).fetchone()
    return dict(row) if row else {}


def overview() -> dict:
    """Totals: how many queries, answered vs not-found, and the answered rate."""
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*)                              AS total_queries,
                SUM(answer_found)                     AS answered,
                SUM(CASE WHEN answer_found = 0 THEN 1 ELSE 0 END) AS not_found
            FROM query_logs
            """
        ).fetchone()
    data = dict(row) if row else {}
    total = data.get("total_queries") or 0
    answered = data.get("answered") or 0
    data["answered_rate"] = round(answered / total, 3) if total else 0.0
    return data


def get_analytics() -> dict:
    """Bundle all analytics for the /analytics endpoint."""
    return {
        "overview": overview(),
        "most_frequent_questions": most_frequent_questions(),
        "no_answer_queries": no_answer_queries(),
        "latency": latency_stats(),
    }


# --- Self-test ---------------------------------------------------------------
# `python -m app.db` inserts a few dummy rows into a temp table-less check and
# prints analytics, to confirm the SQL works before wiring it into the API.

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    # Use a throwaway DB so we don't pollute the real usage.db.
    config.DB_PATH = Path(tempfile.gettempdir()) / "rag_db_selftest.db"
    config.DB_PATH.unlink(missing_ok=True)
    init_db()

    samples = [
        ("What is the late payment interest rate?", True, 0.81, ["3.1"], 5, 540.2),
        ("What is the late payment interest rate?", True, 0.81, ["3.1"], 5, 502.7),
        ("How do I terminate the agreement?", True, 0.77, ["5.2"], 5, 610.5),
        ("What is the capital of France?", False, 0.41, [], 0, 12.3),
        ("What is the price of EC2?", False, 0.64, ["11.2"], 5, 480.9),
    ]
    for q, found, score, secs, n, lat in samples:
        log_query(question=q, answer="..." if found else "Not found",
                  answer_found=found, top_score=score, retrieved_sections=secs,
                  num_chunks=n, latency_ms=lat, model=config.LLM_MODEL)

    import pprint
    pprint.pp(get_analytics())
    config.DB_PATH.unlink(missing_ok=True)
