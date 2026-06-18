"""
FastAPI backend exposing the RAG system.

Endpoints:
  POST /ingest    - parse, chunk and embed the PDF into the vector store
  POST /ask       - run the RAG pipeline, return answer + sources, log the call
  GET  /analytics - usage analytics from the SQL log

The RAG pipeline (models + retrievers) is built once and kept in app state so
requests are fast. Errors return meaningful HTTP codes, not raw stack traces.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app import config, db, ingest, rag, vectorstore
from app.schemas import AnalyticsResponse, AskRequest, AskResponse, IngestResponse

# Holds the single shared RagPipeline (None until a document is ingested).
state: dict = {"pipeline": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """On startup: ensure the DB exists and load the pipeline if already built."""
    db.init_db()
    if vectorstore.vectorstore_exists():
        state["pipeline"] = rag.RagPipeline()
    yield


app = FastAPI(
    title="AWS Customer Agreement RAG API",
    description="Ask questions about the AWS Customer Agreement (RAG over a single PDF).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Liveness probe + whether a document has been ingested."""
    return {"status": "ok", "ingested": state["pipeline"] is not None}


@app.post("/ingest", response_model=IngestResponse)
def ingest_document() -> IngestResponse:
    """(Re)build the vector store from the PDF and load the pipeline."""
    try:
        store = vectorstore.build_vectorstore()
        state["pipeline"] = rag.RagPipeline()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Source PDF not found in data/.")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    sections = sorted({c.section for c in ingest.build_chunks()},
                      key=lambda s: (ingest._major(s), s))
    return IngestResponse(
        status="ingested",
        num_chunks=store._collection.count(),
        sections=sections,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Answer a question via RAG, log the interaction, return answer + sources."""
    pipeline: rag.RagPipeline | None = state["pipeline"]
    if pipeline is None:
        raise HTTPException(
            status_code=409,
            detail="No document has been ingested yet. Call POST /ingest first.",
        )

    start = time.perf_counter()
    try:
        result = pipeline.answer(req.question)
    except RuntimeError as exc:  # e.g. missing GROQ_API_KEY
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - LLM/network failure
        raise HTTPException(status_code=503, detail=f"LLM generation failed: {exc}")
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    db.log_query(
        question=req.question,
        answer=result.answer,
        answer_found=result.answer_found,
        top_score=result.top_score,
        retrieved_sections=result.retrieved_sections,
        num_chunks=result.num_chunks,
        latency_ms=latency_ms,
        model=config.LLM_MODEL,
    )

    return AskResponse(
        answer=result.answer,
        answer_found=result.answer_found,
        sources=result.sources,
        top_score=round(result.top_score, 3),
        retrieved_sections=result.retrieved_sections,
        num_chunks=result.num_chunks,
        latency_ms=latency_ms,
    )


@app.get("/analytics", response_model=AnalyticsResponse)
def analytics() -> AnalyticsResponse:
    """Usage analytics: most frequent questions, no-answer queries, avg latency."""
    return AnalyticsResponse(**db.get_analytics())
