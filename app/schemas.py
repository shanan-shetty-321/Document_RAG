"""
Pydantic models for request/response validation.

Using explicit models gives us automatic validation, clear error messages, and
self-documenting OpenAPI docs at /docs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


# --- /ask --------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="A question about the AWS Customer Agreement.")

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty or whitespace")
        return v.strip()


class Source(BaseModel):
    section: str
    title: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    answer_found: bool
    sources: list[Source]
    top_score: float
    retrieved_sections: list[str]
    num_chunks: int
    latency_ms: float


# --- /ingest -----------------------------------------------------------------

class IngestResponse(BaseModel):
    status: str
    num_chunks: int
    sections: list[str]


# --- /analytics --------------------------------------------------------------

class Overview(BaseModel):
    total_queries: int
    answered: int | None = None
    not_found: int | None = None
    answered_rate: float


class FrequentQuestion(BaseModel):
    question: str
    count: int


class NoAnswerQuery(BaseModel):
    question: str
    top_score: float | None = None
    created_at: str


class LatencyStats(BaseModel):
    avg_latency_ms: float | None = None
    min_latency_ms: float | None = None
    max_latency_ms: float | None = None


class AnalyticsResponse(BaseModel):
    overview: Overview
    most_frequent_questions: list[FrequentQuestion]
    no_answer_queries: list[NoAnswerQuery]
    latency: LatencyStats
