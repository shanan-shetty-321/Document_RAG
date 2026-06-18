"""
RAG pipeline: retrieve relevant chunks -> build a grounded prompt -> generate.

Retrieval is HYBRID: a keyword retriever (BM25) and a dense vector retriever
(Chroma) are fused with LangChain's EnsembleRetriever (reciprocal-rank fusion).
Dense is the primary signal; BM25 catches exact legal terms / section numbers
that embeddings can drift past.

"Answer not in document" is handled in two layers:
  1. Relevance gate: if the best dense relevance score is below MIN_RELEVANCE,
     the question is out-of-scope -> return not-found WITHOUT calling the LLM
     (cheap, and a clean signal for analytics).
  2. Prompt instruction: the LLM is told to reply exactly "NOT_FOUND" if the
     context doesn't contain the answer -> catches the rest, prevents
     hallucination.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# LangChain 1.x moved the legacy retrievers into the `langchain_classic` package.
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app import config, ingest, vectorstore

# Sentinel the LLM must return when the context doesn't answer the question.
NOT_FOUND = "NOT_FOUND"

SYSTEM_PROMPT = (
    "You are a precise assistant answering questions about the AWS Customer "
    "Agreement. Answer ONLY using the provided context excerpts - do not use "
    "any outside knowledge. Be concise and cite the relevant section "
    "number(s). If the answer is not contained in the context, reply with "
    f"exactly: {NOT_FOUND}"
)


@dataclass
class RagResult:
    """Everything the API needs to return an answer and log the interaction."""
    answer: str
    answer_found: bool
    sources: list[dict] = field(default_factory=list)  # [{section, title, snippet}]
    top_score: float = 0.0
    retrieved_sections: list[str] = field(default_factory=list)
    num_chunks: int = 0


def _format_context(docs) -> str:
    """Render retrieved chunks as labelled context blocks for the prompt."""
    blocks = []
    for doc in docs:
        section = doc.metadata.get("section", "?")
        title = doc.metadata.get("title", "")
        blocks.append(f"[Section {section} - {title}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def _to_sources(docs) -> list[dict]:
    """Compact, de-duplicated source list for the API response / UI."""
    seen, sources = set(), []
    for doc in docs:
        key = doc.metadata.get("chunk_id")
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "section": doc.metadata.get("section", "?"),
            "title": doc.metadata.get("title", ""),
            "snippet": doc.page_content[:240].strip(),
        })
    return sources


class RagPipeline:
    """Holds the loaded models/retrievers so the API builds them once on startup."""

    def __init__(self) -> None:
        # Dense store (Chroma) — also used directly for relevance scoring.
        self.store = vectorstore.load_vectorstore()
        # BM25 needs the raw documents; rebuilding chunks is deterministic & fast.
        documents = ingest.to_documents(ingest.build_chunks())
        bm25 = BM25Retriever.from_documents(documents)
        bm25.k = config.TOP_K
        dense = self.store.as_retriever(search_kwargs={"k": config.TOP_K})
        # Reciprocal-rank fusion of keyword + dense results.
        self.retriever = EnsembleRetriever(
            retrievers=[bm25, dense],
            weights=[config.BM25_WEIGHT, config.DENSE_WEIGHT],
        )
        # The LLM is built lazily so retrieval works without a key (and a clear
        # error is raised only when generation is actually attempted).
        self._llm: ChatGroq | None = None

    def _get_llm(self) -> ChatGroq:
        if self._llm is None:
            if not config.GROQ_API_KEY:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
                    "free Groq key from https://console.groq.com/keys"
                )
            self._llm = ChatGroq(
                model=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                api_key=config.GROQ_API_KEY,
            )
        return self._llm

    def _top_relevance(self, question: str) -> float:
        """Best dense relevance score in [0, 1] for the not-found gate."""
        scored = self.store.similarity_search_with_relevance_scores(question, k=1)
        return scored[0][1] if scored else 0.0

    def answer(self, question: str) -> RagResult:
        """Run the full RAG pipeline for one question."""
        top_score = self._top_relevance(question)

        # Layer 1: relevance gate — skip the LLM for clearly out-of-scope queries.
        if top_score < config.MIN_RELEVANCE:
            return RagResult(
                answer="I could not find an answer to that in the AWS Customer Agreement.",
                answer_found=False,
                top_score=top_score,
            )

        # EnsembleRetriever fuses BM25 + dense (RRF); keep the top-K fused chunks
        # so the LLM context stays tight and consistent with our stated top-k.
        docs = self.retriever.invoke(question)[: config.TOP_K]
        context = _format_context(docs)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
        ]
        raw = self._get_llm().invoke(messages).content.strip()

        # Layer 2: the LLM judged the context insufficient.
        if NOT_FOUND in raw.upper():
            return RagResult(
                answer="I could not find an answer to that in the AWS Customer Agreement.",
                answer_found=False,
                top_score=top_score,
                retrieved_sections=sorted({d.metadata.get("section", "?") for d in docs}),
                num_chunks=len(docs),
            )

        return RagResult(
            answer=raw,
            answer_found=True,
            sources=_to_sources(docs),
            top_score=top_score,
            retrieved_sections=sorted({d.metadata.get("section", "?") for d in docs}),
            num_chunks=len(docs),
        )


# --- Inspection entry point --------------------------------------------------
# `python -m app.rag` runs a couple of questions. Retrieval works without a key;
# generation needs GROQ_API_KEY in .env.

if __name__ == "__main__":
    pipe = RagPipeline()
    questions = [
        "What is the interest rate charged on late payments?",
        "How can I terminate the agreement for convenience?",
        "What is the price of an EC2 instance?",  # out-of-scope
    ]
    has_key = bool(config.GROQ_API_KEY)
    if not has_key:
        print("(No GROQ_API_KEY set — showing retrieval only, skipping generation.)\n")
    for q in questions:
        print(f"Q: {q}")
        if has_key:
            res = pipe.answer(q)
            print(f"  answer_found={res.answer_found}  top_score={res.top_score:.3f}")
            print(f"  A: {res.answer}")
            print(f"  sources: {[s['section'] for s in res.sources]}")
        else:
            score = pipe._top_relevance(q)
            docs = pipe.retriever.invoke(q)
            print(f"  top_score={score:.3f}  sections={[d.metadata.get('section') for d in docs]}")
        print()
