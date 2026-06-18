"""
Vector store: embed the chunks and store them in a persisted Chroma collection.

- Embeddings: BAAI/bge-small-en-v1.5 via HuggingFaceEmbeddings. Local, free,
  CPU. We normalize embeddings (bge's recommendation) so cosine similarity is
  well-behaved.
- Store: Chroma configured with cosine space (hnsw:space=cosine), persisted to
  disk so we only embed once and reload instantly on restart.

Chroma's relevance score (from `similarity_search_with_relevance_scores`) is in
[0, 1] = 1 - cosine_distance, i.e. higher = more similar. We use that score for
the "answer not in document" gate in rag.py.
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app import config, ingest


def get_embeddings() -> HuggingFaceEmbeddings:
    """The local bge-small embedding model (shared by ingest and query time)."""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        # bge models are trained for cosine similarity on L2-normalized vectors.
        encode_kwargs={"normalize_embeddings": True},
    )


def vectorstore_exists() -> bool:
    """True if a persisted Chroma store has already been built."""
    return config.CHROMA_DIR.exists() and any(config.CHROMA_DIR.iterdir())


def build_vectorstore() -> Chroma:
    """Parse + chunk the PDF, embed every chunk, and persist the Chroma store.

    Safe to call repeatedly: it rebuilds the collection from scratch so a
    re-ingest always reflects the current document and chunking logic.
    """
    documents = ingest.to_documents(ingest.build_chunks())
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
        collection_name=config.CHROMA_COLLECTION,
        collection_metadata={"hnsw:space": "cosine"},
    )


def load_vectorstore() -> Chroma:
    """Load the already-built Chroma store from disk."""
    return Chroma(
        persist_directory=str(config.CHROMA_DIR),
        embedding_function=get_embeddings(),
        collection_name=config.CHROMA_COLLECTION,
        collection_metadata={"hnsw:space": "cosine"},
    )


# --- Inspection entry point --------------------------------------------------
# `python -m app.vectorstore` builds the index and prints retrieval scores for a
# couple of sample queries (handy for calibrating MIN_RELEVANCE).

if __name__ == "__main__":
    print("Building vector store (first run downloads the embedding model)...")
    vs = build_vectorstore()
    print(f"Indexed {vs._collection.count()} chunks into Chroma.\n")

    samples = [
        "What is the interest rate on late payments?",   # answerable -> Section 3.1
        "How long must I keep AWS confidential information secret?",  # -> 11.9
        "What is the capital of France?",                # out-of-scope -> low score
    ]
    for q in samples:
        print(f"Q: {q}")
        for doc, score in vs.similarity_search_with_relevance_scores(q, k=3):
            sec = doc.metadata.get("section")
            title = doc.metadata.get("title")
            print(f"   score={score:.3f}  [Section {sec} - {title}]  {doc.page_content[:70]!r}")
        print()
