"""
Streamlit frontend for the AWS Customer Agreement RAG system.

Runs as a SEPARATE process from the FastAPI backend and talks to it only over
HTTP (via `requests`). Two views (selected in the sidebar):
  - Chat:      ask a question, see the grounded answer + its source sections.
  - Analytics: usage stats pulled from the backend's /analytics endpoint.

We use a sidebar page selector (not st.tabs) so `st.chat_input` stays at the
top level and pins to the bottom of the page as intended.

Start the backend first (uvicorn app.main:app), then:
    streamlit run frontend/streamlit_app.py
"""

import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")
ASK_TIMEOUT = 60  # seconds; LLM calls can take a moment

st.set_page_config(page_title="AWS Agreement RAG", page_icon="📄", layout="wide")


# --- Backend helpers ---------------------------------------------------------

def backend_health() -> dict | None:
    """Return health JSON, or None if the backend is unreachable."""
    try:
        return requests.get(f"{API_URL}/health", timeout=5).json()
    except requests.RequestException:
        return None


def ask_question(question: str) -> dict:
    """POST /ask and return the parsed response (raises on HTTP error)."""
    resp = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=ASK_TIMEOUT)
    if resp.status_code != 200:
        detail = resp.json().get("detail", resp.text)
        raise RuntimeError(detail)
    return resp.json()


def render_sources(sources: list[dict]) -> None:
    """Render the source sections under an expander."""
    if not sources:
        return
    with st.expander("📎 Sources"):
        for s in sources:
            st.markdown(f"**Section {s['section']} — {s['title']}**")
            st.caption(s["snippet"] + "…")


# --- Sidebar: status + navigation -------------------------------------------

with st.sidebar:
    st.header("📄 AWS Agreement RAG")
    page = st.radio("View", ["💬 Chat", "📊 Analytics"])

    st.divider()
    st.caption(f"Backend: `{API_URL}`")
    health = backend_health()
    if health is None:
        st.error("Backend not reachable.\nStart it with:\n\n`uvicorn app.main:app`")
    elif not health.get("ingested"):
        st.warning("No document ingested yet.")
        if st.button("Ingest document now"):
            with st.spinner("Embedding the PDF..."):
                try:
                    r = requests.post(f"{API_URL}/ingest", timeout=300).json()
                    st.success(f"Ingested {r['num_chunks']} chunks.")
                except requests.RequestException as exc:
                    st.error(f"Ingestion failed: {exc}")
    else:
        st.success("Document ingested ✓")


# --- Chat page ---------------------------------------------------------------

if page == "💬 Chat":
    st.title("💬 Ask the AWS Customer Agreement")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Replay the conversation so far.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("caption"):
                st.caption(msg["caption"])
            render_sources(msg.get("sources", []))

    # st.chat_input at the top level -> pinned to the bottom of the page.
    if prompt := st.chat_input("Ask about the AWS Customer Agreement…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching the agreement…"):
                try:
                    data = ask_question(prompt)
                except Exception as exc:  # noqa: BLE001
                    err = f"⚠️ {exc}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
                else:
                    st.markdown(data["answer"])
                    caption = (f"`answer_found={data['answer_found']}` · "
                               f"`score={data['top_score']:.2f}` · `{data['latency_ms']:.0f} ms`")
                    st.caption(caption)
                    render_sources(data.get("sources", []))
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "caption": caption,
                        "sources": data.get("sources", []),
                    })


# --- Analytics page ----------------------------------------------------------

else:
    st.title("📊 Usage analytics")
    if st.button("🔄 Refresh"):
        st.rerun()

    try:
        analytics = requests.get(f"{API_URL}/analytics", timeout=10).json()
    except requests.RequestException as exc:
        st.error(f"Could not load analytics: {exc}")
        analytics = None

    if analytics:
        ov = analytics["overview"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total queries", ov.get("total_queries", 0))
        c2.metric("Answered", ov.get("answered") or 0)
        c3.metric("Not found", ov.get("not_found") or 0)
        c4.metric("Answered rate", f"{(ov.get('answered_rate') or 0) * 100:.0f}%")

        lat = analytics["latency"]
        st.markdown("**Response latency (ms)**")
        l1, l2, l3 = st.columns(3)
        l1.metric("Average", lat.get("avg_latency_ms") or 0)
        l2.metric("Min", lat.get("min_latency_ms") or 0)
        l3.metric("Max", lat.get("max_latency_ms") or 0)

        st.markdown("**Most frequently asked questions**")
        freq = analytics["most_frequent_questions"]
        if freq:
            df = pd.DataFrame(freq).set_index("question")
            st.bar_chart(df["count"])
            st.dataframe(df, use_container_width=True)

        st.markdown("**Queries with no answer found**")
        na = analytics["no_answer_queries"]
        if na:
            st.dataframe(pd.DataFrame(na), use_container_width=True)
        else:
            st.caption("None yet.")

        st.divider()
        with st.expander("🗄️ SQL queries powering this dashboard"):
            st.markdown("All analytics are computed live from the `query_logs` SQLite table.")
            st.code("""
-- 1. Most frequently asked questions
SELECT question, COUNT(*) AS count
FROM query_logs
GROUP BY LOWER(TRIM(question))
ORDER BY count DESC, question ASC
LIMIT 10;

-- 2. Queries where no answer was found in context
SELECT question, top_score, created_at
FROM query_logs
WHERE answer_found = 0
ORDER BY created_at DESC;

-- 3. Average response latency
SELECT
    ROUND(AVG(latency_ms), 1) AS avg_latency_ms,
    ROUND(MIN(latency_ms), 1) AS min_latency_ms,
    ROUND(MAX(latency_ms), 1) AS max_latency_ms
FROM query_logs;

-- 4. Overview (total, answered, not-found, rate)
SELECT
    COUNT(*)                                          AS total_queries,
    SUM(answer_found)                                 AS answered,
    SUM(CASE WHEN answer_found = 0 THEN 1 ELSE 0 END) AS not_found
FROM query_logs;
""", language="sql")
