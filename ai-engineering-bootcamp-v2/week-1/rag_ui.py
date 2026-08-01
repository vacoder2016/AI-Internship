"""Minimal Streamlit UI for week-1 RAG — calls POST /ingest and POST /ask only.

Run:
  streamlit run rag_ui.py

Optional env (e.g. in .env):
  RAG_API_BASE_URL=https://your-service.onrender.com
"""

import json
import os

import httpx
import streamlit as st
from load_env import load_course_env

load_course_env()

DEFAULT_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000")
REFUSAL_PHRASE = "I don't have enough information to answer that."


def make_client(base_url: str) -> httpx.Client:
    kwargs: dict = {"timeout": 120.0}
    if base_url.strip().lower().startswith("https://"):
        import ssl

        import truststore

        kwargs["verify"] = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return httpx.Client(**kwargs)


def api_post(base_url: str, path: str, payload: dict) -> tuple[int, dict | str]:
    try:
        with make_client(base_url) as client:
            response = client.post(f"{base_url.rstrip('/')}{path}", json=payload)
        try:
            return response.status_code, response.json()
        except json.JSONDecodeError:
            return response.status_code, response.text
    except httpx.ConnectError:
        return 0, {"error": f"Cannot reach {base_url}. Check the URL and that the service is running."}
    except httpx.HTTPError as exc:
        return 0, {"error": str(exc)}


def chunk_document_ids(chunk_ids: list[str]) -> list[str]:
    seen: list[str] = []
    for chunk_id in chunk_ids:
        doc_id = chunk_id.split("::", 1)[0]
        if doc_id not in seen:
            seen.append(doc_id)
    return seen


def render_ask_result(status: int, data: dict | str) -> None:
    st.markdown(f"**HTTP {status}**")

    if not isinstance(data, dict):
        st.error(str(data))
        return

    if "detail" in data:
        st.error(data["detail"])
        st.json(data)
        return

    answer_obj = data.get("answer")
    if not isinstance(answer_obj, dict):
        st.json(data)
        return

    answer_text = answer_obj.get("answer", "")
    chunk_ids = data.get("retrieved_chunk_ids") or []
    doc_ids = chunk_document_ids(chunk_ids)

    if REFUSAL_PHRASE in answer_text:
        st.warning("Refusal — not enough context in ingested docs")
    else:
        st.success("Answer grounded in retrieved context")

    st.markdown("**Answer**")
    st.write(answer_text)

    st.markdown("**Citations**")
    if chunk_ids:
        st.write("Chunk IDs:", ", ".join(f"`{cid}`" for cid in chunk_ids))
        st.write("Document IDs:", ", ".join(f"`{did}`" for did in doc_ids))
    else:
        st.caption("No chunks retrieved.")

    cols = st.columns(4)
    cols[0].metric("confidence", answer_obj.get("confidence", "—"))
    cols[1].metric("tokens_used", data.get("tokens_used", "—"))
    cols[2].metric("cost_usd", data.get("cost_usd", "—"))
    cols[3].metric("latency_ms", data.get("latency_ms", "—"))

    with st.expander("Full JSON response"):
        st.json(data)


st.set_page_config(page_title="Week 1 RAG UI", layout="wide")
st.title("Week 1 — RAG UI")
st.caption("Streamlit calls your FastAPI service. All RAG logic stays in the API.")

env_default = os.getenv("RAG_API_BASE_URL", DEFAULT_BASE_URL)
base_url = st.sidebar.text_input(
    "API base URL",
    value=env_default,
    help="Set RAG_API_BASE_URL in .env or edit here. No secrets needed — public API URL only.",
)
st.sidebar.code(f"RAG_API_BASE_URL={base_url}", language="bash")
st.sidebar.markdown("**Run this page**")
st.sidebar.code(
    "cd ai-engineering-bootcamp-v2/week-1\n"
    "source .venv/bin/activate   # Windows: .\\.venv\\Scripts\\Activate.ps1\n"
    "streamlit run rag_ui.py",
    language="bash",
)

ingest_tab, ask_tab = st.tabs(["Ingest", "Ask"])

with ingest_tab:
    st.subheader("Ingest document")
    st.markdown("Paste text and send it to `POST /ingest`.")

    document_id = st.text_input("document_id", value="newtons-laws", key="ingest_doc_id")
    source = st.text_input("source (optional filename label)", value="notes.txt", key="ingest_source")
    text = st.text_area(
        "text",
        height=220,
        placeholder="Paste document text here…",
        key="ingest_text",
    )

    if st.button("Ingest", type="primary", key="ingest_btn"):
        payload = {
            "document_id": document_id.strip(),
            "source": source.strip() or None,
            "text": text,
        }
        if not payload["document_id"]:
            st.error("document_id is required.")
        elif not text.strip():
            st.error("text is required.")
        else:
            with st.spinner("Calling POST /ingest…"):
                status, data = api_post(base_url, "/ingest", payload)
            st.markdown(f"**HTTP {status}**")
            if isinstance(data, dict) and status == 200:
                st.success(f"Indexed {data.get('chunks_indexed', 0)} chunks for `{data.get('document_id')}`")
            st.json(data)

with ask_tab:
    st.subheader("Ask a question")
    st.markdown("Send a question to `POST /ask`. Citations come from `retrieved_chunk_ids`.")

    question = st.text_input(
        "question",
        value="What is Newton's second law?",
        key="ask_question",
    )

    if st.button("Ask", type="primary", key="ask_btn"):
        if not question.strip():
            st.error("question is required.")
        else:
            with st.spinner("Calling POST /ask…"):
                status, data = api_post(base_url, "/ask", {"question": question.strip()})
            render_ask_result(status, data)

    st.markdown("**Try these for Maven proof**")
    st.markdown(
        "- **In docs:** `What is Newton's second law?` → answer + `newtons-laws::…` chunk IDs\n"
        f"- **Not in docs:** `How do you bake a chocolate cake?` → `{REFUSAL_PHRASE}`"
    )
