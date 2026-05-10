from __future__ import annotations

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import sys

try:
    import pysqlite3

    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import re

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from rag_pipeline import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    LLM_MODEL_NAME,
    build_collection,
    chunk_pages,
    generate_answer,
    load_document,
    retrieve,
)

load_dotenv()

st.set_page_config(page_title="NotebookLM Clone", page_icon="📚", layout="wide")


def _safe_collection_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:50] or "doc"
    if not cleaned[0].isalnum():
        cleaned = "d" + cleaned
    return cleaned


def _get_api_key() -> str | None:
    if "GROQ_API_KEY" in st.session_state and st.session_state["GROQ_API_KEY"]:
        return st.session_state["GROQ_API_KEY"]
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY")


def _reset_chat() -> None:
    st.session_state.messages = []


def main() -> None:
    st.title("📚 Chat with your document")
    st.caption(
        "Upload a PDF or text file, then ask questions. Answers are grounded in the document, "
        "with page-level citations."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "collection" not in st.session_state:
        st.session_state.collection = None
    if "doc_name" not in st.session_state:
        st.session_state.doc_name = None
    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0

    with st.sidebar:
        st.header("⚙️ Setup")

        api_key_input = st.text_input(
            "Groq API key",
            type="password",
            value=_get_api_key() or "",
            help="Get a free key from https://console.groq.com",
        )
        if api_key_input:
            st.session_state["GROQ_API_KEY"] = api_key_input

        st.divider()
        st.subheader("📄 Document")
        uploaded = st.file_uploader("Upload a PDF or .txt file", type=["pdf", "txt"])

        with st.expander("Advanced chunking settings"):
            chunk_size = st.slider("Chunk size (chars)", 300, 2000, DEFAULT_CHUNK_SIZE, 50)
            chunk_overlap = st.slider("Chunk overlap (chars)", 0, 500, DEFAULT_CHUNK_OVERLAP, 25)
            top_k = st.slider("Top-K chunks per query", 1, 10, DEFAULT_TOP_K, 1)

        process = st.button("📥 Process document", type="primary", use_container_width=True)

        if process:
            if not uploaded:
                st.error("Please upload a file first.")
            else:
                with st.spinner("Reading, chunking, embedding…"):
                    file_bytes = uploaded.getvalue()
                    pages = load_document(file_bytes, uploaded.name)
                    if not pages:
                        st.error("Could not extract any text from this document.")
                    else:
                        chunks = chunk_pages(
                            pages,
                            source=uploaded.name,
                            chunk_size=chunk_size,
                            overlap=chunk_overlap,
                        )
                        collection = build_collection(
                            chunks, _safe_collection_name(uploaded.name)
                        )
                        st.session_state.collection = collection
                        st.session_state.doc_name = uploaded.name
                        st.session_state.chunk_count = len(chunks)
                        _reset_chat()
                st.success(
                    f"Indexed **{uploaded.name}** — {len(pages)} pages, "
                    f"{st.session_state.chunk_count} chunks."
                )

        if st.session_state.doc_name:
            st.info(
                f"📌 Active document: **{st.session_state.doc_name}**\n\n"
                f"{st.session_state.chunk_count} chunks indexed."
            )
            if st.button("🔁 Clear conversation", use_container_width=True):
                _reset_chat()
                st.rerun()

        st.divider()
        st.caption(
            "Stack: Streamlit · pypdf · sentence-transformers · ChromaDB · Groq "
            f"({LLM_MODEL_NAME})"
        )

    if not st.session_state.collection:
        st.info("👈 Upload a document and click **Process document** to start chatting.")
        st.stop()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📎 Retrieved chunks ({len(msg['sources'])})"):
                    for src in msg["sources"]:
                        st.markdown(
                            f"**Page {src['page']}** · similarity {src['score']:.2f}"
                        )
                        st.text(src["text"])
                        st.markdown("---")

    prompt = st.chat_input("Ask a question about the document…")
    if not prompt:
        return

    api_key = _get_api_key()
    if not api_key:
        st.error("Please set your Groq API key in the sidebar first.")
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document and drafting an answer…"):
            try:
                hits = retrieve(st.session_state.collection, prompt, top_k=top_k)
                groq_client = Groq(api_key=api_key)
                answer = generate_answer(groq_client, prompt, hits)
            except Exception as exc:
                err = f"Something went wrong: `{exc}`"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
                return

        st.markdown(answer)
        with st.expander(f"📎 Retrieved chunks ({len(hits)})"):
            for h in hits:
                st.markdown(f"**Page {h['page']}** · similarity {h['score']:.2f}")
                st.text(h["text"])
                st.markdown("---")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": hits}
    )


if __name__ == "__main__":
    main()
