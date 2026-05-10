from __future__ import annotations

import io
import re
import uuid
from dataclasses import dataclass
from typing import Iterable

import chromadb
from chromadb.config import Settings
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4


@dataclass
class Chunk:
    text: str
    page: int
    chunk_id: str
    source: str


def load_document(file_bytes: bytes, filename: str) -> list[tuple[int, str]]:
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = _normalise_whitespace(text)
            if text.strip():
                pages.append((i, text))
        return pages
    if name.endswith(".txt"):
        text = _normalise_whitespace(file_bytes.decode("utf-8", errors="ignore"))
        return [(1, text)] if text.strip() else []
    raise ValueError(f"Unsupported file type: {filename}. Upload a .pdf or .txt file.")


def _normalise_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_pages(
    pages: list[tuple[int, str]],
    source: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    chunks: list[Chunk] = []
    for page_no, page_text in pages:
        for piece in _split_recursive(page_text, chunk_size):
            _append_with_overlap(chunks, piece, page_no, source, chunk_size, overlap)
    return chunks


def _append_with_overlap(
    chunks: list[Chunk],
    piece: str,
    page_no: int,
    source: str,
    chunk_size: int,
    overlap: int,
) -> None:
    if not chunks or chunks[-1].page != page_no or len(chunks[-1].text) + len(piece) + 1 > chunk_size:
        seed = ""
        if chunks and overlap > 0 and chunks[-1].page == page_no:
            seed = chunks[-1].text[-overlap:]
        new_text = (seed + " " + piece).strip() if seed else piece
        chunks.append(
            Chunk(
                text=new_text,
                page=page_no,
                chunk_id=str(uuid.uuid4()),
                source=source,
            )
        )
        return
    chunks[-1] = Chunk(
        text=(chunks[-1].text + " " + piece).strip(),
        page=chunks[-1].page,
        chunk_id=chunks[-1].chunk_id,
        source=chunks[-1].source,
    )


def _split_recursive(text: str, chunk_size: int) -> Iterable[str]:
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            yield para
            continue
        for sentence in _split_sentences(para):
            if len(sentence) <= chunk_size:
                yield sentence
            else:
                yield from _split_words(sentence, chunk_size)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _split_words(text: str, chunk_size: int) -> Iterable[str]:
    words = text.split(" ")
    buf: list[str] = []
    length = 0
    for w in words:
        if length + len(w) + 1 > chunk_size and buf:
            yield " ".join(buf)
            buf, length = [], 0
        buf.append(w)
        length += len(w) + 1
    if buf:
        yield " ".join(buf)


_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def build_collection(chunks: list[Chunk], collection_name: str) -> chromadb.Collection:
    if not chunks:
        raise ValueError("No chunks to index — the document appears empty.")

    client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    embedder = get_embedder()
    texts = [c.text for c in chunks]
    embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=False).tolist()

    collection.add(
        ids=[c.chunk_id for c in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"page": c.page, "source": c.source} for c in chunks],
    )
    return collection


def retrieve(collection: chromadb.Collection, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    embedder = get_embedder()
    query_vec = embedder.encode([query]).tolist()
    res = collection.query(query_embeddings=query_vec, n_results=top_k)
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append(
            {
                "text": doc,
                "page": meta.get("page"),
                "source": meta.get("source"),
                "score": 1 - dist,
            }
        )
    return hits


SYSTEM_PROMPT = """You are a careful research assistant answering questions about a document the user has uploaded.

Rules — these are non-negotiable:
1. Use ONLY the provided context excerpts to answer. If the answer is not present, say "I could not find that in the document." Do not fall back on outside knowledge.
2. When you state a fact, cite the page it came from like (p. 4). If multiple pages support a point, cite each one.
3. Quote short phrases verbatim when precision matters; paraphrase otherwise.
4. Keep the answer focused and concise. If the user asks for examples or steps, format them as a list.
"""


def _format_context(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(f"[Excerpt {i} | page {h['page']}]\n{h['text']}")
    return "\n\n".join(blocks)


def generate_answer(
    groq_client: Groq,
    query: str,
    hits: list[dict],
    model: str = LLM_MODEL_NAME,
) -> str:
    context = _format_context(hits)
    user_message = (
        f"Context excerpts from the document:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the excerpts above and cite page numbers."
    )
    response = groq_client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content or ""
