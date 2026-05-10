# 📚 NotebookLM Clone — Chat with your Documents (RAG)

A lightweight, fully open-source clone of Google NotebookLM. Upload any PDF or
text file, and have a grounded conversation with it — every answer is built
from the document itself and cites the page it came from.

> Built for **GenAI Assignment 3** (Scaler).

---

## ✨ Features

- 📄 Upload any PDF or `.txt` file
- 🧩 Recursive paragraph-aware **chunking** with sliding-window overlap
- 🧠 Local **sentence-transformers** embeddings — no embedding API cost
- 🗂 In-process **ChromaDB** vector index — no external DB to host
- ⚡ **Groq**-hosted Llama 3.3 70B for fast, free generation
- 🔎 Page-level **citations** plus a "show retrieved chunks" panel for transparency
- 🛡 System prompt forces the LLM to refuse questions it cannot ground in the doc

---

## 🏗 Architecture

```
┌────────────┐   bytes    ┌──────────────┐  pages   ┌──────────────────┐
│  Streamlit │──────────▶│ pypdf reader │─────────▶│ recursive chunker│
│  uploader  │            └──────────────┘          └─────────┬────────┘
└────────────┘                                                │ chunks
                                                              ▼
┌──────────────────┐  hits   ┌──────────────────┐    ┌──────────────────┐
│    Groq LLM      │◀────────│   ChromaDB       │◀───│ sentence-trans-  │
│ (llama-3.3-70b)  │  query  │ cosine top-k     │    │ formers MiniLM   │
└────────┬─────────┘         └──────────────────┘    └──────────────────┘
         │ grounded answer + page citations
         ▼
   Streamlit chat UI
```

### Pipeline stages

| Stage | What it does | File |
|------|--------------|------|
| Ingest | Extract page-aware text from the upload | `rag_pipeline.load_document` |
| Chunk | Recursive paragraph + sentence + word splitter with overlap | `rag_pipeline.chunk_pages` |
| Embed | Encode each chunk with `all-MiniLM-L6-v2` (384-dim) | `rag_pipeline.get_embedder` |
| Store | Cosine-distance ChromaDB collection in-process | `rag_pipeline.build_collection` |
| Retrieve | Top-K nearest chunks for the user query | `rag_pipeline.retrieve` |
| Generate | Groq Llama 3.3 70B prompted with retrieved context only | `rag_pipeline.generate_answer` |

---

## 🧩 Chunking strategy (documented)

Chunking is the most important knob in any RAG system, so it is implemented
explicitly rather than borrowed from a framework default.

The strategy in `rag_pipeline.chunk_pages` is **recursive
paragraph-aware splitting with a sliding-window overlap**:

1. Text is normalised (line endings, repeated whitespace, blank-line collapse).
2. For each PDF page, the text is split on blank-line paragraph breaks.
3. Paragraphs are greedily packed into a chunk until the configured
   `chunk_size` (default **900 chars**) is reached.
4. When a chunk closes, the next chunk is seeded with the last
   `overlap` characters of the previous one (default **150 chars**) so that
   ideas spanning a chunk boundary are not lost.
5. **Fallback ladder** for oversized paragraphs: split on sentence enders
   (`. ! ?`), then on word boundaries. This is the "recursive" part — we
   try the most semantically meaningful split first and only fall back
   when forced.
6. Every chunk records its **source filename and page number** so answers
   can cite where each fact came from.

Chunk size, overlap, and top-K are all tunable from the sidebar so you can
A/B different settings on the same document.

### Why these defaults?

- `all-MiniLM-L6-v2` has a 256-token effective window. **900 chars** comfortably
  fits within it while leaving room for context.
- **150-char overlap** (~17%) is the well-tested industry sweet spot — enough
  to bridge boundaries, small enough to avoid retrieval duplicates.
- **Top-K = 4** balances precision (smaller K) against recall (larger K) for
  most single-document Q&A.

---

## 🧱 Tech stack

| Layer | Choice | Why |
|------|--------|-----|
| UI | Streamlit | Pure Python, zero JS, free hosting on Streamlit Community Cloud |
| PDF parsing | pypdf | Pure-Python, no system deps |
| Chunking | Custom (this repo) | Page-aware, fully documented |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Free, local, fast (384 dims) |
| Vector DB | ChromaDB (ephemeral, in-process) | Zero-ops, no server to host |
| LLM | Groq Llama 3.3 70B Versatile | Free tier, very fast inference |

---

## 🚀 Run it locally

### 1. Clone & install

```bash
git clone https://github.com/shubhamkumar9199/GenAi-Assignment-3.git
cd GenAi-Assignment-3
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure your Groq key

Get a free key at <https://console.groq.com/keys>, then either:

- Create a `.env` file:
  ```
  GROQ_API_KEY=gsk_your_key_here
  ```
- Or paste it into the sidebar at runtime.

### 3. Launch

```bash
streamlit run app.py
```

The app opens at <http://localhost:8501>. Upload a PDF, click
**Process document**, and start asking questions.

---

## ☁️ Deploy to Streamlit Community Cloud (free)

1. Push this repo to GitHub (it's already public).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **New app** → pick this repo → main branch → `app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Click **Deploy**. Your live URL will look like
   `https://<your-app>.streamlit.app`.

> The first cold start downloads the 90 MB MiniLM model — give it ~60 seconds
> the first time. Subsequent visits are instant.

---

## 📁 Project structure

```
.
├── app.py              # Streamlit UI + chat session
├── rag_pipeline.py     # Ingest, chunk, embed, retrieve, generate
├── requirements.txt    # Python deps
├── .env.example        # API key template
├── .gitignore
├── .streamlit/
│   └── config.toml     # Theme + server settings
└── README.md
```

---

## 📝 License

MIT — use it, learn from it, ship your own.
