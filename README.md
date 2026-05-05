# BLKPHXLABS — Dissertation RAG Pipeline

A local retrieval-augmented generation (RAG) pipeline that lets you chat with a PDF dissertation using Google Gemini. Built with PyMuPDF, ChromaDB, sentence-transformers, and Streamlit.

---

## Overview

```
dissertation.pdf
      │
      ▼
extraction.py  →  dissertation.md      (PDF → Markdown)
      │
      ▼
chunker.py     →  dissertation.jsonl   (Markdown → chunks with metadata)
      │
      ▼
embedder.py    →  chroma_db/           (chunks → vector store)
      │
      ▼
app.py                                 (Streamlit chat UI powered by Gemini)
```

---

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Set your Gemini API key**

Get a free key at [aistudio.google.com](https://aistudio.google.com), then set it in your terminal:

```powershell
$env:GEMINI_API_KEY="your-key-here"
```

---

## Usage

### Step 1 — Convert PDF to Markdown

```bash
python extraction.py dissertation.pdf dissertation.md
```

Detects headings from font sizes and outputs structured Markdown with page markers.

### Step 2 — Chunk the Markdown

```bash
python chunker.py dissertation.md dissertation.jsonl
```

Splits at heading boundaries, caps chunks at ~768 tokens with 12% overlap, and writes each chunk as a JSON line with metadata (`chunk_index`, `header_title`, `source_file`, `timestamp`).

### Step 3 — Embed and store

```bash
python embedder.py dissertation.jsonl
```

Embeds each chunk using `all-MiniLM-L6-v2` (downloads ~80MB on first run) and stores them in a local ChromaDB vector store at `./chroma_db`.

### Step 4 — Chat

**Streamlit UI (recommended)**

```bash
python -m streamlit run app.py
```

**Command-line interface**

```bash
python chat.py
```

---

## Scripts

| File | Description |
|---|---|
| `extraction.py` | Converts a PDF to Markdown using PyMuPDF |
| `chunker.py` | Splits Markdown into RAG-ready JSONL chunks |
| `embedder.py` | Embeds chunks and loads them into ChromaDB |
| `app.py` | Streamlit chat UI |
| `chat.py` | Terminal chat interface |

---

## Notes

- The PDF, Markdown, JSONL, and `chroma_db/` are excluded from version control — run the pipeline locally to generate them.
- Gemini model can be changed by editing `GEMINI_MODEL` in `app.py` or `chat.py`.
- ChromaDB persists locally — you only need to run `embedder.py` once per document.
