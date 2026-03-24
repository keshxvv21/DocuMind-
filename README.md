# DocuMind — Ask Your Notes

> Upload your study notes. Ask questions. Get answers — straight from your documents.

A local, zero-cost **Retrieval-Augmented Generation (RAG)** study assistant built with Python Flask. No API keys, no cloud services, no cost — runs entirely on your machine.

---

## What it does

Traditional search (Ctrl+F) only finds exact words. **Ask Your Notes** understands the *meaning* of your question and finds the most relevant section from your uploaded notes — even if the exact words don't match.

**Example:**
- Upload: `Physics Chapter 5.pdf`
- Ask: `"Explain Kirchhoff's Voltage Law"`
- Get: A concise answer generated only from your document

---

## Tech Stack

| Layer | Technology | Cost |
|---|---|---|
| Backend | Python + Flask | Free |
| PDF Parsing | PyMuPDF (fitz) | Free |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace) | Free / Local |
| Vector DB | FAISS (faiss-cpu) | Free |
| LLM | TinyLlama via Ollama | Free / Local |
| Frontend | HTML + CSS + JS | Free |

---

## How the RAG Pipeline Works

```
User uploads PDF
      ↓
Text extracted (PyMuPDF)
      ↓
Split into 300-word chunks (50-word overlap)
      ↓
Each chunk → 384-dim vector (MiniLM embedding)
      ↓
Vectors stored in FAISS index → saved as store.pkl
      ↓
User asks a question → question embedded
      ↓
FAISS finds Top-4 most similar chunks
      ↓
TinyLlama generates answer from those 4 chunks only
      ↓
Answer + source excerpts shown in chat UI
```

---

## Project Structure

```
ask-your-notes/
├── app.py                  ← Flask backend (upload, chunking, embedding, search, answer)
├── requirements.txt        ← Python dependencies
├── README.md
├── uploads/                ← Raw uploaded PDF/TXT files
│   └── your-file.pdf
└── vector_store/
    └── store.pkl           ← FAISS index + text chunks (the actual search database)
```

---

## Setup & Run

### Prerequisites
- Python 3.x (any version)
- [Ollama](https://ollama.com) installed:
  ```bash
  brew install ollama        # macOS
  ```
- TinyLlama model pulled:
  ```bash
  ollama pull tinyllama
  ```

### Install dependencies
```bash
pip install flask pymupdf sentence-transformers faiss-cpu numpy werkzeug requests
```

### Run
```bash
# Terminal 1 — keep this running
ollama serve

# Terminal 2
python app.py
```

Open your browser at: `http://localhost:5000`

---

## Chunking Strategy

| Parameter | Value | Reason |
|---|---|---|
| Chunk size | 300 words | Enough context, precise retrieval |
| Chunk overlap | 50 words | Prevents losing boundary sentences |
| Top-K results | 4 chunks | Best balance of context vs noise |
| Embedding dim | 384 | MiniLM-L6-v2 output size |

---

## Why Ollama + TinyLlama?

The original plan was Flan-T5 via HuggingFace transformers, but the `tokenizers` package fails to build on Python 3.13 due to a Rust/PyO3 compilation issue. Ollama runs as a standalone binary outside of Python — no dependency conflicts, 2-command install, and serves local LLMs via a REST API on `localhost:11434`.

---

## Future Scope

- Multi-file search across multiple PDFs simultaneously
- Source highlighting — show exact sentence in PDF
- Voice input support
- Chat history export (PDF / text)
- Upgrade to Mistral-7B for better answer quality
- React frontend for better state management

---

## Resume Description

> Built a Retrieval-Augmented Generation (RAG) system enabling users to query personal study documents with contextual AI responses. Used FAISS for vector similarity search, HuggingFace sentence-transformers for semantic embeddings, and Ollama (TinyLlama) as a local LLM. Backend: Python + Flask. Frontend: HTML/CSS/JS.

---

*B.Tech CSE Project · Uka Tarsadia University · 2026*
