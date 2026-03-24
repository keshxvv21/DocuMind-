import os
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

# PDF reading
import fitz  # PyMuPDF

# HuggingFace embeddings (free, no API key needed)
from sentence_transformers import SentenceTransformer

# Vector search
import faiss

# LLM — using HuggingFace pipeline (free)


# ─── Config ────────────────────────────────────────────────────────────────────

UPLOAD_FOLDER = "uploads"
VECTOR_STORE_PATH = "vector_store/store.pkl"
CHUNK_SIZE = 300          # words per chunk
CHUNK_OVERLAP = 50        # overlapping words between chunks
TOP_K = 4                 # how many chunks to retrieve
EMBED_MODEL = "all-MiniLM-L6-v2"   # fast, free, good quality
  

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("vector_store", exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB max

# ─── Load Models (once at startup) ─────────────────────────────────────────────

print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Loading LLM...")

print("✅ Models ready!")

# ─── In-memory store (chunks + FAISS index) ────────────────────────────────────

class VectorStore:
    def __init__(self):
        self.chunks = []          # list of {"text": ..., "source": ..., "chunk_id": ...}
        self.index = None         # FAISS index
        self.dim = 384            # MiniLM embedding dimension

    def add_chunks(self, new_chunks):
        """Embed new chunks and add to FAISS index."""
        texts = [c["text"] for c in new_chunks]
        vectors = embedder.encode(texts, show_progress_bar=False)
        vectors = np.array(vectors, dtype="float32")

        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dim)

        self.index.add(vectors)
        self.chunks.extend(new_chunks)
        self._save()

    def search(self, query, k=TOP_K):
        """Return top-k relevant chunks for the query."""
        if self.index is None or len(self.chunks) == 0:
            return []

        q_vec = embedder.encode([query], show_progress_bar=False)
        q_vec = np.array(q_vec, dtype="float32")

        distances, indices = self.index.search(q_vec, min(k, len(self.chunks)))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    **self.chunks[idx],
                    "score": float(dist)
                })
        return results

    def clear(self):
        self.chunks = []
        self.index = None
        if os.path.exists(VECTOR_STORE_PATH):
            os.remove(VECTOR_STORE_PATH)

    def _save(self):
        with open(VECTOR_STORE_PATH, "wb") as f:
            pickle.dump({"chunks": self.chunks, "index": self.index}, f)

    def load(self):
        if os.path.exists(VECTOR_STORE_PATH):
            with open(VECTOR_STORE_PATH, "rb") as f:
                data = pickle.load(f)
                self.chunks = data["chunks"]
                self.index = data["index"]

store = VectorStore()
store.load()  # Load persisted data if any

# ─── Utility Functions ──────────────────────────────────────────────────────────

def extract_text_from_pdf(path):
    """Extract all text from a PDF file."""
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_into_chunks(text, source_name):
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    i = 0
    chunk_id = 0
    while i < len(words):
        chunk_words = words[i: i + CHUNK_SIZE]
        chunk_text = " ".join(chunk_words)
        if len(chunk_text.strip()) > 20:   # skip near-empty chunks
            chunks.append({
                "text": chunk_text,
                "source": source_name,
                "chunk_id": chunk_id
            })
        i += CHUNK_SIZE - CHUNK_OVERLAP
        chunk_id += 1
    return chunks

import requests

def generate_answer(question, context_chunks):
    context = "\n\n---\n\n".join([c["text"] for c in context_chunks])
    
    prompt = (
        f"Answer using ONLY the context below. "
        f"If answer not found, say 'This information is not in your notes.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "tinyllama",
        "prompt": prompt,
        "stream": False
    })
    
    return response.json()["response"].strip()
# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    allowed_ext = {".pdf", ".txt"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        return jsonify({"error": "Only PDF and TXT files allowed"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Extract text
    if ext == ".pdf":
        text = extract_text_from_pdf(filepath)
    else:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    if len(text.strip()) < 50:
        return jsonify({"error": "Could not extract enough text from file"}), 400

    # Chunk + embed + store
    chunks = split_into_chunks(text, filename)
    store.add_chunks(chunks)

    return jsonify({
        "message": f"✅ '{filename}' processed successfully!",
        "chunks": len(chunks),
        "total_chunks": len(store.chunks)
    })

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question is empty"}), 400

    if len(store.chunks) == 0:
        return jsonify({"error": "No notes uploaded yet. Please upload a file first."}), 400

    # Retrieve relevant chunks
    relevant_chunks = store.search(question, k=TOP_K)

    # Generate answer
    answer = generate_answer(question, relevant_chunks)

    # Return answer + sources for citation highlighting
    sources = list({c["source"] for c in relevant_chunks})
    excerpts = [c["text"][:200] + "..." for c in relevant_chunks]

    return jsonify({
        "answer": answer,
        "sources": sources,
        "excerpts": excerpts
    })

@app.route("/files", methods=["GET"])
def list_files():
    """Return list of uploaded files and chunk count per file."""
    file_counts = {}
    for chunk in store.chunks:
        src = chunk["source"]
        file_counts[src] = file_counts.get(src, 0) + 1
    return jsonify({"files": file_counts})

@app.route("/clear", methods=["POST"])
def clear():
    store.clear()
    return jsonify({"message": "All notes cleared."})

# ─── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
