import hashlib
import os
import threading
import uuid
from pathlib import Path

import chromadb
import torch
from docx import Document
from flask import Flask, jsonify, render_template, request
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from werkzeug.utils import secure_filename

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
DB_DIR = BASE_DIR / "file_rag_database"
ALLOWED_EXTENSIONS = {"txt", "pdf", "docx"}
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-small"
COLLECTION_NAME = "local_rag_documents"

UPLOAD_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
model_lock = threading.Lock()
embedding_model = None
tokenizer = None
llm = None
device = "cuda" if torch.cuda.is_available() else "cpu"

client = chromadb.PersistentClient(path=str(DB_DIR), settings=chromadb.config.Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
    embedding_function=None,
)


def extension_allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_sections(path):
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return [{"text": path.read_text(encoding="utf-8-sig", errors="replace"), "section": "text file"}]
    if suffix == ".pdf":
        sections = []
        for page_number, page in enumerate(PdfReader(str(path)).pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                sections.append({"text": text, "section": f"page {page_number}"})
        return sections
    if suffix == ".docx":
        document = Document(str(path))
        parts = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return [{"text": "\n".join(parts), "section": "document body"}]
    raise ValueError("Only TXT, PDF, and DOCX files are supported.")


def make_chunks(sections, source_name, source_id, chunk_size=60, overlap=10):
    chunks, metadata = [], []
    for section in sections:
        words = section["text"].split()
        for start in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[start:start + chunk_size]).strip()
            if chunk:
                chunks.append(chunk)
                metadata.append({"source": source_name, "source_id": source_id, "section": section["section"]})
            if start + chunk_size >= len(words):
                break
    return chunks, metadata


def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        with model_lock:
            if embedding_model is None:
                embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    return embedding_model


def index_file(path, source_id):
    sections = extract_sections(path)
    chunks, metadata = make_chunks(sections, path.name, source_id)
    if not any(s["text"].strip() for s in sections) or not chunks:
        raise ValueError("The file has no readable text. Scanned PDFs require OCR.")
    # Re-uploading a file replaces its old chunks instead of creating duplicates.
    collection.delete(where={"source_id": source_id})
    vectors = get_embedding_model().encode(chunks, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    ids = [f"{source_id}_{i}_{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
    collection.upsert(ids=ids, documents=chunks, embeddings=vectors.tolist(), metadatas=metadata)
    return {"chunks": len(chunks), "characters": sum(len(s["text"]) for s in sections)}


def list_files():
    files = []
    for path in sorted(UPLOAD_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if path.is_file():
            source_id = hashlib.sha256(path.name.encode()).hexdigest()[:16]
            files.append({"name": path.name, "size": path.stat().st_size, "source_id": source_id})
    return files


def retrieve(query, top_k=4):
    if collection.count() == 0:
        return []
    vector = get_embedding_model().encode([query], normalize_embeddings=True, convert_to_numpy=True)
    result = collection.query(query_embeddings=vector.tolist(), n_results=min(top_k, collection.count()), include=["documents", "metadatas", "distances"])
    return [{"text": doc, "metadata": meta, "distance": distance} for doc, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])]


def generate_answer(query, matches):
    if not matches:
        return "I don't know yet. Upload and index a document first."
    global tokenizer, llm
    if tokenizer is None or llm is None:
        with model_lock:
            if tokenizer is None or llm is None:
                tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
                llm = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME).to(device)
                llm.eval()
    prefix = "Answer using only the context. If the answer is missing, say you do not know.\nContext:\n"
    suffix = f"\nQuestion: {query}\nAnswer:"
    context = ""
    for match in matches:
        candidate = context + match["text"] + "\n"
        if len(tokenizer.encode(prefix + candidate + suffix)) <= 512:
            context = candidate
    prompt = prefix + context + suffix
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = llm.generate(**inputs, max_new_tokens=150, do_sample=False)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


@app.get("/")
def index():
    return render_template("index.html", files=list_files(), chunks=collection.count(), device=device)


@app.get("/api/files")
def api_files():
    return jsonify({"files": list_files(), "chunks": collection.count()})


@app.post("/api/upload")
def upload():
    incoming = request.files.getlist("files")
    if not incoming or all(not f.filename for f in incoming):
        return jsonify({"error": "Choose at least one TXT, PDF, or DOCX file."}), 400
    results, errors = [], []
    for file in incoming:
        if not file.filename or not extension_allowed(file.filename):
            errors.append(f"{file.filename or 'Unnamed file'}: unsupported type")
            continue
        original = secure_filename(file.filename)
        path = UPLOAD_DIR / original
        if path.exists():
            path = UPLOAD_DIR / f"{path.stem}_{uuid.uuid4().hex[:6]}{path.suffix.lower()}"
        try:
            file.save(path)
            source_id = hashlib.sha256(path.name.encode()).hexdigest()[:16]
            info = index_file(path, source_id)
            results.append({"name": path.name, **info})
        except Exception as exc:
            if path.exists():
                path.unlink()
            errors.append(f"{file.filename}: {exc}")
    status = 207 if results and errors else (400 if errors and not results else 200)
    return jsonify({"uploaded": results, "errors": errors, "chunks": collection.count()}), status


@app.post("/api/query")
def query():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()
    if not question:
        return jsonify({"error": "Enter a question."}), 400
    try:
        matches = retrieve(question)
        answer = generate_answer(question, matches)
        sources = [{"name": m["metadata"].get("source", "Unknown"), "section": m["metadata"].get("section", "")} for m in matches]
        return jsonify({"answer": answer, "sources": sources})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Files must be 50 MB or smaller."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
