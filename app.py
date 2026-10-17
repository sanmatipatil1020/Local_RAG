"""
app.py — Main Flask application entry point
────────────────────────────────────────────
Privacy-Preserving Local RAG System
Final Year Project

All AI processing happens locally:
  - Embeddings:  Sentence Transformers (local model)
  - Vector DB:   ChromaDB (local persistent storage)
  - LLM:         Ollama Mistral (local inference)
  - No document content ever leaves this machine.

API Endpoints:
  GET  /                          → serves frontend HTML
  GET  /api/health                → system health check
  POST /api/chat                  → ask a question (main RAG endpoint)
  GET  /api/documents             → list indexed documents
  POST /api/documents/upload      → upload a document via browser
  POST /api/documents/rescan      → trigger manual rescan of documents/ folder
  DELETE /api/documents/<path>    → remove a document from the index
  GET  /api/status                → file monitor + system status
"""

import os
import sys
import logging
import threading
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add project root to path so imports work from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ─── Flask App ────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder="frontend",
    static_url_path="",
)
CORS(app)  # Allow frontend running on the same origin

# ─── Ensure Required Directories Exist ───────────────────────────────────────
os.makedirs(config.DOCUMENTS_DIR, exist_ok=True)
os.makedirs(config.CHROMA_DIR, exist_ok=True)

# ─── Startup Flag ─────────────────────────────────────────────────────────────
_startup_done = False


def startup_tasks():
    """
    Run once when the first request comes in (or on startup thread).
    Loads the embedding model, syncs ChromaDB status, starts file monitor.
    """
    global _startup_done
    if _startup_done:
        return
    _startup_done = True

    logger.info("=== RAG System Startup ===")

    # 1. Pre-load the embedding model so first query isn't slow
    try:
        from rag.embedder import get_embedder
        get_embedder()
        logger.info("Embedding model loaded.")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")

    # 2. Sync in-memory status with what's already in ChromaDB
    try:
        from rag.indexer import sync_status_with_store
        sync_status_with_store()
        logger.info("Status synced with ChromaDB.")
    except Exception as e:
        logger.error(f"Status sync failed: {e}")

    # 3. Start the file monitor (background thread)
    try:
        from rag.monitor import start_monitor
        start_monitor(config.DOCUMENTS_DIR)
        logger.info(f"File monitor started on: {config.DOCUMENTS_DIR}")
    except Exception as e:
        logger.error(f"File monitor failed to start: {e}")

    logger.info("=== Startup Complete ===")


# Run startup in a background thread so Flask starts immediately
threading.Thread(target=startup_tasks, daemon=True).start()


# ═══════════════════════════════════════════════════════════════════════════════
# Frontend Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the frontend SPA."""
    return send_from_directory("frontend", "index.html")


@app.route("/<path:path>")
def static_files(path):
    """Serve frontend static files (CSS, JS)."""
    return send_from_directory("frontend", path)


# ═══════════════════════════════════════════════════════════════════════════════
# API: Health Check
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def api_health():
    """
    Returns the health of all system components:
    - Embedding model status
    - ChromaDB status (total chunks)
    - Ollama availability and model check
    - File monitor status
    """
    from rag.generator import check_ollama_available
    from rag.vector_store import get_total_chunks, get_collection
    from rag.monitor import get_monitor_status

    # Check ChromaDB
    try:
        get_collection()
        chroma_ok = True
        total_chunks = get_total_chunks()
    except Exception as e:
        chroma_ok = False
        total_chunks = 0

    # Check Ollama
    ollama_info = check_ollama_available()

    # Monitor status
    monitor_info = get_monitor_status()

    return jsonify({
        "status":       "ok",
        "timestamp":    datetime.now().isoformat(),
        "embedding": {
            "model":   config.EMBEDDING_MODEL,
            "loaded":  _startup_done,
        },
        "vector_store": {
            "ok":           chroma_ok,
            "total_chunks": total_chunks,
            "directory":    config.CHROMA_DIR,
        },
        "ollama": {
            "available":   ollama_info["available"],
            "model":       config.OLLAMA_MODEL,
            "model_found": ollama_info["model_found"],
            "all_models":  ollama_info["models"],
        },
        "monitor": monitor_info,
        "documents_dir": config.DOCUMENTS_DIR,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# API: Chat (Main RAG Endpoint)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Main RAG pipeline:
      1. Accept a question from the user.
      2. Retrieve relevant chunks (hybrid: semantic + BM25).
      3. Generate a grounded answer via Ollama.
      4. Return answer + source citations.

    Request body (JSON): {"query": "your question here"}
    Response (JSON): {
        "answer":   str,
        "sources":  list[source],
        "model":    str,
        "query":    str,
        "chunks_found": int,
    }
    """
    data = request.get_json(silent=True)
    if not data or not data.get("query"):
        return jsonify({"error": "Missing 'query' in request body."}), 400

    query = data["query"].strip()
    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    logger.info(f"Chat query: '{query[:80]}'")

    from rag.retriever import hybrid_retrieve
    from rag.generator import generate_answer
    from rag.vector_store import get_total_chunks

    # Check if there's anything in the index
    if get_total_chunks() == 0:
        return jsonify({
            "answer":      "No documents are indexed yet. Please upload documents to the 'documents' folder or use the upload button.",
            "sources":     [],
            "model":       config.OLLAMA_MODEL,
            "query":       query,
            "chunks_found": 0,
        })

    # Retrieve relevant chunks
    try:
        chunks = hybrid_retrieve(query, top_k=config.TOP_K_FINAL)
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return jsonify({"error": f"Retrieval failed: {str(e)}"}), 500

    # Generate answer
    try:
        result = generate_answer(query, chunks)
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({"error": f"Answer generation failed: {str(e)}"}), 500

    return jsonify({
        "answer":       result["answer"],
        "sources":      result["sources"],
        "model":        result["model"],
        "query":        query,
        "chunks_found": len(chunks),
        "error":        result.get("error"),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# API: Document Management
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/documents", methods=["GET"])
def api_documents():
    """
    List all documents known to the system (both indexed and errored).
    Merges the in-memory status registry with what's in ChromaDB.
    """
    from rag.indexer import get_all_statuses
    from rag.vector_store import get_total_chunks

    statuses = get_all_statuses()
    total_chunks = get_total_chunks()

    return jsonify({
        "documents":    statuses,
        "total_files":  len(statuses),
        "total_chunks": total_chunks,
    })


@app.route("/api/documents/upload", methods=["POST"])
def api_upload():
    """
    Accept a file upload via multipart/form-data.
    Saves it to the documents/ folder and indexes it immediately.
    The file monitor will also detect it, but we index proactively.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file in request. Use field name 'file'."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Filename is empty."}), 400

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in config.SUPPORTED_EXTENSIONS:
        return jsonify({
            "error": f"File type '{ext}' is not supported. "
                     f"Supported: {', '.join(sorted(config.SUPPORTED_EXTENSIONS))}"
        }), 400

    save_path = os.path.join(config.DOCUMENTS_DIR, filename)
    file.save(save_path)
    logger.info(f"File uploaded: {filename} → {save_path}")

    # Index in a background thread so response returns immediately
    from rag.indexer import index_file
    threading.Thread(target=index_file, args=(save_path,), daemon=True).start()

    return jsonify({
        "message":  f"File '{filename}' uploaded and indexing started.",
        "filepath": save_path,
        "status":   "indexing",
    })


@app.route("/api/documents/rescan", methods=["POST"])
def api_rescan():
    """
    Manually trigger a full rescan of the documents/ folder.
    Useful if files were added outside the running application.
    Runs in a background thread; returns immediately.
    """
    from rag.indexer import index_directory

    logger.info("Manual rescan triggered.")

    def _run_rescan():
        result = index_directory(config.DOCUMENTS_DIR)
        logger.info(f"Rescan complete: {result}")

    threading.Thread(target=_run_rescan, daemon=True).start()

    return jsonify({
        "message":     "Rescan started in background.",
        "watched_dir": config.DOCUMENTS_DIR,
    })


@app.route("/api/documents/delete", methods=["POST"])
def api_delete_document():
    """
    Remove a document from the vector index by its filepath.
    Does NOT delete the file from disk (to be safe).
    Request body: {"filepath": "/absolute/path/to/file.pdf"}
    """
    data = request.get_json(silent=True)
    if not data or not data.get("filepath"):
        return jsonify({"error": "Missing 'filepath' in request body."}), 400

    filepath = data["filepath"]
    from rag.indexer import remove_file
    result = remove_file(filepath)

    return jsonify(result)


# ═══════════════════════════════════════════════════════════════════════════════
# API: System Status
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/status", methods=["GET"])
def api_status():
    """Return file monitor and system status."""
    from rag.monitor import get_monitor_status
    return jsonify(get_monitor_status())


# ═══════════════════════════════════════════════════════════════════════════════
# Error Handlers
# ═══════════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error.", "detail": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Privacy-Preserving Local RAG System")
    print(f"  URL:         http://localhost:{config.FLASK_PORT}")
    print(f"  Documents:   {config.DOCUMENTS_DIR}")
    print(f"  ChromaDB:    {config.CHROMA_DIR}")
    print(f"  LLM Model:   {config.OLLAMA_MODEL} via Ollama")
    print("="*60 + "\n")

    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
        use_reloader=False,   # Disable reloader to prevent double startup
    )
