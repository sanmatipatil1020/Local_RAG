"""
rag/indexer.py
──────────────
High-level indexing orchestrator.

Provides the functions called by both:
  - Flask API endpoints (manual rescan, upload)
  - The file monitor (automatic indexing on file events)

Tracks per-file indexing status in a thread-safe in-memory dictionary.
"""

import os
import logging
import threading
from typing import Dict, Any, List
from datetime import datetime

import config
from rag.document_processor import process_file
from rag.vector_store import add_document_chunks, delete_document, list_documents

logger = logging.getLogger(__name__)

# ─── Indexing Status Registry ─────────────────────────────────────────────────
# Maps filepath → status dict
# This is read by the /api/documents endpoint so users can see progress.
_status_lock = threading.Lock()
_file_status: Dict[str, Dict[str, Any]] = {}


def _set_status(filepath: str, status: str, error: str = None, chunk_count: int = 0):
    """Update the in-memory status for a file."""
    with _status_lock:
        _file_status[filepath] = {
            "filepath":    filepath,
            "filename":    os.path.basename(filepath),
            "status":      status,   # "indexing" | "indexed" | "error" | "deleted"
            "error":       error,
            "chunk_count": chunk_count,
            "updated_at":  datetime.now().isoformat(),
        }


def get_all_statuses() -> List[Dict[str, Any]]:
    """Return a snapshot of all file statuses."""
    with _status_lock:
        return list(_file_status.values())


def get_status(filepath: str) -> Dict[str, Any]:
    """Return the status for a specific file, or None if not tracked."""
    with _status_lock:
        return _file_status.get(filepath)


# ─── Core Indexing Functions ──────────────────────────────────────────────────

def index_file(filepath: str) -> Dict[str, Any]:
    """
    Index a single file:
      1. Extract text + chunk it.
      2. Remove old chunks from ChromaDB (handles re-indexing on file change).
      3. Embed and store new chunks.
      4. Update status registry.

    Returns the status dict for this file.
    """
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filepath)[1].lower()

    if ext not in config.SUPPORTED_EXTENSIONS:
        logger.warning(f"Skipping unsupported file type: {filename}")
        return {"status": "skipped", "filename": filename}

    logger.info(f"Indexing: {filename}")
    _set_status(filepath, "indexing")

    try:
        # Step 1: Extract + chunk
        chunks = process_file(filepath)
        if not chunks:
            _set_status(filepath, "error", error="No text could be extracted.")
            return get_status(filepath)

        # Step 2: Remove any previously indexed version of this file
        delete_document(filepath)

        # Step 3: Embed + store
        count = add_document_chunks(chunks)
        _set_status(filepath, "indexed", chunk_count=count)
        logger.info(f"Indexed '{filename}': {count} chunks stored.")
        return get_status(filepath)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error indexing '{filename}': {error_msg}")
        _set_status(filepath, "error", error=error_msg)
        return get_status(filepath)


def remove_file(filepath: str) -> Dict[str, Any]:
    """
    Remove a file from the index:
      1. Delete all its chunks from ChromaDB.
      2. Mark its status as "deleted".
    """
    filename = os.path.basename(filepath)
    logger.info(f"Removing from index: {filename}")

    try:
        deleted_count = delete_document(filepath)
        _set_status(filepath, "deleted")
        return {"status": "deleted", "filename": filename, "chunks_removed": deleted_count}
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error removing '{filename}': {error_msg}")
        _set_status(filepath, "error", error=error_msg)
        return {"status": "error", "filename": filename, "error": error_msg}


def index_directory(directory: str = None) -> Dict[str, Any]:
    """
    Scan a directory and index all supported files.
    Used on startup and for manual rescan.

    Returns a summary: {"indexed": int, "skipped": int, "errors": int}
    """
    directory = directory or config.DOCUMENTS_DIR
    os.makedirs(directory, exist_ok=True)

    results = {"indexed": 0, "skipped": 0, "errors": 0, "files": []}

    for root, _, files in os.walk(directory):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in config.SUPPORTED_EXTENSIONS:
                results["skipped"] += 1
                continue

            filepath = os.path.join(root, fname)
            status = index_file(filepath)

            if status.get("status") == "indexed":
                results["indexed"] += 1
            elif status.get("status") == "error":
                results["errors"] += 1
            else:
                results["skipped"] += 1

            results["files"].append(status)

    logger.info(
        f"Directory scan complete: {results['indexed']} indexed, "
        f"{results['skipped']} skipped, {results['errors']} errors."
    )
    return results


def sync_status_with_store():
    """
    On startup, populate _file_status from ChromaDB
    so the UI shows existing indexed files without re-indexing.
    """
    from rag.vector_store import list_documents
    docs = list_documents()
    for doc in docs:
        fp = doc["filepath"]
        # Only mark as indexed if the file still exists on disk
        if os.path.exists(fp):
            _set_status(fp, "indexed", chunk_count=doc["chunk_count"])
        else:
            _set_status(fp, "deleted")
