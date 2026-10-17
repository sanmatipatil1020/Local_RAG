"""
rag/vector_store.py
────────────────────
Manages the ChromaDB persistent vector database.

Responsibilities:
  - Initialize and return the ChromaDB collection.
  - Add chunks (text + embedding + metadata) for a document.
  - Delete all chunks belonging to a document (by filepath).
  - Query for the top-K most similar chunks to a query vector.
  - List all unique documents currently indexed.

ChromaDB stores data in the CHROMA_DIR folder and persists between restarts.
Each chunk is stored as a ChromaDB "document" with:
  - id:        unique string (filepath + chunk_index)
  - embedding: 384-float vector
  - document:  the chunk text
  - metadata:  filename, filepath, page, chunk_index, file_type
"""

import logging
import os
import hashlib
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings

import config

logger = logging.getLogger(__name__)

# Module-level cache
_client: Optional[chromadb.PersistentClient] = None
_collection = None


def _get_client():
    """Return (or create) the persistent ChromaDB client."""
    global _client
    if _client is None:
        os.makedirs(config.CHROMA_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        logger.info(f"ChromaDB client initialized at: {config.CHROMA_DIR}")
    return _client


def get_collection():
    """Return (or create) the ChromaDB collection."""
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )
        logger.info(f"ChromaDB collection '{config.COLLECTION_NAME}' ready. "
                    f"Documents in store: {_collection.count()}")
    return _collection


def _make_chunk_id(filepath: str, chunk_index: int) -> str:
    """
    Create a stable, unique ID for a chunk.
    We hash the filepath to keep IDs short and filesystem-safe.
    """
    path_hash = hashlib.md5(filepath.encode()).hexdigest()[:12]
    return f"{path_hash}_{chunk_index}"


def add_document_chunks(chunks: List[Dict[str, Any]]) -> int:
    """
    Add a list of chunk records (from document_processor.process_file)
    to ChromaDB with their embeddings.

    Returns the number of chunks actually stored.
    """
    if not chunks:
        return 0

    from rag.embedder import embed_texts

    collection = get_collection()
    texts = [c["text"] for c in chunks]

    logger.info(f"Embedding {len(texts)} chunks for '{chunks[0]['filename']}'...")
    embeddings = embed_texts(texts)

    ids = []
    metadatas = []
    documents = []

    for i, chunk in enumerate(chunks):
        chunk_id = _make_chunk_id(chunk["filepath"], chunk["chunk_index"])
        ids.append(chunk_id)
        documents.append(chunk["text"])
        metadatas.append({
            "filename":    chunk["filename"],
            "filepath":    chunk["filepath"],
            "page":        chunk["page"] if chunk["page"] is not None else -1,
            "chunk_index": chunk["chunk_index"],
            "file_type":   chunk["file_type"],
        })

    # ChromaDB upsert: adds new or updates existing chunks
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    logger.info(f"Stored {len(ids)} chunks for '{chunks[0]['filename']}'")
    return len(ids)


def delete_document(filepath: str) -> int:
    """
    Remove all chunks belonging to a specific file (matched by filepath metadata).
    Returns the number of chunks deleted.
    """
    collection = get_collection()

    # Query all chunks for this filepath
    results = collection.get(
        where={"filepath": filepath},
        include=["metadatas"],
    )

    if not results["ids"]:
        logger.info(f"No chunks found for '{filepath}' — nothing to delete.")
        return 0

    ids_to_delete = results["ids"]
    collection.delete(ids=ids_to_delete)
    logger.info(f"Deleted {len(ids_to_delete)} chunks for '{filepath}'")
    return len(ids_to_delete)


def query_similar(
    query_embedding: List[float],
    top_k: int = None,
) -> List[Dict[str, Any]]:
    """
    Find the top_k most semantically similar chunks to the query vector.

    Returns a list of dicts:
    {
        "text":     str,
        "filename": str,
        "filepath": str,
        "page":     int | None,
        "score":    float,   # cosine similarity (0–1, higher = better)
    }
    """
    top_k = top_k or config.TOP_K_SEMANTIC
    collection = get_collection()

    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for text, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB returns cosine distance (0 = identical, 2 = opposite).
        # Convert to similarity score in [0, 1]:
        score = 1 - (distance / 2)
        output.append({
            "text":        text,
            "filename":    meta.get("filename", ""),
            "filepath":    meta.get("filepath", ""),
            "page":        meta.get("page", -1),
            "chunk_index": meta.get("chunk_index", 0),
            "file_type":   meta.get("file_type", ""),
            "score":       round(score, 4),
        })

    return output


def list_documents() -> List[Dict[str, Any]]:
    """
    Return a summary of all unique documents currently in the vector store.
    Each entry: {"filename": str, "filepath": str, "chunk_count": int, "file_type": str}
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    # Fetch all metadata (no embeddings needed)
    results = collection.get(include=["metadatas"])
    seen = {}  # filepath → summary dict

    for meta in results["metadatas"]:
        fp = meta.get("filepath", "")
        if fp not in seen:
            seen[fp] = {
                "filename":    meta.get("filename", ""),
                "filepath":    fp,
                "file_type":   meta.get("file_type", ""),
                "chunk_count": 0,
            }
        seen[fp]["chunk_count"] += 1

    return list(seen.values())


def get_total_chunks() -> int:
    """Return total number of chunks stored in ChromaDB."""
    return get_collection().count()
