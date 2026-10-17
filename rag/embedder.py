"""
rag/embedder.py
───────────────
Wraps Sentence Transformers to produce local embeddings.

Key points:
  - The model runs entirely on your CPU (or GPU if available).
  - No text is sent to any external server.
  - The model is loaded once and cached in memory for speed.
  - We use 'all-MiniLM-L6-v2': 384-dimensional vectors, ~80MB download.

Usage:
    emb = get_embedder()
    vectors = emb.encode(["hello world", "another sentence"])
"""

import logging
from sentence_transformers import SentenceTransformer
import config

logger = logging.getLogger(__name__)

# Module-level cache — model is loaded once per process
_embedder = None


def get_embedder() -> SentenceTransformer:
    """
    Return the cached SentenceTransformer model instance.
    Loads from disk on first call; returns cached instance thereafter.
    """
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully.")
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings and return a list of float vectors.
    Each vector has 384 dimensions (for all-MiniLM-L6-v2).
    """
    model = get_embedder()
    # show_progress_bar=False keeps logs clean in production
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string. Returns one vector.
    """
    return embed_texts([query])[0]
