"""
rag/retriever.py
─────────────────
Implements HYBRID RETRIEVAL by combining:
  1. Semantic search  — ChromaDB cosine similarity on embeddings
  2. Keyword search   — BM25 (Best Match 25) using rank-bm25

The two result sets are merged using Reciprocal Rank Fusion (RRF),
a standard technique that boosts chunks appearing in BOTH result lists.

Why hybrid?
  - Pure semantic search struggles with exact keyword matches (e.g., a
    specific function name or a proper noun).
  - Pure BM25 misses synonyms and paraphrases.
  - Combining both gives better recall and precision.

Reciprocal Rank Fusion formula:
  RRF_score(d) = sum over each ranker: 1 / (k + rank_of_d)
  where k=60 is a smoothing constant that prevents top-1 from dominating.
"""

import logging
from typing import List, Dict, Any

import config
from rag.embedder import embed_query
from rag.vector_store import query_similar, get_collection

logger = logging.getLogger(__name__)

# RRF smoothing constant
RRF_K = 60


def _bm25_search(query: str, top_k: int) -> List[Dict[str, Any]]:
    """
    Perform BM25 keyword search over ALL chunks currently in ChromaDB.

    We fetch all documents from ChromaDB and run BM25 in-memory.
    For large corpora (>50k chunks) this may be slow; for a final-year
    project it is perfectly fine.
    """
    from rank_bm25 import BM25Okapi

    collection = get_collection()
    if collection.count() == 0:
        return []

    # Retrieve all stored chunks
    all_results = collection.get(include=["documents", "metadatas"])
    all_texts = all_results["documents"]
    all_metas = all_results["metadatas"]

    if not all_texts:
        return []

    # Tokenise (simple whitespace split — adequate for BM25)
    tokenised_corpus = [doc.lower().split() for doc in all_texts]
    bm25 = BM25Okapi(tokenised_corpus)

    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)

    # Pair each chunk with its BM25 score
    scored = sorted(
        zip(scores, all_texts, all_metas),
        key=lambda x: x[0],
        reverse=True,
    )

    results = []
    for score, text, meta in scored[:top_k]:
        results.append({
            "text":        text,
            "filename":    meta.get("filename", ""),
            "filepath":    meta.get("filepath", ""),
            "page":        meta.get("page", -1),
            "chunk_index": meta.get("chunk_index", 0),
            "file_type":   meta.get("file_type", ""),
            "score":       round(float(score), 4),
        })

    return results


def _rrf_merge(
    semantic_results: List[Dict],
    bm25_results: List[Dict],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    Each unique chunk receives a combined RRF score.
    Chunks appearing in both lists score higher.
    """
    rrf_scores: Dict[str, float] = {}
    chunk_data: Dict[str, Dict] = {}

    def _chunk_key(chunk: Dict) -> str:
        """Unique key per chunk: filepath + chunk_index."""
        return f"{chunk['filepath']}::{chunk['chunk_index']}"

    # Score from semantic ranking
    for rank, chunk in enumerate(semantic_results, start=1):
        key = _chunk_key(chunk)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (RRF_K + rank)
        chunk_data[key] = chunk

    # Score from BM25 ranking
    for rank, chunk in enumerate(bm25_results, start=1):
        key = _chunk_key(chunk)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (RRF_K + rank)
        if key not in chunk_data:
            chunk_data[key] = chunk

    # Sort by combined RRF score (descending)
    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

    final = []
    for key in sorted_keys[:top_k]:
        item = chunk_data[key].copy()
        item["rrf_score"] = round(rrf_scores[key], 6)
        final.append(item)

    return final


def hybrid_retrieve(query: str, top_k: int = None) -> List[Dict[str, Any]]:
    """
    Main retrieval function. Call this from the Flask endpoint.

    1. Embed the query.
    2. Run semantic search (ChromaDB).
    3. Run BM25 keyword search (in-memory).
    4. Merge with RRF.
    5. Return top_k results, each containing:
       - text, filename, filepath, page, file_type, rrf_score
    """
    top_k = top_k or config.TOP_K_FINAL

    logger.info(f"Hybrid retrieval for query: '{query[:80]}...' (top_k={top_k})")

    # Step 1: embed query
    query_vec = embed_query(query)

    # Step 2: semantic search
    semantic_results = query_similar(query_vec, top_k=config.TOP_K_SEMANTIC)
    logger.info(f"Semantic search returned {len(semantic_results)} results")

    # Step 3: BM25 search
    bm25_results = _bm25_search(query, top_k=config.TOP_K_BM25)
    logger.info(f"BM25 search returned {len(bm25_results)} results")

    # Step 4: RRF merge
    merged = _rrf_merge(semantic_results, bm25_results, top_k=top_k)
    logger.info(f"After RRF merge: {len(merged)} final chunks")

    return merged
