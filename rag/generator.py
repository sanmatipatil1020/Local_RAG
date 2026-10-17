"""
rag/generator.py
─────────────────
Sends a question + retrieved context to the local Ollama LLM
and returns a grounded answer.

Privacy guarantee:
  - All calls go to http://localhost:11434 (Ollama running on YOUR machine).
  - Zero data is sent to any cloud AI provider.

How it works:
  1. We receive the top-K chunks from the retriever.
  2. We build a prompt that instructs Mistral to answer ONLY from
     those chunks and to cite file names.
  3. We call Ollama's /api/generate endpoint via HTTP POST.
  4. We return the answer text plus the source citations.
"""

import logging
import requests
from typing import List, Dict, Any, Optional

import config

logger = logging.getLogger(__name__)


def _build_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Build a RAG prompt that instructs the LLM to answer only from context.

    We include the filename (and page number if available) with each chunk
    so the model can reference them in its answer.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        source_label = chunk["filename"]
        if chunk.get("page") and chunk["page"] != -1:
            source_label += f" (page {chunk['page']})"
        context_parts.append(f"[Source {i}: {source_label}]\n{chunk['text']}")

    context_block = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant. Answer the user's question based ONLY on the provided context below.
If the answer is not found in the context, say "I could not find relevant information in the indexed documents."
Do not make up facts. You may quote short excerpts from the context when helpful.
Always mention which source(s) (by filename and page number if given) your answer is based on.

=== CONTEXT ===
{context_block}

=== QUESTION ===
{query}

=== ANSWER ==="""

    return prompt


def generate_answer(
    query: str,
    chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Call Ollama to generate an answer grounded in the retrieved chunks.

    Returns:
    {
        "answer":  str,           # LLM-generated answer text
        "sources": list[dict],    # deduplicated source citations
        "model":   str,           # model name used
        "error":   str | None,    # error message if something went wrong
    }
    """
    if not chunks:
        return {
            "answer":  "No relevant documents found. Please upload documents and try again.",
            "sources": [],
            "model":   config.OLLAMA_MODEL,
            "error":   None,
        }

    prompt = _build_prompt(query, chunks)

    try:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model":  config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,          # Wait for full response
                "options": {
                    "temperature": 0.1,   # Low temperature = factual, grounded
                    "num_predict": 512,   # Max output tokens
                },
            },
            timeout=config.OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        answer_text = data.get("response", "").strip()

    except requests.exceptions.ConnectionError:
        error_msg = (
            "Cannot connect to Ollama. "
            "Please ensure Ollama is running: open a terminal and run 'ollama serve'."
        )
        logger.error(error_msg)
        return {"answer": error_msg, "sources": [], "model": config.OLLAMA_MODEL, "error": error_msg}

    except requests.exceptions.Timeout:
        error_msg = f"Ollama timed out after {config.OLLAMA_TIMEOUT}s. The model may be loading."
        logger.error(error_msg)
        return {"answer": error_msg, "sources": [], "model": config.OLLAMA_MODEL, "error": error_msg}

    except Exception as e:
        error_msg = f"LLM generation error: {str(e)}"
        logger.error(error_msg)
        return {"answer": error_msg, "sources": [], "model": config.OLLAMA_MODEL, "error": error_msg}

    # Build deduplicated source list for the frontend
    sources = _build_sources(chunks)

    return {
        "answer":  answer_text,
        "sources": sources,
        "model":   config.OLLAMA_MODEL,
        "error":   None,
    }


def _build_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build a deduplicated list of source citations from retrieved chunks.

    Each source includes:
    - filename
    - page (or null)
    - excerpt (first 200 chars of the chunk)
    - file_type
    """
    seen = set()
    sources = []
    for chunk in chunks:
        key = (chunk["filename"], chunk.get("page"))
        if key not in seen:
            seen.add(key)
            page = chunk.get("page")
            sources.append({
                "filename":  chunk["filename"],
                "filepath":  chunk.get("filepath", ""),
                "page":      page if (page and page != -1) else None,
                "excerpt":   chunk["text"][:220].strip() + "…" if len(chunk["text"]) > 220 else chunk["text"],
                "file_type": chunk.get("file_type", ""),
                "score":     chunk.get("rrf_score", chunk.get("score", 0)),
            })
    return sources


def check_ollama_available() -> Dict[str, Any]:
    """
    Check whether Ollama is running and whether the configured model is available.
    Returns: {"available": bool, "model_found": bool, "models": list[str], "error": str|None}
    """
    try:
        resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models_data = resp.json().get("models", [])
        model_names = [m["name"] for m in models_data]
        # Check if our model is in the list (handle "mistral:latest" vs "mistral")
        model_found = any(
            m.startswith(config.OLLAMA_MODEL) for m in model_names
        )
        return {
            "available":   True,
            "model_found": model_found,
            "models":      model_names,
            "error":       None,
        }
    except Exception as e:
        return {
            "available":   False,
            "model_found": False,
            "models":      [],
            "error":       str(e),
        }
