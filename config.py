"""
config.py — Central configuration for the RAG system.
All paths, model names, and tunable parameters are in one place.
Students: Change MODEL_NAME to any Ollama model you have downloaded.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder the system watches for new / modified / deleted documents
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")

# ChromaDB stores its data here — persistent across restarts
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# ChromaDB collection name
COLLECTION_NAME = "rag_documents"

# ─── Embedding model (local, no internet needed after first download) ──────────
# all-MiniLM-L6-v2 is fast, ~80 MB, good accuracy for most use-cases
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ─── Ollama (local LLM) ───────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral"          # Change to any model you have: llama3, phi3, etc.
OLLAMA_TIMEOUT = 120              # seconds to wait for LLM response

# ─── Chunking parameters ──────────────────────────────────────────────────────
CHUNK_SIZE = 512        # characters per chunk (not tokens)
CHUNK_OVERLAP = 80      # characters of overlap between consecutive chunks

# ─── Retrieval parameters ─────────────────────────────────────────────────────
TOP_K_SEMANTIC = 10     # candidates from ChromaDB semantic search
TOP_K_BM25 = 10         # candidates from BM25 keyword search
TOP_K_FINAL = 5         # final chunks sent to LLM after hybrid re-ranking

# ─── Supported file types ─────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".java", ".cs", ".py", ".md"}

# ─── Flask ────────────────────────────────────────────────────────────────────
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False     # Set True only for development
