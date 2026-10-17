# Privacy-Preserving Local RAG System

**Final Year Project** | Python · Flask · ChromaDB · Ollama · Sentence Transformers

A fully local, privacy-preserving Retrieval-Augmented Generation (RAG) system.
Ask questions about your documents using a local LLM — **zero data sent to cloud APIs**.

---

## Features

| Feature | Implementation |
|---|---|
| Web dashboard + chat UI | Vanilla HTML/CSS/JS (dark glassmorphism) |
| Python Flask backend | REST API (`/api/chat`, `/api/documents`, etc.) |
| PDF ingestion with page numbers | `pypdf` |
| DOCX ingestion | `python-docx` |
| TXT, Java (.java), C# (.cs), Python (.py), Markdown (.md) | Built-in reader |
| Text chunking | Sliding-window, configurable size + overlap |
| Local embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Persistent vector database | `ChromaDB` (local storage) |
| Semantic search | ChromaDB cosine similarity |
| Keyword search | `rank-bm25` (BM25Okapi) |
| Hybrid retrieval | Reciprocal Rank Fusion (RRF) |
| Local LLM answer generation | Ollama (`mistral`) |
| Automatic folder monitoring | `watchdog` |
| Source citations (filename + page) | Real metadata from ChromaDB |
| Document management + rescan | Via dashboard or API |
| 100% local after setup | ✅ |

---

## Prerequisites

1. **Python 3.10+** — [python.org](https://python.org)
2. **Ollama** — [ollama.ai](https://ollama.ai)
   - After installing, run: `ollama pull mistral`

---

## Quick Start (Windows)

### First time only:
```bat
setup.bat
```

### Every time you want to run:
```bat
REM In a separate terminal (keep it open):
ollama serve

REM Then run the app:
run.bat
```

Open **http://localhost:5000** in your browser.

---

## Manual Start (without .bat files)

```powershell
# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Start the server
python app.py
```

---

## How to Use

1. **Upload documents** — Click the **Documents** tab → drag and drop or use the file picker.
   - Or simply copy files into the `documents/` folder — the system auto-detects them.
2. **Ask questions** — Click the **Chat** tab → type your question → press Enter.
3. **View sources** — Every answer includes clickable source cards with filename, page number, and excerpt.
4. **Manage documents** — Delete documents from the index or trigger a manual rescan.
5. **Monitor system** — Click **System Status** to see Ollama, ChromaDB, and monitor health.

---

## Project Structure

```
RAG_Final_New/
├── app.py                    # Flask backend — main entry point
├── config.py                 # All configuration (paths, model, chunk size)
├── requirements.txt          # Python dependencies
├── run.bat                   # Windows launch script
├── setup.bat                 # Windows first-time setup
│
├── rag/
│   ├── document_processor.py # PDF/DOCX/TXT/Java/C# extraction + chunking
│   ├── embedder.py           # Sentence Transformers local embedding
│   ├── vector_store.py       # ChromaDB CRUD operations
│   ├── retriever.py          # Hybrid retrieval (semantic + BM25 + RRF)
│   ├── generator.py          # Ollama LLM answer generation
│   ├── indexer.py            # Indexing orchestrator + status tracking
│   └── monitor.py            # watchdog file system monitor
│
├── documents/                # Drop your documents here (auto-watched)
├── chroma_db/                # ChromaDB persistent storage (auto-created)
│
├── frontend/
│   ├── index.html            # Main SPA dashboard
│   ├── css/style.css         # Full design system
│   └── js/
│       ├── app.js            # Chat logic
│       ├── documents.js      # Document management
│       └── status.js         # System status panel
│
└── tests/
    └── test_rag.py           # Pytest test suite (34 tests)
```

---

## Running Tests

```powershell
.\venv\Scripts\Activate.ps1
pytest tests\ -v
```

---

## Configuration

Edit `config.py` to change:

| Setting | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `"mistral"` | Any Ollama model (`llama3`, `phi3`, etc.) |
| `EMBEDDING_MODEL` | `"all-MiniLM-L6-v2"` | Sentence Transformers model |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `80` | Overlap between chunks |
| `TOP_K_FINAL` | `5` | Chunks sent to LLM |
| `DOCUMENTS_DIR` | `./documents` | Watched folder path |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | System health (Ollama, ChromaDB, monitor) |
| `POST` | `/api/chat` | Ask a question → grounded answer + sources |
| `GET` | `/api/documents` | List all indexed documents |
| `POST` | `/api/documents/upload` | Upload a document file |
| `POST` | `/api/documents/rescan` | Rescan documents/ folder |
| `POST` | `/api/documents/delete` | Remove a document from the index |
| `GET` | `/api/status` | File monitor status |

---

## Privacy Guarantee

- **No cloud APIs** — all inference runs locally via Ollama.
- **No telemetry** — the app makes zero outbound network requests (except to Ollama on localhost).
- **Local vector store** — ChromaDB stores all embeddings on your local disk.
- **Local embeddings** — Sentence Transformers runs entirely on your CPU/GPU.

---

## SRS Compliance Checklist

| # | SRS Requirement | Status |
|---|---|---|
| 1 | Professional web dashboard and chat interface | ✅ Implemented |
| 2 | Python Flask backend | ✅ Implemented |
| 3 | PDF, DOCX, TXT, Java, C# file ingestion | ✅ Implemented |
| 4 | Text extraction and chunking | ✅ Implemented |
| 5 | Local embeddings using Sentence Transformers | ✅ Implemented |
| 6 | Persistent vector database using ChromaDB | ✅ Implemented |
| 7 | Hybrid retrieval (semantic + BM25 keyword) | ✅ Implemented |
| 8 | Local answer generation via Ollama Mistral | ✅ Implemented |
| 9 | Automatic monitoring for new/modified/deleted files | ✅ Implemented |
| 10 | Real source filenames, excerpts, PDF page numbers | ✅ Implemented |
| 11 | Document management, indexing status, error handling, manual rescan | ✅ Implemented |
| 12 | Fully local processing after initial setup | ✅ Implemented |
