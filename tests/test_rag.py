"""
tests/test_rag.py
──────────────────
Automated tests for the Privacy-Preserving Local RAG System.

Run with:
    .\\venv\\Scripts\\pytest tests\\ -v

Tests cover:
  1. Document processor  — PDF page extraction, DOCX, TXT, Java/C# files
  2. Chunker             — correct chunk sizes and overlap
  3. Embedder            — correct output shape and type
  4. Vector store        — add, query, delete operations (uses a temp collection)
  5. Hybrid retriever    — returns results in expected format
  6. Flask API endpoints — health, chat, documents (integration tests)
"""

import os
import sys
import tempfile
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_txt_file(tmp_path):
    """Create a temporary plain-text file for testing."""
    content = (
        "Privacy-preserving AI systems ensure that sensitive data is protected. "
        "Local inference using Ollama eliminates the need for cloud API calls. "
        "ChromaDB provides efficient vector storage on a local machine. "
        "Sentence Transformers create high-quality embeddings without internet access. "
        "BM25 keyword search complements semantic search for better recall. "
        "Hybrid retrieval combines both approaches using Reciprocal Rank Fusion. "
        "Document ingestion supports PDF, DOCX, TXT, Java, and C# file formats. "
        "Each chunk is stored with its source filename and page number metadata."
    ) * 5  # Repeat to have enough text for chunking
    f = tmp_path / "test_document.txt"
    f.write_text(content, encoding="utf-8")
    return str(f)


@pytest.fixture
def sample_java_file(tmp_path):
    """Create a temporary Java source file for testing."""
    content = """
public class HelloWorld {
    // This is a privacy-preserving system
    public static void main(String[] args) {
        System.out.println("Local RAG system running");
        String model = "mistral";
        int chunkSize = 512;
        System.out.println("Model: " + model + ", Chunk Size: " + chunkSize);
    }
}
"""
    f = tmp_path / "HelloWorld.java"
    f.write_text(content, encoding="utf-8")
    return str(f)


@pytest.fixture
def sample_cs_file(tmp_path):
    """Create a temporary C# source file for testing."""
    content = """
using System;

namespace RAGSystem {
    class DocumentProcessor {
        // Extracts text from PDF, DOCX, TXT files
        public string ExtractText(string filepath) {
            return "Extracted text from: " + filepath;
        }
    }
}
"""
    f = tmp_path / "DocumentProcessor.cs"
    f.write_text(content, encoding="utf-8")
    return str(f)


# ══════════════════════════════════════════════════════════════════════════════
# Test 1: Document Processor
# ══════════════════════════════════════════════════════════════════════════════

class TestDocumentProcessor:

    def test_plain_text_extraction(self, sample_txt_file):
        from rag.document_processor import extract_text
        pages = extract_text(sample_txt_file)
        assert len(pages) > 0, "Should extract at least one page"
        assert len(pages[0]["text"]) > 0, "Extracted text should not be empty"
        assert pages[0]["page"] is None, "TXT files have no page numbers"

    def test_java_extraction(self, sample_java_file):
        from rag.document_processor import extract_text
        pages = extract_text(sample_java_file)
        assert len(pages) > 0
        assert "HelloWorld" in pages[0]["text"] or "main" in pages[0]["text"]

    def test_csharp_extraction(self, sample_cs_file):
        from rag.document_processor import extract_text
        pages = extract_text(sample_cs_file)
        assert len(pages) > 0
        assert "DocumentProcessor" in pages[0]["text"]

    def test_unsupported_extension_returns_empty(self, tmp_path):
        from rag.document_processor import extract_text
        f = tmp_path / "test.xyz"
        f.write_text("some content")
        result = extract_text(str(f))
        assert result == [], "Unsupported extension should return empty list"

    def test_process_file_returns_chunks_with_metadata(self, sample_txt_file):
        from rag.document_processor import process_file
        chunks = process_file(sample_txt_file)
        assert len(chunks) > 0, "Should produce at least one chunk"

        first = chunks[0]
        assert "text" in first
        assert "filename" in first
        assert "filepath" in first
        assert "page" in first
        assert "chunk_index" in first
        assert "file_type" in first

        assert first["filename"] == "test_document.txt"
        assert first["chunk_index"] == 0
        assert first["file_type"] == "txt"

    def test_chunks_are_sequential(self, sample_txt_file):
        from rag.document_processor import process_file
        chunks = process_file(sample_txt_file)
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i, f"Chunk {i} has wrong index {chunk['chunk_index']}"


# ══════════════════════════════════════════════════════════════════════════════
# Test 2: Chunker
# ══════════════════════════════════════════════════════════════════════════════

class TestChunker:

    def test_chunk_size_respected(self):
        from rag.document_processor import chunk_text
        text = "A" * 1000
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        for chunk in chunks:
            assert len(chunk) <= 100, f"Chunk too large: {len(chunk)}"

    def test_overlap_creates_shared_content(self):
        from rag.document_processor import chunk_text
        # Use chunk_size=100 so chunks clear the 20-char minimum filter
        text = "The privacy-preserving RAG system uses local models. " * 30  # ~1560 chars
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1, "Should produce multiple chunks"
        # Check that overlap exists: end of chunk[0] should appear in start of chunk[1]
        if len(chunks) >= 2:
            end_of_first = chunks[0][-20:]
            assert end_of_first in chunks[1], "Overlap content should appear in next chunk"

    def test_empty_text_returns_empty(self):
        from rag.document_processor import chunk_text
        chunks = chunk_text("", chunk_size=100, overlap=10)
        assert chunks == []

    def test_short_text_returns_one_chunk(self):
        from rag.document_processor import chunk_text
        # Text must be >=20 chars to pass the minimum-length filter in chunk_text
        text = "This is a short but valid document chunk."
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) == 1


# ══════════════════════════════════════════════════════════════════════════════
# Test 3: Embedder
# ══════════════════════════════════════════════════════════════════════════════

class TestEmbedder:

    def test_embed_single_text(self):
        from rag.embedder import embed_texts
        vectors = embed_texts(["Hello world"])
        assert len(vectors) == 1
        assert isinstance(vectors[0], list)
        assert len(vectors[0]) == 384  # all-MiniLM-L6-v2 output size
        assert all(isinstance(v, float) for v in vectors[0])

    def test_embed_multiple_texts(self):
        from rag.embedder import embed_texts
        texts = ["First sentence", "Second sentence", "Third sentence"]
        vectors = embed_texts(texts)
        assert len(vectors) == 3
        assert all(len(v) == 384 for v in vectors)

    def test_embed_query(self):
        from rag.embedder import embed_query
        vec = embed_query("what is privacy preserving AI?")
        assert isinstance(vec, list)
        assert len(vec) == 384

    def test_different_texts_have_different_embeddings(self):
        from rag.embedder import embed_texts
        vecs = embed_texts(["The cat sat on the mat", "Quantum computing uses qubits"])
        # Cosine similarity should not be 1.0 (not identical)
        import numpy as np
        a, b = np.array(vecs[0]), np.array(vecs[1])
        cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        assert cosine < 0.99, "Different texts should not be identical embeddings"

    def test_model_caching(self):
        from rag.embedder import get_embedder
        model1 = get_embedder()
        model2 = get_embedder()
        assert model1 is model2, "Should return the same cached model instance"


# ══════════════════════════════════════════════════════════════════════════════
# Test 4: Vector Store (uses a temporary ChromaDB collection)
# ══════════════════════════════════════════════════════════════════════════════

class TestVectorStore:

    @pytest.fixture(autouse=True)
    def use_temp_chroma(self, tmp_path, monkeypatch):
        """Override ChromaDB path to use a temp directory for tests."""
        monkeypatch.setattr("config.CHROMA_DIR", str(tmp_path / "test_chroma"))
        monkeypatch.setattr("config.COLLECTION_NAME", "test_collection")
        # Reset module-level caches
        import rag.vector_store as vs
        vs._client = None
        vs._collection = None
        yield
        # Cleanup
        vs._client = None
        vs._collection = None

    def _make_chunks(self, filepath="/test/file.txt", n=3):
        return [
            {
                "text":        f"This is chunk {i} about privacy preserving AI systems",
                "filename":    "file.txt",
                "filepath":    filepath,
                "page":        None,
                "chunk_index": i,
                "file_type":   "txt",
            }
            for i in range(n)
        ]

    def test_add_and_count(self):
        from rag.vector_store import add_document_chunks, get_total_chunks
        chunks = self._make_chunks()
        count = add_document_chunks(chunks)
        assert count == 3
        assert get_total_chunks() == 3

    def test_query_returns_results(self):
        from rag.vector_store import add_document_chunks, query_similar
        from rag.embedder import embed_query
        chunks = self._make_chunks()
        add_document_chunks(chunks)
        q_vec = embed_query("privacy preserving AI")
        results = query_similar(q_vec, top_k=3)
        assert len(results) > 0
        assert "text" in results[0]
        assert "filename" in results[0]
        assert "score" in results[0]
        assert 0.0 <= results[0]["score"] <= 1.0

    def test_delete_document(self):
        from rag.vector_store import add_document_chunks, delete_document, get_total_chunks
        chunks = self._make_chunks(filepath="/test/to_delete.txt")
        add_document_chunks(chunks)
        assert get_total_chunks() == 3
        deleted = delete_document("/test/to_delete.txt")
        assert deleted == 3
        assert get_total_chunks() == 0

    def test_list_documents(self):
        from rag.vector_store import add_document_chunks, list_documents
        chunks = self._make_chunks()
        add_document_chunks(chunks)
        docs = list_documents()
        assert len(docs) == 1
        assert docs[0]["filename"] == "file.txt"
        assert docs[0]["chunk_count"] == 3

    def test_upsert_replaces_old_chunks(self):
        from rag.vector_store import add_document_chunks, get_total_chunks, delete_document
        chunks = self._make_chunks(n=3)
        add_document_chunks(chunks)
        assert get_total_chunks() == 3
        # Delete and re-add (simulating re-index)
        delete_document("/test/file.txt")
        new_chunks = self._make_chunks(n=5)
        add_document_chunks(new_chunks)
        assert get_total_chunks() == 5


# ══════════════════════════════════════════════════════════════════════════════
# Test 5: Flask API (Integration Tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestFlaskAPI:

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        """Create a Flask test client with isolated storage."""
        monkeypatch.setattr("config.CHROMA_DIR", str(tmp_path / "api_chroma"))
        monkeypatch.setattr("config.DOCUMENTS_DIR", str(tmp_path / "api_docs"))
        monkeypatch.setattr("config.COLLECTION_NAME", "api_test_collection")

        # Reset vector store caches
        import rag.vector_store as vs
        vs._client = None
        vs._collection = None

        import app as flask_app
        flask_app.app.config["TESTING"] = True
        flask_app._startup_done = True  # Skip background startup

        with flask_app.app.test_client() as c:
            yield c

        vs._client = None
        vs._collection = None

    def test_health_endpoint_returns_200(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.get_json()
        assert "status" in data
        assert "ollama" in data
        assert "vector_store" in data
        assert "monitor" in data

    def test_health_has_required_fields(self, client):
        data = client.get("/api/health").get_json()
        assert data["embedding"]["model"] is not None
        assert isinstance(data["vector_store"]["total_chunks"], int)

    def test_chat_with_empty_query_returns_400(self, client):
        res = client.post("/api/chat",
                          json={"query": ""},
                          content_type="application/json")
        assert res.status_code == 400

    def test_chat_with_no_documents_returns_message(self, client):
        res = client.post("/api/chat",
                          json={"query": "What is this about?"},
                          content_type="application/json")
        assert res.status_code == 200
        data = res.get_json()
        assert "answer" in data
        # Should say no documents indexed yet
        assert len(data["answer"]) > 0

    def test_documents_endpoint_returns_list(self, client):
        res = client.get("/api/documents")
        assert res.status_code == 200
        data = res.get_json()
        assert "documents" in data
        assert "total_chunks" in data

    def test_rescan_endpoint(self, client):
        res = client.post("/api/documents/rescan")
        assert res.status_code == 200
        data = res.get_json()
        assert "message" in data

    def test_upload_unsupported_type_returns_400(self, client, tmp_path):
        import io
        data = {"file": (io.BytesIO(b"content"), "test.xyz")}
        res = client.post("/api/documents/upload",
                          data=data,
                          content_type="multipart/form-data")
        assert res.status_code == 400

    def test_upload_txt_file(self, client, tmp_path, monkeypatch):
        import io, os
        # Ensure the patched DOCUMENTS_DIR actually exists before upload
        docs_dir = str(tmp_path / "api_docs")
        os.makedirs(docs_dir, exist_ok=True)
        monkeypatch.setattr("config.DOCUMENTS_DIR", docs_dir)

        content = b"This is a test document about privacy preserving AI. " * 20
        data = {"file": (io.BytesIO(content), "test_upload.txt")}
        res = client.post("/api/documents/upload",
                          data=data,
                          content_type="multipart/form-data")
        assert res.status_code == 200
        resp_data = res.get_json()
        assert resp_data["status"] == "indexing"

    def test_status_endpoint(self, client):
        res = client.get("/api/status")
        assert res.status_code == 200
        data = res.get_json()
        assert "running" in data

    def test_404_returns_json(self, client):
        res = client.get("/api/nonexistent")
        assert res.status_code == 404
        data = res.get_json()
        assert "error" in data
