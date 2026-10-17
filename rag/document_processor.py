"""
rag/document_processor.py
─────────────────────────
Responsible for:
  1. Extracting raw text from PDF, DOCX, TXT, .java, .cs, .py, .md files.
  2. For PDFs: tracking which page each piece of text comes from.
  3. Splitting text into overlapping chunks for embedding.

Key design decisions (for student understanding):
  - We use a sliding-window character-based chunker (not token-based)
    because it is simple, fast, and does not require a tokenizer.
  - Each chunk carries metadata: filename, page number (PDF only),
    chunk_index, and the source file type.
  - Page numbers are 1-indexed to match what a human sees in a PDF viewer.
"""

import os
import logging
from typing import List, Dict, Any

import config

logger = logging.getLogger(__name__)


# ─── Text Extraction ──────────────────────────────────────────────────────────

def extract_text_pdf(filepath: str) -> List[Dict[str, Any]]:
    """
    Extract text from a PDF file, page by page.
    Returns a list of dicts: {"text": str, "page": int}
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        pages = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({"text": text, "page": page_num})
        return pages
    except Exception as e:
        logger.error(f"PDF extraction failed for {filepath}: {e}")
        return []


def extract_text_docx(filepath: str) -> List[Dict[str, Any]]:
    """
    Extract text from a DOCX file.
    DOCX files do not have pages in the same sense as PDFs,
    so page is set to None.
    """
    try:
        from docx import Document
        doc = Document(filepath)
        full_text = "\n".join(
            para.text for para in doc.paragraphs if para.text.strip()
        )
        return [{"text": full_text, "page": None}] if full_text else []
    except Exception as e:
        logger.error(f"DOCX extraction failed for {filepath}: {e}")
        return []


def extract_text_plain(filepath: str) -> List[Dict[str, Any]]:
    """
    Extract text from plain-text files: .txt, .java, .cs, .py, .md
    Tries UTF-8 first, then falls back to latin-1.
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(filepath, "r", encoding=encoding) as f:
                text = f.read().strip()
            return [{"text": text, "page": None}] if text else []
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Plain text extraction failed for {filepath}: {e}")
            return []
    return []


def extract_text(filepath: str) -> List[Dict[str, Any]]:
    """
    Dispatch to the correct extractor based on file extension.
    Returns list of {"text": str, "page": int|None}
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return extract_text_pdf(filepath)
    elif ext == ".docx":
        return extract_text_docx(filepath)
    elif ext in {".txt", ".java", ".cs", ".py", ".md"}:
        return extract_text_plain(filepath)
    else:
        logger.warning(f"Unsupported file type: {ext}")
        return []


# ─── Text Chunking ────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    Split a long string into overlapping chunks.

    Example with chunk_size=10, overlap=3 and text="ABCDEFGHIJKLMNOP":
      chunk 0: "ABCDEFGHIJ"
      chunk 1: "HIJKLMNOP"   ← starts 7 chars later (10-3=7)

    Chunks smaller than 20 chars are discarded (too small to be meaningful).
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    step = chunk_size - overlap
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if len(chunk) >= 20:
            chunks.append(chunk)
        start += step
    return chunks


# ─── Main entry: process a file into chunks with metadata ─────────────────────

def process_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Full pipeline: extract text → chunk → annotate with metadata.

    Returns a list of chunk records, each being a dict:
    {
        "text":        str,   # the chunk text, sent to embedder
        "filename":    str,   # basename of source file
        "filepath":    str,   # absolute path
        "page":        int|None,  # PDF page (1-indexed), or None
        "chunk_index": int,   # position of this chunk within the file
        "file_type":   str,   # e.g. "pdf", "docx", "txt"
    }
    """
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filepath)[1].lower().lstrip(".")
    pages = extract_text(filepath)

    if not pages:
        logger.warning(f"No text extracted from {filename}")
        return []

    all_chunks = []
    chunk_index = 0

    for page_data in pages:
        raw_text = page_data["text"]
        page_num = page_data["page"]

        for chunk_text_item in chunk_text(raw_text):
            all_chunks.append({
                "text":        chunk_text_item,
                "filename":    filename,
                "filepath":    filepath,
                "page":        page_num,
                "chunk_index": chunk_index,
                "file_type":   ext,
            })
            chunk_index += 1

    logger.info(f"Processed '{filename}': {len(pages)} page(s), {len(all_chunks)} chunks")
    return all_chunks
