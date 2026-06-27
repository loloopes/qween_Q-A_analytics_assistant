"""Persistent RAG: ingest unstructured docs into pgvector, retrieve by cosine similarity."""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

import embeddings

logger = logging.getLogger(__name__)

API_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = Path(
    os.getenv("DOCUMENTS_DIR", str(API_DIR / "documents"))
).resolve()

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))

_last_error: str | None = None
_schema_ready = False


@dataclass
class IngestResult:
    document_id: str
    filename: str
    chunk_count: int
    replaced: bool


@dataclass
class StoreResult:
    filename: str
    path: str
    byte_size: int
    replaced: bool


@dataclass
class ProcessResult:
    ingested: list[IngestResult]
    skipped: list[str]
    failed: list[dict[str, str]]


def _pg_settings() -> dict[str, str | int]:
    in_docker = os.getenv("LLM_IN_DOCKER", "").strip().lower() in {"1", "true", "yes", "on"}
    default_host = "pgvector" if in_docker else "localhost"
    default_port = 5432 if in_docker else 5433
    return {
        "host": os.getenv("PGVECTOR_HOST", default_host),
        "port": int(os.getenv("PGVECTOR_PORT", str(default_port))),
        "user": os.getenv("PGVECTOR_USER", "vector"),
        "password": os.getenv("PGVECTOR_PASSWORD", "vector"),
        "dbname": os.getenv("PGVECTOR_DB", "vectors"),
    }


def _connect():
    import psycopg

    return psycopg.connect(**_pg_settings())


def _vector_literal(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.8f}" for x in vec.tolist()) + "]"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix in {".txt", ".md", ".markdown"}:
        return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {suffix}")


def _split_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return [chunk for chunk in splitter.split_text(text) if chunk.strip()]


def ensure_schema() -> None:
    global _schema_ready, _last_error
    if _schema_ready:
        return

    dim = embeddings.embedding_dimension()
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_documents (
                id UUID PRIMARY KEY,
                filename TEXT NOT NULL,
                source_path TEXT,
                content_type TEXT,
                byte_size BIGINT,
                content_hash TEXT,
                chunk_count INT NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS rag_document_chunks (
                id UUID PRIMARY KEY,
                document_id UUID NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
                chunk_index INT NOT NULL,
                content TEXT NOT NULL,
                embedding vector({dim}) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS rag_documents_filename_idx
            ON rag_documents (filename)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS rag_document_chunks_document_id_idx
            ON rag_document_chunks (document_id)
            """
        )
        conn.commit()

    _schema_ready = True
    _last_error = None


def _delete_document_by_filename(filename: str, conn) -> str | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM rag_documents WHERE filename = %s",
            (filename,),
        )
        row = cur.fetchone()
        if not row:
            return None
        document_id = str(row[0])
        cur.execute("DELETE FROM rag_documents WHERE id = %s", (document_id,))
        return document_id


def ingest_file(path: Path, *, replace: bool = True) -> IngestResult:
    """Chunk, embed, and store a file from disk."""
    global _last_error

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    ensure_schema()
    text = _extract_text(path)
    chunks = _split_text(text)
    if not chunks:
        raise ValueError(f"No text extracted from {path.name}")

    content_hash = _file_hash(path)
    document_id = str(uuid.uuid4())
    replaced = False

    with _connect() as conn, conn.cursor() as cur:
        if replace:
            replaced = _delete_document_by_filename(path.name, conn) is not None

        cur.execute(
            """
            INSERT INTO rag_documents (
                id, filename, source_path, content_type, byte_size, content_hash, chunk_count
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                document_id,
                path.name,
                str(path),
                path.suffix.lower().lstrip("."),
                path.stat().st_size,
                content_hash,
                len(chunks),
            ),
        )

        for index, chunk in enumerate(chunks):
            vector = embeddings.embed_text(chunk)
            cur.execute(
                """
                INSERT INTO rag_document_chunks (
                    id, document_id, chunk_index, content, embedding
                ) VALUES (%s, %s, %s, %s, %s::vector)
                """,
                (
                    str(uuid.uuid4()),
                    document_id,
                    index,
                    chunk,
                    _vector_literal(vector),
                ),
            )
        conn.commit()

    _last_error = None
    return IngestResult(
        document_id=document_id,
        filename=path.name,
        chunk_count=len(chunks),
        replaced=replaced,
    )


def save_upload(filename: str, content: bytes) -> StoreResult:
    """Save an uploaded file under DOCUMENTS_DIR (does not index into pgvector)."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name
    if not safe_name:
        raise ValueError("filename is required")
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix}")

    path = DOCUMENTS_DIR / safe_name
    replaced = path.is_file()
    path.write_bytes(content)
    return StoreResult(
        filename=safe_name,
        path=str(path),
        byte_size=len(content),
        replaced=replaced,
    )


def list_disk_files() -> list[dict[str, Any]]:
    """Files currently stored under DOCUMENTS_DIR (uploaded, not necessarily indexed)."""
    if not DOCUMENTS_DIR.is_dir():
        return []

    indexed_hashes: dict[str, str] = {}
    try:
        ensure_schema()
        with _connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT filename, content_hash FROM rag_documents")
            for filename, content_hash in cur.fetchall():
                indexed_hashes[str(filename)] = str(content_hash)
    except Exception:
        pass

    files: list[dict[str, Any]] = []
    for path in sorted(DOCUMENTS_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        digest = _file_hash(path)
        files.append(
            {
                "filename": path.name,
                "path": str(path),
                "byte_size": path.stat().st_size,
                "content_hash": digest,
                "indexed": indexed_hashes.get(path.name) == digest,
                "pending_ingest": indexed_hashes.get(path.name) != digest,
            }
        )
    return files


def ingest_documents_dir(*, replace: bool = True) -> list[IngestResult]:
    """Ingest every supported file in DOCUMENTS_DIR."""
    if not DOCUMENTS_DIR.is_dir():
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        return []

    results: list[IngestResult] = []
    for path in sorted(DOCUMENTS_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            results.append(ingest_file(path, replace=replace))
    return results


def process_documents(*, force: bool = False) -> ProcessResult:
    """
    Chunk, embed, and store every file in DOCUMENTS_DIR into pgvector.

    Skips files already indexed with the same content hash unless force=True.
    """
    if not DOCUMENTS_DIR.is_dir():
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        return ProcessResult(ingested=[], skipped=[], failed=[])

    ensure_schema()
    embeddings.warm_embedder()

    known_hashes: dict[str, str] = {}
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT filename, content_hash FROM rag_documents")
        for filename, content_hash in cur.fetchall():
            known_hashes[str(filename)] = str(content_hash)

    ingested: list[IngestResult] = []
    skipped: list[str] = []
    failed: list[dict[str, str]] = []

    for path in sorted(DOCUMENTS_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        digest = _file_hash(path)
        if not force and known_hashes.get(path.name) == digest:
            skipped.append(path.name)
            continue
        try:
            result = ingest_file(path, replace=True)
            ingested.append(result)
            known_hashes[path.name] = digest
        except Exception as exc:
            logger.exception("Failed to ingest %s", path.name)
            failed.append({"filename": path.name, "error": str(exc)})

    return ProcessResult(ingested=ingested, skipped=skipped, failed=failed)


def auto_ingest_documents_dir() -> list[IngestResult]:
    """Ingest files in DOCUMENTS_DIR that are new or changed (by content hash)."""
    return process_documents(force=False).ingested


def list_documents() -> list[dict[str, Any]]:
    ensure_schema()
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, filename, source_path, content_type, byte_size, chunk_count, created_at, updated_at
            FROM rag_documents
            ORDER BY created_at DESC
            """
        )
        rows = cur.fetchall()
    return [
        {
            "id": str(row[0]),
            "filename": row[1],
            "source_path": row[2],
            "content_type": row[3],
            "byte_size": row[4],
            "chunk_count": row[5],
            "created_at": row[6].isoformat() if row[6] else None,
            "updated_at": row[7].isoformat() if row[7] else None,
        }
        for row in rows
    ]


def delete_document(document_id: str) -> bool:
    ensure_schema()
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM rag_documents WHERE id = %s", (document_id,))
        deleted = cur.rowcount > 0
        conn.commit()
    return deleted


def chunk_count() -> int:
    ensure_schema()
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM rag_document_chunks")
        row = cur.fetchone()
    return int(row[0]) if row else 0


def retrieve_context(question: str, top_k: int | None = None) -> str:
    """Return top-k chunks by cosine similarity (pgvector <=> operator)."""
    global _last_error

    top_k = top_k or DEFAULT_TOP_K
    ensure_schema()
    if chunk_count() == 0:
        return ""

    query_vec = embeddings.embed_text(question)
    vector_sql = _vector_literal(query_vec)

    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.content, d.filename, 1 - (c.embedding <=> %s::vector) AS score
            FROM rag_document_chunks c
            JOIN rag_documents d ON d.id = c.document_id
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            (vector_sql, vector_sql, top_k),
        )
        rows = cur.fetchall()

    if not rows:
        return ""

    _last_error = None
    return "\n\n---\n\n".join(row[0] for row in rows)


def stats() -> dict[str, Any]:
    global _last_error
    try:
        ensure_schema()
        docs = list_documents()
        chunks = chunk_count()
        return {
            "backend": "pgvector",
            "ready": True,
            "documents_dir": str(DOCUMENTS_DIR),
            "document_count": len(docs),
            "chunk_count": chunks,
            "embedding_model": embeddings.embedding_model_name(),
            "embedding_dimension": embeddings.embedding_dimension(),
            "top_k": DEFAULT_TOP_K,
            "last_error": _last_error,
        }
    except Exception as exc:
        _last_error = str(exc)
        return {
            "backend": "pgvector",
            "ready": False,
            "documents_dir": str(DOCUMENTS_DIR),
            "document_count": 0,
            "chunk_count": 0,
            "embedding_model": embeddings.embedding_model_name(),
            "last_error": _last_error,
        }


def warm() -> None:
    """Ensure schema and embedder; optional auto-ingest when RAG_AUTO_INGEST=true."""
    ensure_schema()
    embeddings.warm_embedder()
    if os.getenv("RAG_AUTO_INGEST", "false").strip().lower() in {"1", "true", "yes", "on"}:
        result = process_documents(force=False)
        if result.ingested:
            logger.info("Auto-ingested %s document(s) into pgvector", len(result.ingested))
