"""FastAPI server wrapping langchain.ipynb."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import rag_store
import service


class TraceRequest(BaseModel):
    request_id: str | None = Field(
        default=None,
        description="External request ID; deterministically maps to the Langfuse trace",
    )
    user_id: str | None = Field(default=None, description="End-user ID for Langfuse filtering")
    session_id: str | None = Field(
        default=None,
        description="Conversation/session ID; groups related traces in Langfuse",
    )


class AskRequest(TraceRequest):
    question: str = Field(..., min_length=1)


class AskResponse(BaseModel):
    question: str
    answer: str
    cached: bool = False
    request_id: str | None = None
    trace_id: str | None = None


class AskTrinoRequest(TraceRequest):
    question: str = Field(..., min_length=1)
    catalog: str = "iceberg"
    schema: str = "forecast"
    max_rows: int = Field(default=100, ge=1, le=10_000)


class AskTrinoResponse(BaseModel):
    question: str
    sql: str
    result: dict
    answer: str
    cached: bool = False
    request_id: str | None = None
    trace_id: str | None = None


class TrinoToolRequest(TraceRequest):
    arguments: dict = Field(default_factory=dict)


class TrinoToolResponse(BaseModel):
    result: dict
    request_id: str | None = None
    trace_id: str | None = None


class DocumentInfo(BaseModel):
    id: str
    filename: str
    source_path: str | None = None
    content_type: str | None = None
    byte_size: int | None = None
    chunk_count: int
    created_at: str | None = None
    updated_at: str | None = None


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    replaced: bool


class StoreResponse(BaseModel):
    filename: str
    path: str
    byte_size: int
    replaced: bool


class UploadBatchResponse(BaseModel):
    stored: list[StoreResponse]


class ProcessResponse(BaseModel):
    ingested: list[IngestResponse]
    skipped: list[str]
    failed: list[dict[str, str]]
    total_chunks: int


class ReindexResponse(BaseModel):
    ingested: list[IngestResponse]


def _trace_context(
    body: TraceRequest,
    x_request_id: str | None = Header(None, alias="X-Request-Id"),
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    x_session_id: str | None = Header(None, alias="X-Session-Id"),
) -> service.ApiTraceContext:
    """Body fields take precedence; headers fill in missing values."""
    return service.ApiTraceContext(
        request_id=body.request_id or x_request_id,
        user_id=body.user_id or x_user_id,
        session_id=body.session_id or x_session_id,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.load_model()
    chunk_count = service.ensure_rag_index()
    tool_names = await service.init_trino_mcp()
    print(f"Model loaded on {service.DEVICE}")
    print(f"RAG chunks in pgvector: {chunk_count}")
    if tool_names:
        print(f"Trino MCP tools: {tool_names}")
    elif service.trino_mcp_error:
        print(f"Trino MCP skipped: {service.trino_mcp_error}")
    if service.langfuse_enabled():
        print("Langfuse tracing enabled")
    try:
        service.warm_semantic_cache()
        stats = service.cache_stats()
        if stats.get("embedder_ready"):
            print("Semantic cache embedder ready")
        else:
            print(f"Semantic cache: exact-match only ({stats.get('last_error', 'embedder not loaded')})")
    except Exception as exc:
        print(f"Semantic cache warmup skipped: {exc}")
    yield
    service.flush_langfuse()


app = FastAPI(
    title="LLM Q&A API",
    description="Document RAG (pgvector) and Trino natural-language queries",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    langgraph_port = os.getenv("LLM_LANGGRAPH_PORT", "8002")
    return {
        "status": "ok",
        "device": service.DEVICE,
        "model_loaded": service.llm is not None,
        "rag": service.rag_stats(),
        "documents_dir": str(service.DOCUMENTS_DIR),
        "trino_mcp_ready": service.trino_mcp_ready,
        "trino_tools": service.trino_tool_names,
        "trino_mcp_error": service.trino_mcp_error,
        "langfuse_enabled": service.langfuse_enabled(),
        "semantic_cache": service.cache_stats(),
        "trino_cache_enabled": service.trino_semantic_cache_enabled(),
        "langgraph_health_url": f"http://localhost:{langgraph_port}/health/langgraph",
    }


@app.get("/health/langgraph")
def health_langgraph_moved():
    langgraph_port = os.getenv("LLM_LANGGRAPH_PORT", "8002")
    raise HTTPException(
        status_code=404,
        detail=(
            "LangGraph runs in the llm-langgraph-api container. "
            f"Use http://localhost:{langgraph_port}/health/langgraph"
        ),
    )


@app.post("/ask", response_model=AskResponse)
def ask_rag(
    body: AskRequest,
    x_request_id: str | None = Header(None, alias="X-Request-Id"),
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    x_session_id: str | None = Header(None, alias="X-Session-Id"),
):
    if not service.qa_chain:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if rag_store.chunk_count() == 0:
        raise HTTPException(
            status_code=503,
            detail="No documents indexed. Upload files then POST /documents/ingest",
        )

    trace_ctx = _trace_context(body, x_request_id, x_user_id, x_session_id)
    try:
        with service.langfuse_api_trace(trace_ctx, operation="ask_rag") as trace_id:
            answer, cached = service.answer_from_rag(body.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskResponse(
        question=body.question,
        answer=answer,
        cached=cached,
        request_id=trace_ctx.request_id,
        trace_id=trace_id,
    )


async def _ensure_trino_mcp() -> None:
    if service.trino_mcp_ready:
        return
    await service.init_trino_mcp()
    if not service.trino_mcp_ready:
        raise HTTPException(
            status_code=503,
            detail=service.trino_mcp_error or "Trino MCP is not connected",
        )


@app.post("/trino/ask", response_model=AskTrinoResponse)
async def trino_ask(
    body: AskTrinoRequest,
    x_request_id: str | None = Header(None, alias="X-Request-Id"),
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    x_session_id: str | None = Header(None, alias="X-Session-Id"),
):
    await _ensure_trino_mcp()
    trace_ctx = _trace_context(body, x_request_id, x_user_id, x_session_id)
    try:
        with service.langfuse_api_trace(trace_ctx, operation="trino_ask") as trace_id:
            payload, cached = await service.ask_trino(
                body.question,
                catalog=body.catalog,
                schema=body.schema,
                max_rows=body.max_rows,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskTrinoResponse(
        **payload,
        cached=cached,
        request_id=trace_ctx.request_id,
        trace_id=trace_id,
    )


@app.post("/trino/tools/{tool_name}", response_model=TrinoToolResponse)
async def trino_tool(
    tool_name: str,
    body: TrinoToolRequest,
    x_request_id: str | None = Header(None, alias="X-Request-Id"),
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    x_session_id: str | None = Header(None, alias="X-Session-Id"),
):
    await _ensure_trino_mcp()
    trace_ctx = _trace_context(body, x_request_id, x_user_id, x_session_id)
    try:
        with service.langfuse_api_trace(trace_ctx, operation=f"mcp_tool_{tool_name}") as trace_id:
            result = await service.call_trino_tool(tool_name, body.arguments)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return TrinoToolResponse(
        result=result,
        request_id=trace_ctx.request_id,
        trace_id=trace_id,
    )


@app.post("/trino/reconnect")
async def trino_reconnect():
    tool_names = await service.init_trino_mcp()
    if not tool_names:
        raise HTTPException(
            status_code=503,
            detail=service.trino_mcp_error or "Failed to connect to Trino MCP",
        )
    return {"tools": tool_names}


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents():
    try:
        return rag_store.list_documents()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/documents/staged")
def list_staged_documents():
    """Files on disk under DOCUMENTS_DIR and whether each is indexed in pgvector."""
    return {"documents_dir": str(rag_store.DOCUMENTS_DIR), "files": rag_store.list_disk_files()}


@app.post("/documents/upload", response_model=StoreResponse)
async def upload_document(file: UploadFile = File(...)):
    """Save one file to DOCUMENTS_DIR. Call POST /documents/ingest to index into pgvector."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename is required")
    try:
        content = await file.read()
        result = rag_store.save_upload(file.filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return StoreResponse(
        filename=result.filename,
        path=result.path,
        byte_size=result.byte_size,
        replaced=result.replaced,
    )


@app.post("/documents/uploads", response_model=UploadBatchResponse)
async def upload_documents(files: list[UploadFile] = File(...)):
    """Save multiple files to DOCUMENTS_DIR. Call POST /documents/ingest to index all."""
    if not files:
        raise HTTPException(status_code=400, detail="at least one file is required")

    stored: list[StoreResponse] = []
    errors: list[dict[str, str]] = []
    for file in files:
        if not file.filename:
            errors.append({"filename": "", "error": "missing filename"})
            continue
        try:
            content = await file.read()
            result = rag_store.save_upload(file.filename, content)
            stored.append(
                StoreResponse(
                    filename=result.filename,
                    path=result.path,
                    byte_size=result.byte_size,
                    replaced=result.replaced,
                )
            )
        except ValueError as exc:
            errors.append({"filename": file.filename, "error": str(exc)})
        except Exception as exc:
            errors.append({"filename": file.filename, "error": str(exc)})

    if errors and not stored:
        raise HTTPException(status_code=400, detail={"errors": errors})
    response = UploadBatchResponse(stored=stored)
    if errors:
        return JSONResponse(
            status_code=207,
            content={**response.model_dump(), "errors": errors},
        )
    return response


@app.post("/documents/ingest", response_model=ProcessResponse)
def ingest_documents(force: bool = False):
    """
    Chunk, embed, and index every file in DOCUMENTS_DIR into pgvector.

    Upload first via POST /documents/upload or /documents/uploads.
    Set force=true to re-process files already indexed with the same content.
    """
    try:
        result = rag_store.process_documents(force=force)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    ingested = [
        IngestResponse(
            document_id=item.document_id,
            filename=item.filename,
            chunk_count=item.chunk_count,
            replaced=item.replaced,
        )
        for item in result.ingested
    ]
    total_chunks = sum(item.chunk_count for item in result.ingested)
    if result.failed and not ingested:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "All documents failed to ingest",
                "failed": result.failed,
                "skipped": result.skipped,
            },
        )
    return ProcessResponse(
        ingested=ingested,
        skipped=result.skipped,
        failed=result.failed,
        total_chunks=total_chunks,
    )


@app.post("/documents/reindex", response_model=ReindexResponse)
def reindex_documents():
    """Alias for POST /documents/ingest?force=true."""
    try:
        result = rag_store.process_documents(force=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ReindexResponse(
        ingested=[
            IngestResponse(
                document_id=item.document_id,
                filename=item.filename,
                chunk_count=item.chunk_count,
                replaced=item.replaced,
            )
            for item in result.ingested
        ]
    )


@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    try:
        deleted = rag_store.delete_document(document_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": True, "document_id": document_id}


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8001"))
    print(f"Starting LLM API on http://{host}:{port}", flush=True)
    print(f"  LangGraph: http://localhost:{port}/health/langgraph", flush=True)
    uvicorn.run("server:app", host=host, port=port, reload=False)
