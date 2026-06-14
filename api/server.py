"""FastAPI server wrapping langchain.ipynb."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

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


class AskPdfRequest(TraceRequest):
    question: str = Field(..., min_length=1)


class AskPdfResponse(BaseModel):
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
    chunk_count = service.load_pdf_index()
    tool_names = await service.init_trino_mcp()
    print(f"Model loaded on {service.DEVICE}")
    print(f"PDF indexed: {chunk_count} chunk(s)")
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
    description="PDF RAG and Trino natural-language queries (langchain.ipynb)",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": service.DEVICE,
        "model_loaded": service.llm is not None,
        "pdf_chunks": len(service.chunks),
        "pdf_path": service.PDF_PATH,
        "trino_mcp_ready": service.trino_mcp_ready,
        "trino_tools": service.trino_tool_names,
        "trino_mcp_error": service.trino_mcp_error,
        "langfuse_enabled": service.langfuse_enabled(),
        "semantic_cache": service.cache_stats(),
        "trino_cache_enabled": service.trino_semantic_cache_enabled(),
    }


@app.post("/ask", response_model=AskPdfResponse)
def ask_pdf(
    body: AskPdfRequest,
    x_request_id: str | None = Header(None, alias="X-Request-Id"),
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    x_session_id: str | None = Header(None, alias="X-Session-Id"),
):
    if not service.qa_chain:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not service.chunks:
        raise HTTPException(status_code=503, detail="PDF index not loaded")

    trace_ctx = _trace_context(body, x_request_id, x_user_id, x_session_id)
    try:
        with service.langfuse_api_trace(trace_ctx, operation="ask_pdf") as trace_id:
            answer, cached = service.answer_from_pdf(body.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskPdfResponse(
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


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("server:app", host=host, port=port, reload=False)
