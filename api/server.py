"""FastAPI server wrapping langchain.ipynb."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import service


class AskPdfRequest(BaseModel):
    question: str = Field(..., min_length=1)


class AskPdfResponse(BaseModel):
    question: str
    answer: str


class AskTrinoRequest(BaseModel):
    question: str = Field(..., min_length=1)
    catalog: str = "iceberg"
    schema: str = "forecast"
    max_rows: int = Field(default=100, ge=1, le=10_000)


class TrinoToolRequest(BaseModel):
    arguments: dict = Field(default_factory=dict)


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
    }


@app.post("/ask", response_model=AskPdfResponse)
def ask_pdf(body: AskPdfRequest):
    if not service.qa_chain:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not service.chunks:
        raise HTTPException(status_code=503, detail="PDF index not loaded")
    try:
        answer = service.answer_from_pdf(body.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AskPdfResponse(question=body.question, answer=answer)


async def _ensure_trino_mcp() -> None:
    if service.trino_mcp_ready:
        return
    await service.init_trino_mcp()
    if not service.trino_mcp_ready:
        raise HTTPException(
            status_code=503,
            detail=service.trino_mcp_error or "Trino MCP is not connected",
        )


@app.post("/trino/ask")
async def trino_ask(body: AskTrinoRequest):
    await _ensure_trino_mcp()
    try:
        return await service.ask_trino(
            body.question,
            catalog=body.catalog,
            schema=body.schema,
            max_rows=body.max_rows,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/trino/tools/{tool_name}")
async def trino_tool(tool_name: str, body: TrinoToolRequest):
    await _ensure_trino_mcp()
    try:
        return await service.call_trino_tool(tool_name, body.arguments)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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
