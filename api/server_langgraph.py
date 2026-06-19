"""LangGraph-only FastAPI server — runs in the llm-langgraph-api container."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

import service
import service_langgraph as lg


class AskGraphRequest(BaseModel):
    question: str = Field(..., min_length=1)
    request_id: str | None = Field(
        default=None,
        description="External request ID; deterministically maps to the Langfuse trace",
    )
    user_id: str | None = Field(default=None, description="End-user ID for Langfuse filtering")
    session_id: str | None = Field(
        default=None,
        description="Conversation/session ID; groups related traces in Langfuse",
    )


class AskGraphResponse(BaseModel):
    question: str
    generation: str
    reflection: str
    answer: str
    request_id: str | None = None
    trace_id: str | None = None


def _trace_context(
    body: AskGraphRequest,
    x_request_id: str | None = Header(None, alias="X-Request-Id"),
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    x_session_id: str | None = Header(None, alias="X-Session-Id"),
) -> service.ApiTraceContext:
    return service.ApiTraceContext(
        request_id=body.request_id or x_request_id,
        user_id=body.user_id or x_user_id,
        session_id=body.session_id or x_session_id,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    print("Loading model for LangGraph API...", flush=True)
    service.load_model()
    lg.get_graph()
    print(f"LangGraph API ready on {service.DEVICE}", flush=True)
    if service.langfuse_enabled():
        print("Langfuse tracing enabled", flush=True)
    yield
    service.flush_langfuse()


app = FastAPI(
    title="LangGraph Qwen API",
    description="Generate → evaluate → reflect loop (service_langgraph.py)",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/graph/mermaid", response_class=PlainTextResponse)
def graph_mermaid():
    return lg.draw_mermaid()


@app.get("/health/langgraph")
def health():
    return {
        "status": "ok",
        "device": service.DEVICE,
        "model_loaded": service.llm is not None,
        "graph_ready": lg.get_graph() is not None,
        "langfuse_enabled": service.langfuse_enabled(),
    }


@app.post("/ask/langgraph", response_model=AskGraphResponse)
def ask_graph(
    body: AskGraphRequest,
    x_request_id: str | None = Header(None, alias="X-Request-Id"),
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    x_session_id: str | None = Header(None, alias="X-Session-Id"),
):
    if service.llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    trace_ctx = _trace_context(body, x_request_id, x_user_id, x_session_id)
    try:
        with service.langfuse_api_trace(trace_ctx, operation="ask_langgraph") as trace_id:
            result = lg.invoke_graph(body.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskGraphResponse(
        question=body.question,
        generation=result["generation"],
        reflection=result["reflection"],
        answer=result["generation"],
        request_id=trace_ctx.request_id,
        trace_id=trace_id,
    )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    print(f"Listening on http://{host}:{port}", flush=True)
    uvicorn.run("server_langgraph:app", host=host, port=port, reload=False)
