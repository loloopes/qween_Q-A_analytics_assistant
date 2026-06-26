"""LangSmith tracing for the LangGraph API (replaces Langfuse on that path)."""

from __future__ import annotations

import contextvars
import os
from contextlib import contextmanager
from typing import Any, Iterator

from service import ApiTraceContext

try:
    from langsmith import traceable as _langsmith_traceable
except ImportError:  # pragma: no cover
    def _langsmith_traceable(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator


_token_totals: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar(
    "_token_totals",
    default=None,
)


def langsmith_project() -> str:
    return (
        os.getenv("LANGSMITH_PROJECT")
        or os.getenv("LANGCHAIN_PROJECT")
        or "llm-langgraph"
    )


def langsmith_model_name() -> str:
    return os.getenv("LANGSMITH_MODEL_NAME") or os.getenv("BASE_MODEL", "qween")


def langsmith_input_price_per_unit() -> float:
    return float(os.getenv("LANGSMITH_INPUT_PRICE_PER_UNIT", "0"))


def langsmith_output_price_per_unit() -> float:
    return float(os.getenv("LANGSMITH_OUTPUT_PRICE_PER_UNIT", "0"))


def langsmith_enabled() -> bool:
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        return False
    tracing = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2")
    if tracing is None:
        return True
    return tracing.strip().lower() in {"1", "true", "yes", "on"}


def configure_langsmith() -> None:
    """Enable LangChain/LangGraph tracing env vars for Studio and local runs."""
    if not langsmith_enabled():
        return
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", langsmith_project())
    os.environ.setdefault("LANGSMITH_PROJECT", langsmith_project())


def flush_langsmith() -> None:
    if not langsmith_enabled():
        return
    try:
        from langsmith import Client

        Client().flush()
    except Exception:
        pass


def count_tokens(text: str) -> int:
    import service

    if not text or service.tokenizer is None:
        return 0
    return service._count_tokens(text)


def usage_metadata(input_tokens: int, output_tokens: int) -> dict[str, int]:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def token_cost(input_tokens: int, output_tokens: int) -> dict[str, float] | None:
    input_price = langsmith_input_price_per_unit()
    output_price = langsmith_output_price_per_unit()
    if input_price == 0 and output_price == 0:
        return None
    input_cost = input_tokens * input_price
    output_cost = output_tokens * output_price
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


def _accumulate_tokens(input_tokens: int, output_tokens: int) -> None:
    totals = _token_totals.get()
    if totals is None:
        return
    totals["input_tokens"] += input_tokens
    totals["output_tokens"] += output_tokens
    totals["total_tokens"] += input_tokens + output_tokens


@contextmanager
def token_tracking_context() -> Iterator[dict[str, int]]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    token = _token_totals.set(totals)
    try:
        yield totals
    finally:
        _token_totals.reset(token)


def traceable(*args, **kwargs):
    """LangSmith @traceable when enabled; no-op decorator otherwise."""
    if not langsmith_enabled():
        def decorator(func):
            return func

        if args and callable(args[0]) and not kwargs:
            return args[0]
        return decorator
    return _langsmith_traceable(*args, **kwargs)


def langsmith_callbacks() -> list[Any]:
    if not langsmith_enabled():
        return []
    try:
        from langchain_core.tracers import LangChainTracer

        return [LangChainTracer(project_name=langsmith_project())]
    except Exception:
        return []


def langsmith_run_config(**metadata: Any) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if metadata:
        config["metadata"] = metadata
    callbacks = langsmith_callbacks()
    if callbacks:
        config["callbacks"] = callbacks
    return config


def run_traced_llm(name: str, prompt: str, invoke) -> str:
    if not langsmith_enabled():
        return invoke().strip()

    from langsmith.run_helpers import trace

    model = langsmith_model_name()
    with trace(
        name=name,
        run_type="llm",
        inputs={"prompt": prompt},
        metadata={"ls_model_name": model},
    ) as run:
        output = invoke().strip()
        input_tokens = count_tokens(prompt)
        output_tokens = count_tokens(output)
        usage = usage_metadata(input_tokens, output_tokens)
        _accumulate_tokens(input_tokens, output_tokens)

        if run is not None:
            extra_metadata = {
                "ls_model_name": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": usage["total_tokens"],
            }
            costs = token_cost(input_tokens, output_tokens)
            if costs:
                extra_metadata.update(costs)
            run.set(usage_metadata=usage, metadata=extra_metadata)
            run.end(outputs={"text": output})

        return output


def update_current_run_metadata(**metadata: Any) -> None:
    if not langsmith_enabled():
        return
    try:
        from langsmith import get_current_run_tree

        run = get_current_run_tree()
        if run is not None:
            run.metadata.update(metadata)
    except Exception:
        pass


def update_current_run_usage(
    *,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int | None = None,
) -> None:
    if not langsmith_enabled():
        return
    try:
        from langsmith import get_current_run_tree

        run = get_current_run_tree()
        if run is None:
            return

        total = total_tokens if total_tokens is not None else input_tokens + output_tokens
        usage = usage_metadata(input_tokens, output_tokens)
        run.set(usage_metadata=usage)
        run.metadata["input_tokens"] = input_tokens
        run.metadata["output_tokens"] = output_tokens
        run.metadata["total_tokens"] = total
        costs = token_cost(input_tokens, output_tokens)
        if costs:
            run.metadata.update(costs)
    except Exception:
        pass


@contextmanager
def langsmith_api_trace(ctx: ApiTraceContext, *, operation: str) -> Iterator[str | None]:
    if not langsmith_enabled():
        yield None
        return

    from langsmith.run_helpers import trace

    metadata: dict[str, Any] = {}
    if ctx.request_id:
        metadata["request_id"] = ctx.request_id
    if ctx.user_id:
        metadata["user_id"] = ctx.user_id
    if ctx.session_id:
        metadata["session_id"] = ctx.session_id

    with trace(
        name=operation,
        run_type="chain",
        metadata=metadata or None,
    ) as run:
        yield str(run.id) if run is not None else None
