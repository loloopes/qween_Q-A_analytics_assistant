"""Core logic from langchain.ipynb — model, RAG, and Trino MCP."""

from __future__ import annotations

import json
import os
import re
import sys
import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFacePipeline
from langchain_mcp_adapters.client import MultiServerMCPClient
from peft import PeftModel
import rag_store
import guardrails
from semantic_cache import (
    cache_lookup,
    cache_stats,
    cache_store,
    semantic_cache_enabled,
    warm_semantic_cache,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

try:
    from langfuse import observe
except ImportError:  # pragma: no cover - optional until langfuse is installed
    def observe(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator

API_DIR = Path(__file__).resolve().parent
# Do not override variables already set by Docker Compose / the shell.
load_dotenv(API_DIR.parent / ".env", override=False)
load_dotenv(API_DIR / ".env", override=False)


def langfuse_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def flush_langfuse() -> None:
    if not langfuse_enabled():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        pass


def _mark_cache_hit() -> None:
    if not langfuse_enabled():
        return
    try:
        from langfuse import get_client

        client = get_client()
        client.update_current_span(metadata={"cache_hit": True})
        client.score_current_span(name="cache_hit", value=1, data_type="BOOLEAN")
    except Exception:
        pass


def _count_tokens(text: str) -> int:
    if not text or tokenizer is None:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def trino_semantic_cache_enabled() -> bool:
    """Trino full-response cache. When SEMANTIC_CACHE_TRINO_ENABLED is unset, follows SEMANTIC_CACHE_ENABLED."""
    explicit = os.getenv("SEMANTIC_CACHE_TRINO_ENABLED")
    if explicit is not None and explicit.strip():
        return explicit.strip().lower() in {"1", "true", "yes", "on"}
    return semantic_cache_enabled()


def _run_llm_generation(name: str, prompt: str, invoke) -> str:
    """Run LLM inference and record model, tokens, and cost on the generation observation."""
    raw_invoke = invoke
    invoke = lambda: guardrails.sanitize(raw_invoke().strip())

    if not langfuse_enabled():
        return invoke()

    from langfuse import get_client

    client = get_client()
    with client.start_as_current_observation(
        as_type="generation",
        name=name,
        model=LANGFUSE_MODEL_NAME,
        input=guardrails.sanitize(prompt),
    ) as generation:
        output = invoke()
        input_tokens = _count_tokens(prompt)
        output_tokens = _count_tokens(output)
        input_cost = input_tokens * LANGFUSE_INPUT_PRICE_PER_UNIT
        output_cost = output_tokens * LANGFUSE_OUTPUT_PRICE_PER_UNIT
        generation.update(
            output=output,
            model=LANGFUSE_MODEL_NAME,
            usage_details={
                "input": input_tokens,
                "output": output_tokens,
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total": input_tokens + output_tokens,
            },
            cost_details={
                "input": input_cost,
                "output": output_cost,
                "total": input_cost + output_cost,
            },
        )
        return output


async def _await_llm_generation(name: str, prompt: str, invoke) -> str:
    """Run sync LLM inference off the event loop so MCP SSE stays alive."""
    return await asyncio.to_thread(_run_llm_generation, name, prompt, invoke)


@dataclass
class ApiTraceContext:
    request_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None


@contextmanager
def langfuse_api_trace(ctx: ApiTraceContext, *, operation: str) -> Iterator[str | None]:
    """Attach client IDs to the Langfuse trace for this API request."""
    if not langfuse_enabled():
        yield None
        return

    from langfuse import get_client, propagate_attributes

    client = get_client()
    trace_context = None
    if ctx.request_id:
        trace_context = {
            "trace_id": client.create_trace_id(seed=ctx.request_id),
            "parent_span_id": "0000000000000000",
        }

    prop_kwargs: dict = {}
    if ctx.user_id:
        prop_kwargs["user_id"] = ctx.user_id
    if ctx.session_id:
        prop_kwargs["session_id"] = ctx.session_id
    if ctx.request_id:
        prop_kwargs.setdefault("metadata", {})["request_id"] = ctx.request_id

    with client.start_as_current_observation(
        trace_context=trace_context,
        as_type="span",
        name=operation,
    ):
        if prop_kwargs:
            with propagate_attributes(**prop_kwargs):
                yield client.get_current_trace_id()
        else:
            yield client.get_current_trace_id()


BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_REPO = os.getenv("ADAPTER_REPO") or os.getenv("HF_REPO_ID") or "Glccampos/llm_qween"
HF_TOKEN = os.getenv("HF_TOKEN")
# Must match a Langfuse model definition regex (e.g. (?i)^(qween)$ → send "qween").
LANGFUSE_MODEL_NAME = os.getenv("LANGFUSE_MODEL_NAME", "qween")
LANGFUSE_INPUT_PRICE_PER_UNIT = float(os.getenv("LANGFUSE_INPUT_PRICE_PER_UNIT", "0.01"))
LANGFUSE_OUTPUT_PRICE_PER_UNIT = float(os.getenv("LANGFUSE_OUTPUT_PRICE_PER_UNIT", "0.01"))
DOCUMENTS_DIR = rag_store.DOCUMENTS_DIR

USE_SAMPLING = True
TEMPERATURE = 0.1
TOP_P = 0.9
TOP_K = 50

PREDICTION_EVENT_SAMPLES = """
prediction_events (combined): metadata event_id, event_ts, model_name, model_stage
plus response columns request_id, probability, threshold_decision, status
plus request columns:
id_cliente, id_contrato, tipo_contrato, status_contrato, tipo_pagamento, finalidade_emprestimo, tipo_cliente, tipo_portfolio, tipo_produto, categoria_bem, setor_vendedor, canal_venda, faixa_rendimento, combinacao_produto, area_venda, dia_semana_solicitacao, data_nascimento, data_decisao, data_liberacao, data_primeiro_vencimento, data_ultimo_vencimento_original, data_ultimo_vencimento, data_encerramento, valor_solicitado, valor_credito, valor_bem, valor_parcela, valor_entrada, percentual_entrada, qtd_parcelas_planejadas, taxa_juros_padrao, taxa_juros_promocional, hora_solicitacao, flag_ultima_solicitacao_contrato, flag_ultima_solicitacao_dia, acompanhantes_cliente, flag_seguro_contratado, motivo_recusa, renda_anual, qtd_membros_familia, possui_carro, possui_imovel

prediction_requests (request only): same metadata + request columns; join to prediction_events on event_id
"""

tokenizer = None
llm = None
qa_chain = None

mcp_client: MultiServerMCPClient | None = None
trino_tools: list = []
trino_tool_names: list[str] = []
trino_mcp_ready = False
trino_mcp_error: str | None = None

def _resolve_device() -> str:
    """Pick inference device from LLM_DEVICE or auto-detect CUDA."""
    explicit = (os.getenv("LLM_DEVICE") or "").strip().lower()
    if explicit == "cpu":
        return "cpu"
    if explicit in {"cuda", "gpu"} or explicit.startswith("cuda:"):
        if not torch.cuda.is_available():
            print(
                "WARNING: LLM_DEVICE requests CUDA but torch.cuda.is_available() is False; using CPU.",
                flush=True,
            )
            return "cpu"
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = _resolve_device()
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Heuristic keywords for routing /ask → Trino (credit-risk / prediction_events domain).
_TRINO_ROUTE_PATTERN = re.compile(
    r"(?i)\b("
    r"cliente?s?|clientes|customers?|"
    r"empr[eé]stimo|empr[eé]stimos|loan|loans|credito|cr[eé]dito|"
    r"negad[oa]s?|denied|aprovad[oa]s?|approved|"
    r"contrato|contratos|contract|"
    r"predi[cç][aã]o|predi[cç][oõ]es|prediction|probabilidade|probability|"
    r"threshold|forecast|iceberg|trino|"
    r"quantos?|quanto|how\s+many|count|total|"
    r"prediction_events|id_cliente|threshold_decision|"
    r"solicita[cç][aã]o|aplica[cç][aã]o"
    r")\b"
)


def should_route_to_trino(question: str) -> bool:
    """True when a natural-language question should use Trino instead of document RAG."""
    return bool(_TRINO_ROUTE_PATTERN.search(question.strip()))


def _find_trino_mcp_dir() -> Path:
    cwd = Path.cwd().resolve()
    candidates: list[Path] = []
    if env_dir := os.getenv("TRINO_MCP_DIR"):
        candidates.append(Path(env_dir))
    candidates.extend(
        [
            API_DIR.parent.parent / "mcp",
            API_DIR / "mcp",
            cwd / "mcp",
        ]
    )
    if sys.platform == "linux":
        candidates.append(Path("/mnt/c/Users/guslc/project/mcp"))
    elif sys.platform == "win32":
        candidates.append(Path(r"C:\Users\guslc\project\mcp"))

    for path in candidates:
        resolved = path.resolve()
        if (resolved / "trino_mcp.py").exists():
            return resolved
    tried = ", ".join(str(p.resolve()) for p in candidates)
    raise FileNotFoundError(
        f"Could not find mcp/trino_mcp.py. Tried: {tried}. "
        "Set TRINO_MCP_DIR in .env to your mcp folder."
    )


def _path_usable(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _check_sse_reachable(url: str) -> None:
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8765
    try:
        with socket.create_connection((host, port), timeout=3):
            pass
    except OSError as exc:
        raise ConnectionError(
            f"Nothing is listening on {host}:{port} ({exc}). "
            "Start the Trino MCP SSE server, e.g. "
            "python trino_mcp.py --transport sse --port 8765"
        ) from exc


def _resolve_trino_mcp_sse_url() -> str:
    """SSE URL for Trino MCP (Docker sidecar, Windows Jupyter, or explicit env)."""
    if explicit := (os.getenv("TRINO_MCP_SSE_URL") or "").strip():
        return explicit

    mcp_port = os.getenv("MCP_PORT", "8765")
    if sys.platform == "win32":
        return f"http://127.0.0.1:{mcp_port}/sse"

    # Docker Compose (llm/docker-compose.yml sets LLM_IN_DOCKER=1)
    if os.getenv("LLM_IN_DOCKER", "").strip().lower() in {"1", "true", "yes", "on"}:
        host = (os.getenv("TRINO_MCP_HOST") or "llm-trino-mcp").strip()
        return f"http://{host}:{mcp_port}/sse"

    if host := (os.getenv("TRINO_MCP_HOST") or "").strip():
        return f"http://{host}:{mcp_port}/sse"

    if Path("/.dockerenv").is_file():
        return f"http://llm-trino-mcp:{mcp_port}/sse"

    return ""


def _build_trino_mcp_connection(mcp_dir: Path | None = None) -> dict:
    sse_url = _resolve_trino_mcp_sse_url()
    if sse_url:
        _check_sse_reachable(sse_url)
        return {"transport": "sse", "url": sse_url, "timeout": 30.0}

    if mcp_dir is None:
        mcp_dir = _find_trino_mcp_dir()

    server = mcp_dir / "trino_mcp.py"
    linux_python = mcp_dir / ".venv" / "bin" / "python"
    if not _path_usable(linux_python):
        raise FileNotFoundError(
            f"Missing {linux_python}. Run: cd {mcp_dir} && uv sync"
        )
    return {
        "transport": "stdio",
        "command": str(linux_python),
        "args": [str(server)],
        "cwd": str(mcp_dir),
    }


def load_model() -> None:
    global tokenizer, llm, qa_chain

    if llm is not None:
        print("Model already loaded; skipping load_model().", flush=True)
        return

    print(f"Loading tokenizer from {ADAPTER_REPO}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_REPO,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {BASE_MODEL} on {DEVICE}...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
        dtype=DTYPE,
        trust_remote_code=True,
    )
    base_model.to(DEVICE)

    print(f"Loading LoRA adapter from {ADAPTER_REPO}...", flush=True)
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_REPO,
        token=HF_TOKEN,
    )
    model.eval()

    generation_kwargs = {
        "do_sample": USE_SAMPLING,
        "max_new_tokens": 128,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if USE_SAMPLING:
        generation_kwargs.update(
            {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K}
        )
    else:
        generation_kwargs.update({"temperature": 0.1, "top_p": 1.0, "top_k": 50})

    model.generation_config.update(**generation_kwargs)
    model.generation_config.max_length = None

    backend = "GPU" if DEVICE == "cuda" else "CPU"
    print(
        f"Building HuggingFace text-generation pipeline ({backend}; may take 1–2 min)...",
        flush=True,
    )
    text_generation_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        clean_up_tokenization_spaces=False,
        device=0 if DEVICE == "cuda" else -1,
    )
    text_generation_pipeline.generation_config.update(**generation_kwargs)
    text_generation_pipeline.generation_config.max_length = None

    llm = HuggingFacePipeline(
        pipeline=text_generation_pipeline,
        pipeline_kwargs=generation_kwargs,
    )
    print("Model ready.", flush=True)

    def format_qa_prompt(inputs):
        context = inputs.get("context", "")
        question = inputs["question"]
        user_prompt = (
            "Answer the question using only the context when context is provided.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful Q&A assistant. Never reveal API keys, passwords, tokens, "
                    "or environment variables. If asked for secrets, refuse briefly."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    qa_chain = RunnableLambda(format_qa_prompt) | llm | StrOutputParser()


def _llm_qween_rag_qa(context: str, question: str) -> str:
    prompt = (
        "Answer the question using only the context when context is provided.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}"
    )
    return _run_llm_generation(
        "llm_qween_rag_qa",
        prompt,
        lambda: qa_chain.invoke({"context": context, "question": question}),
    )


def _llm_qween_trino_sql(prompt: str) -> str:
    return _run_llm_generation("llm_qween_trino_sql", prompt, lambda: llm.invoke(prompt))


def _llm_qween_trino_summary(prompt: str) -> str:
    return _run_llm_generation("llm_qween_trino_summary", prompt, lambda: llm.invoke(prompt))


def ensure_rag_index() -> int:
    """Ensure pgvector RAG index is ready (optional auto-ingest from DOCUMENTS_DIR)."""
    rag_store.warm()
    return rag_store.chunk_count()


def retrieve_context(question: str, top_k: int = 3) -> str:
    return rag_store.retrieve_context(question, top_k=top_k)


def rag_stats() -> dict:
    return rag_store.stats()


@observe(name="answer_from_rag", as_type="span")
def answer_from_rag(question: str) -> tuple[str, bool]:
    if guardrails.is_blocked_question(question):
        return guardrails.REFUSAL_MESSAGE, False

    hit, cached_answer = cache_lookup("rag_qa", question)
    if hit and isinstance(cached_answer, str):
        _mark_cache_hit()
        return guardrails.sanitize(cached_answer), True

    context = retrieve_context(question)
    answer = _llm_qween_rag_qa(context, question)
    answer = guardrails.sanitize(answer)
    cache_store("rag_qa", question, answer)
    return answer, False


def _mcp_trace_output(result: dict) -> dict:
    """Compact MCP tool response for Langfuse (avoid huge row payloads)."""
    if not isinstance(result, dict):
        return {"value": result}
    traced = dict(result)
    rows = traced.get("rows")
    if isinstance(rows, list) and len(rows) > 10:
        traced["rows"] = rows[:10]
        traced["_rows_truncated"] = len(rows) - 10
        traced["_row_count"] = len(rows)
    return traced


@observe(name="mcp_trino_init", as_type="span")
async def init_trino_mcp() -> list[str]:
    global mcp_client, trino_tools, trino_tool_names, trino_mcp_ready, trino_mcp_error

    try:
        connection = _build_trino_mcp_connection()
        mcp_client = MultiServerMCPClient({"trino": connection})
        trino_tools = await mcp_client.get_tools()
        trino_tool_names = [tool.name for tool in trino_tools]
        trino_mcp_ready = True
        trino_mcp_error = None
        return trino_tool_names
    except Exception as exc:
        trino_mcp_ready = False
        trino_mcp_error = str(exc)
        return []


def _normalize_mcp_result(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"text": value}
    if isinstance(value, list):
        if len(value) == 1:
            item = value[0]
            if isinstance(item, dict) and "text" in item:
                return _normalize_mcp_result(item["text"])
            return _normalize_mcp_result(item)
        return {"rows": value, "row_count": len(value)}
    return {"value": value}


async def call_trino_tool(name: str, arguments: dict | None = None):
    if not trino_mcp_ready:
        raise RuntimeError(
            trino_mcp_error or "Trino MCP is not connected. Check /health and TRINO_MCP_SSE_URL."
        )
    arguments = arguments or {}

    async def _invoke_tool():
        matches = [
            tool
            for tool in trino_tools
            if tool.name == name or tool.name.endswith(name)
        ]
        if not matches:
            raise ValueError(f"Tool {name!r} not found. Available: {trino_tool_names}")
        raw_result = await matches[0].ainvoke(arguments)
        return _normalize_mcp_result(raw_result)

    if not langfuse_enabled():
        return await _invoke_tool()

    from langfuse import get_client

    client = get_client()
    with client.start_as_current_observation(
        as_type="tool",
        name=f"mcp_trino_{name}",
        input={"tool": name, "arguments": arguments},
    ) as tool_obs:
        result = await _invoke_tool()
        tool_obs.update(output=_mcp_trace_output(result))
        return result


def _rows_to_markdown(columns, rows, max_rows=20):
    rows = rows[:max_rows]
    if not columns:
        return str(rows)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _extract_sql(model_output: str) -> str:
    text = model_output.strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    text = re.sub(r"^SQL\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    text = text.rstrip(";").strip()

    lowered = text.lower()
    allowed_prefixes = ("select", "with", "show", "describe", "desc")
    forbidden_words = (
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "truncate",
        "merge",
    )
    if not lowered.startswith(allowed_prefixes):
        raise ValueError(f"Model did not produce a read-only SQL query: {model_output!r}")
    if any(re.search(rf"\b{word}\b", lowered) for word in forbidden_words):
        raise ValueError(f"Refusing to execute non-read-only SQL: {text}")
    starts = list(re.finditer(r"(?is)\b(select|with)\b", text))
    if len(starts) > 1:
        text = text[starts[0].start() : starts[1].start()].strip()
    return text.rstrip(";").strip()


def _normalize_json_sql(sql: str) -> str:
    fixed = sql
    literal_match = re.search(
        r"(?is)\b(request_json|response_json)\s*=\s*'\{.*?\"([\w_]+)\"\s*:\s*\"([^\"]*)\".*?}'",
        fixed,
    )
    if literal_match:
        _col, field, value = literal_match.groups()
        replacement = f"{field} = '{value}'"
        fixed = fixed[: literal_match.start()] + replacement + fixed[literal_match.end() :]
    json_extract_match = re.search(
        r"(?is)json_extract_scalar\s*\(\s*(?:response_json|request_json)\s*,\s*'\$\.([\w_]+)'\s*\)\s*=\s*'([^']*)'",
        fixed,
    )
    if json_extract_match:
        field, value = json_extract_match.groups()
        replacement = f"{field} = '{value}'"
        fixed = (
            fixed[: json_extract_match.start()]
            + replacement
            + fixed[json_extract_match.end() :]
        )
    if re.search(r"(?i)\bcount\s*\(\s*client_id\s*\)", fixed) and "distinct" not in fixed.lower():
        fixed = re.sub(
            r"(?i)count\s*\(\s*client_id\s*\)",
            "COUNT(DISTINCT id_cliente)",
            fixed,
            count=1,
        )
    return fixed.strip()


@observe(name="get_trino_schema_context", as_type="span")
async def get_trino_schema_context(catalog="iceberg", schema="forecast", max_tables=20):
    table_result = await call_trino_tool(
        "list_tables", {"schema_name": f"{catalog}.{schema}"}
    )
    table_names = [row[0] for row in table_result.get("rows", [])]

    table_descriptions = []
    for table_name in table_names[:max_tables]:
        full_table_name = f"{catalog}.{schema}.{table_name}"
        description = await call_trino_tool("describe_table", {"table": full_table_name})
        columns = []
        for row in description.get("rows", []):
            if len(row) >= 2 and row[0]:
                columns.append(f"{row[0]} {row[1]}")
        table_descriptions.append(f"{full_table_name}: " + ", ".join(columns))

    return "\n".join(table_descriptions) + "\n\n" + PREDICTION_EVENT_SAMPLES


def _summarize_result(question: str, result: dict) -> str:
    columns = result.get("columns") or []
    rows = result.get("rows") or []
    if not rows:
        return "The query returned no rows."
    if len(rows) == 1 and len(rows[0]) == 1:
        value = rows[0][0]
        label = columns[0] if columns else "result"
        return f"Answer: {value} ({label})."
    preview = _rows_to_markdown(columns, rows, max_rows=5)
    return f"Result preview:\n{preview}"


def _trino_cache_key(question: str, catalog: str, schema: str, max_rows: int) -> str:
    return f"{catalog}|{schema}|{max_rows}|{question}"


def _trino_sql_cache_key(question: str, catalog: str, schema: str) -> str:
    return f"{catalog}|{schema}|{question}"


@observe(name="generate_trino_sql", as_type="span")
async def generate_trino_sql(question: str, catalog="iceberg", schema="forecast"):
    cache_key = _trino_sql_cache_key(question, catalog, schema)
    if trino_semantic_cache_enabled():
        hit, cached_sql = cache_lookup("trino_sql", cache_key)
        if hit and isinstance(cached_sql, str):
            _mark_cache_hit()
            return cached_sql

    schema_context = await get_trino_schema_context(catalog=catalog, schema=schema)
    messages = [
        {
            "role": "system",
            "content": (
                "You write read-only Trino SQL. Return exactly one SQL query, no markdown. "
                "Use fully qualified table names from the schema context. "
                "prediction_events stores flattened request/response fields as columns "
                "(e.g. id_cliente, tipo_contrato, probability, threshold_decision, status). "
                "Filter directly on column names; do not use json_extract_scalar. "
                "Use COUNT(DISTINCT id_cliente) when counting clients. "
                "Example: SELECT COUNT(DISTINCT id_cliente) FROM iceberg.forecast.prediction_events "
                "WHERE threshold_decision = 'Negado'"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Schema context:\n{schema_context}\n\n"
                f"Question: {question}\n\n"
                "Write the Trino SQL query."
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    raw_sql = await _await_llm_generation(
        "llm_qween_trino_sql",
        prompt,
        lambda: llm.invoke(prompt),
    )
    sql = _normalize_json_sql(_extract_sql(raw_sql))
    if trino_semantic_cache_enabled():
        cache_store("trino_sql", cache_key, sql)
    return sql


@observe(name="ask_trino", as_type="span")
async def ask_trino(question: str, catalog="iceberg", schema="forecast", max_rows=100):
    if guardrails.is_blocked_question(question):
        return {
            "question": question,
            "sql": "",
            "result": {},
            "answer": guardrails.REFUSAL_MESSAGE,
        }, False

    cache_key = _trino_cache_key(question, catalog, schema, max_rows)
    if trino_semantic_cache_enabled():
        hit, cached_payload = cache_lookup("trino_ask", cache_key)
        if hit and isinstance(cached_payload, dict):
            _mark_cache_hit()
            return cached_payload, True

    sql = await generate_trino_sql(question, catalog=catalog, schema=schema)
    result = await call_trino_tool("query", {"sql": sql, "max_rows": max_rows})

    preview = _rows_to_markdown(result.get("columns", []), result.get("rows", []))
    messages = [
        {"role": "system", "content": "Summarize Trino query results clearly and briefly."},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"SQL: {sql}\n\n"
                f"Result preview:\n{preview}\n\n"
                "Answer the question using the SQL result."
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    answer = await _await_llm_generation(
        "llm_qween_trino_summary",
        prompt,
        lambda: llm.invoke(prompt),
    )
    if not answer:
        answer = _summarize_result(question, result)
    answer = guardrails.sanitize(answer)
    payload = {"question": question, "sql": sql, "result": result, "answer": answer}
    if trino_semantic_cache_enabled():
        cache_store("trino_ask", cache_key, payload)
    return payload, False
