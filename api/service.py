"""Core logic from langchain.ipynb — model, RAG, and Trino MCP."""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFacePipeline
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from peft import PeftModel
from pypdf import PdfReader
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


BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_REPO = os.getenv("ADAPTER_REPO", "Glccampos/llm_qween")
HF_TOKEN = os.getenv("HF_TOKEN")
PDF_PATH = os.getenv("PDF_PATH", str(API_DIR / "Analytics Engineer .pdf"))

USE_SAMPLING = True
TEMPERATURE = 0.1
TOP_P = 0.9
TOP_K = 50

PREDICTION_JSON_SAMPLES = """
request_json example:
{"id_cliente": "292659", "id_contrato": null, "tipo_contrato": "Cash loans", "status_contrato": "ativo", "tipo_pagamento": "boleto", "finalidade_emprestimo": "compra_veiculo", "tipo_cliente": "pessoa_fisica", "tipo_portfolio": "varejo", "tipo_produto": "credito_pessoal", "categoria_bem": "automovel", "setor_vendedor": "digital", "canal_venda": "online", "faixa_rendimento": null, "combinacao_produto": null, "area_venda": null, "dia_semana_solicitacao": "SATURDAY", "data_nascimento": "1997-04-08", "data_decisao": "2024-01-01", "data_liberacao": null, "data_primeiro_vencimento": null, "data_ultimo_vencimento_original": null, "data_ultimo_vencimento": null, "data_encerramento": null, "valor_solicitado": 0.0, "valor_credito": 439740.0, "valor_bem": 315000.0, "valor_parcela": 23985.0, "valor_entrada": 0.0, "percentual_entrada": 0.0, "qtd_parcelas_planejadas": 12, "taxa_juros_padrao": 0.03, "taxa_juros_promocional": 0.03, "hora_solicitacao": 14, "flag_ultima_solicitacao_contrato": 0, "flag_ultima_solicitacao_dia": 0, "acompanhantes_cliente": 0, "flag_seguro_contratado": 0, "motivo_recusa": null, "renda_anual": 157500.0, "qtd_membros_familia": 1, "possui_carro": "N", "possui_imovel": "Y"}

response_json example:
{"request_id": "02e60220-9a7e-418f-abc8-251c8111dfb3", "probability": 0.9504, "threshold_decision": "Negado", "status": "success"}
"""

tokenizer = None
llm = None
qa_chain = None
chunks: list[str] = []
chunk_term_counts: list[Counter] = []
document_frequencies: Counter = Counter()

mcp_client: MultiServerMCPClient | None = None
trino_tools: list = []
trino_tool_names: list[str] = []
trino_mcp_ready = False
trino_mcp_error: str | None = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


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

    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_REPO,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
        dtype=DTYPE,
        trust_remote_code=True,
    )
    base_model.to(DEVICE)

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

    def format_qa_prompt(inputs):
        context = inputs.get("context", "")
        question = inputs["question"]
        user_prompt = (
            "Answer the question using only the context when context is provided.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}"
        )
        messages = [
            {"role": "system", "content": "You are a helpful Q&A assistant."},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    qa_chain = RunnableLambda(format_qa_prompt) | llm | StrOutputParser()


@observe(name="llm_qween_pdf_qa", as_type="generation")
def _llm_qween_pdf_qa(context: str, question: str) -> str:
    return qa_chain.invoke({"context": context, "question": question}).strip()


@observe(name="llm_qween_trino_sql", as_type="generation")
def _llm_qween_trino_sql(prompt: str) -> str:
    return llm.invoke(prompt).strip()


@observe(name="llm_qween_trino_summary", as_type="generation")
def _llm_qween_trino_summary(prompt: str) -> str:
    return llm.invoke(prompt).strip()


def load_pdf_index(pdf_path: str | None = None) -> int:
    global chunks, chunk_term_counts, document_frequencies

    path = Path(pdf_path or PDF_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(str(path))
    pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )
    chunks = text_splitter.split_text(pdf_text)
    chunk_term_counts = [Counter(_tokenize_for_retrieval(chunk)) for chunk in chunks]
    document_frequencies = Counter(
        term for counts in chunk_term_counts for term in counts.keys()
    )
    return len(chunks)


def _tokenize_for_retrieval(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def retrieve_context(question: str, top_k: int = 3) -> str:
    query_terms = _tokenize_for_retrieval(question)
    if not query_terms:
        return "\n\n".join(chunks[:top_k])

    scored_chunks = []
    for index, (chunk, term_counts) in enumerate(zip(chunks, chunk_term_counts)):
        score = 0.0
        for term in query_terms:
            if term not in term_counts:
                continue
            idf = math.log((len(chunks) + 1) / (document_frequencies[term] + 1)) + 1
            score += term_counts[term] * idf
        scored_chunks.append((score, index, chunk))

    best_chunks = [
        chunk
        for score, _, chunk in sorted(scored_chunks, reverse=True)[:top_k]
        if score > 0
    ]
    return "\n\n---\n\n".join(best_chunks or chunks[:top_k])


@observe(name="answer_from_pdf", as_type="span")
def answer_from_pdf(question: str) -> str:
    context = retrieve_context(question)
    return _llm_qween_pdf_qa(context, question)


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


def _annotate_mcp_tool_span(name: str, arguments: dict) -> None:
    if not langfuse_enabled():
        return
    try:
        from langfuse import get_client

        get_client().update_current_span(
            name=f"mcp_trino_{name}",
            input={"tool": name, "arguments": arguments},
        )
    except Exception:
        pass


def _record_mcp_tool_output(result: dict) -> None:
    if not langfuse_enabled():
        return
    try:
        from langfuse import get_client

        get_client().update_current_span(output=_mcp_trace_output(result))
    except Exception:
        pass


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


@observe(as_type="tool", capture_output=False)
async def call_trino_tool(name: str, arguments: dict | None = None):
    if not trino_mcp_ready:
        raise RuntimeError(
            trino_mcp_error or "Trino MCP is not connected. Check /health and TRINO_MCP_SSE_URL."
        )
    arguments = arguments or {}
    _annotate_mcp_tool_span(name, arguments)
    matches = [
        tool
        for tool in trino_tools
        if tool.name == name or tool.name.endswith(name)
    ]
    if not matches:
        raise ValueError(f"Tool {name!r} not found. Available: {trino_tool_names}")
    raw_result = await matches[0].ainvoke(arguments)
    result = _normalize_mcp_result(raw_result)
    _record_mcp_tool_output(result)
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
        col, field, value = literal_match.groups()
        replacement = f"json_extract_scalar({col}, '$.{field}') = '{value}'"
        fixed = fixed[: literal_match.start()] + replacement + fixed[literal_match.end() :]
    if re.search(r"(?i)\bcount\s*\(\s*client_id\s*\)", fixed) and "distinct" not in fixed.lower():
        fixed = re.sub(
            r"(?i)count\s*\(\s*client_id\s*\)",
            "COUNT(DISTINCT client_id)",
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

    return "\n".join(table_descriptions) + "\n\n" + PREDICTION_JSON_SAMPLES


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


@observe(name="generate_trino_sql", as_type="span")
async def generate_trino_sql(question: str, catalog="iceberg", schema="forecast"):
    schema_context = await get_trino_schema_context(catalog=catalog, schema=schema)
    messages = [
        {
            "role": "system",
            "content": (
                "You write read-only Trino SQL. Return exactly one SQL query, no markdown. "
                "Use fully qualified table names from the schema context. "
                "prediction_events: request_json = HTTP request body (VARCHAR JSON); "
                "response_json = model output (VARCHAR JSON) with keys like threshold_decision, probability, status. "
                "NEVER compare request_json/response_json to a JSON literal string. "
                "ALWAYS filter with json_extract_scalar(column, '$.field') = 'value'. "
                "Use COUNT(DISTINCT client_id) when counting clients. "
                "Example: SELECT COUNT(DISTINCT client_id) FROM iceberg.forecast.prediction_events "
                "WHERE json_extract_scalar(response_json, '$.threshold_decision') = 'Negado'"
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
    raw_sql = _llm_qween_trino_sql(prompt)
    return _normalize_json_sql(_extract_sql(raw_sql))


@observe(name="ask_trino", as_type="span")
async def ask_trino(question: str, catalog="iceberg", schema="forecast", max_rows=100):
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
    answer = _llm_qween_trino_summary(prompt)
    if not answer:
        answer = _summarize_result(question, result)
    return {"question": question, "sql": sql, "result": result, "answer": answer}
