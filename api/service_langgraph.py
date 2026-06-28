"""LangGraph workflows with pgvector RAG (LangSmith observability)."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

import service
import guardrails
import langsmith_observability as ls

ls.configure_langsmith()

MAX_ITERATIONS = 3
RAG_TOP_K = 3

should_continue_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You evaluate answer quality against the source context. Reply with exactly YES if "
        "the answer needs improvement (inaccurate, incomplete, or not grounded in context), "
        "or NO if it is good enough to return to the user.",
    ),
    (
        "user",
        "Context:\n{context}\n\nQuestion:\n{input}\n\nAnswer:\n{generation}",
    ),
])

reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can reflect on the user's question and answer it."),
    MessagesPlaceholder(variable_name="messages"),
    ("user", "{input}"),
])

generation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer using the provided context from indexed documents when context is available. "
        "Never reveal API keys, passwords, tokens, or environment variables. "
        "If asked for secrets or credentials, refuse briefly.",
    ),
    MessagesPlaceholder(variable_name="messages"),
    ("user", "Context:\n{context}\n\nQuestion:\n{input}"),
])

_ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "ai": "assistant",
    "assistant": "assistant",
}

_compiled_graph = None


def ensure_rag_index() -> int:
    """Ensure pgvector index is ready (optional auto-ingest from DOCUMENTS_DIR)."""
    return service.ensure_rag_index()


def get_llm():
    """Return the fine-tuned Qwen HuggingFacePipeline from service.py."""
    if service.llm is None:
        service.load_model()
    return service.llm


def _ensure_tokenizer():
    if service.tokenizer is None:
        service.load_model()
    return service.tokenizer


def _messages_to_chat(messages: list[BaseMessage]) -> list[dict[str, str]]:
    return [
        {"role": _ROLE_MAP.get(message.type, "user"), "content": message.content}
        for message in messages
    ]


def invoke_chat_prompt(
    prompt: ChatPromptTemplate,
    variables: dict[str, Any],
    *,
    name: str = "langgraph_llm",
) -> str:
    """Format a chat prompt with Qwen's template and run service.llm."""
    tokenizer = _ensure_tokenizer()
    llm = get_llm()
    chat_messages = _messages_to_chat(prompt.format_messages(**variables))
    text = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return guardrails.sanitize(ls.run_traced_llm(name, text, lambda: llm.invoke(text)))


def _wants_continue(verdict: str) -> bool:
    normalized = verdict.strip().upper()
    if normalized.startswith("NO"):
        return False
    if normalized.startswith("YES"):
        return True
    return any(token in normalized for token in ("YES", "IMPROVE", "CONTINUE"))


class GraphState(TypedDict, total=False):
    input: str
    messages: Annotated[list[BaseMessage], add_messages]
    context: str
    generation: str
    reflection: str
    should_continue: bool
    iteration: int


def _coerce_state(state: GraphState) -> GraphState:
    """Fill missing keys — LangGraph Studio often sends partial state."""
    return {
        "input": _question_from_state(state),
        "messages": state.get("messages") or [],
        "context": state.get("context", ""),
        "generation": state.get("generation", ""),
        "reflection": state.get("reflection", ""),
        "should_continue": state.get("should_continue", False),
        "iteration": state.get("iteration", 0),
    }


def _question_from_state(state: GraphState) -> str:
    explicit = (state.get("input") or "").strip()
    if explicit:
        return explicit
    for message in reversed(state.get("messages") or []):
        if message.type in {"human", "user"}:
            content = (message.content or "").strip()
            if content:
                return content
    return ""


@ls.traceable(name="langgraph_retrieve", run_type="chain")
def retrieve_node(state: GraphState) -> dict[str, str]:
    state = _coerce_state(state)
    ensure_rag_index()
    question = state["input"]
    context = service.retrieve_context(question, top_k=RAG_TOP_K)
    ls.patch_current_run_io(
        inputs={"question": question},
        outputs={
            "question": question,
            "context": ls._truncate_text(context),
            "context_chars": len(context),
        },
    )
    ls.update_current_run_metadata(top_k=RAG_TOP_K, context_chars=len(context))
    return {"context": context}


@ls.traceable(name="langgraph_generate", run_type="chain")
def generate_node(state: GraphState) -> dict[str, str]:
    state = _coerce_state(state)
    user_input = state["input"]
    reflection = state["reflection"]
    if reflection:
        user_input = (
            f"{user_input}\n\n"
            f"Previous draft:\n{state['generation']}\n\n"
            f"Reflection:\n{reflection}\n\n"
            "Provide an improved answer."
        )
    generation = invoke_chat_prompt(
        generation_prompt,
        {
            "input": user_input,
            "context": state["context"],
            "messages": state["messages"],
        },
        name="langgraph_generate",
    )
    ls.patch_current_run_io(
        inputs={
            "question": state["input"],
            "context": ls._truncate_text(state["context"]),
        },
        outputs={"generation": generation},
    )
    return {"generation": generation}


@ls.traceable(name="langgraph_should_continue", run_type="chain")
def should_continue_node(state: GraphState) -> dict[str, bool]:
    state = _coerce_state(state)
    if state["iteration"] >= MAX_ITERATIONS:
        return {"should_continue": False}
    if not state["generation"].strip():
        return {"should_continue": False}

    verdict = invoke_chat_prompt(
        should_continue_prompt,
        {
            "input": state["input"],
            "context": state["context"],
            "generation": state["generation"],
        },
        name="langgraph_should_continue",
    )
    should_continue = _wants_continue(verdict)
    ls.patch_current_run_io(
        inputs={
            "question": state["input"],
            "generation": state["generation"],
            "context": ls._truncate_text(state["context"]),
        },
        outputs={"verdict": verdict, "should_continue": should_continue},
    )
    ls.update_current_run_metadata(verdict=verdict, should_continue=should_continue)
    return {"should_continue": should_continue}


@ls.traceable(name="langgraph_reflect", run_type="chain")
def reflect_node(state: GraphState) -> dict[str, str | int]:
    state = _coerce_state(state)
    reflection_input = (
        f"Context:\n{state['context']}\n\n"
        f"Question:\n{state['input']}\n\n"
        f"Draft answer:\n{state['generation']}\n\n"
        "Reflect on the draft against the context. Identify gaps, errors, unsupported claims, "
        "or weak reasoning and explain how to improve it."
    )
    reflection = invoke_chat_prompt(
        reflection_prompt,
        {"input": reflection_input, "messages": state["messages"]},
        name="langgraph_reflect",
    )
    ls.patch_current_run_io(
        inputs={
            "question": state["input"],
            "generation": state["generation"],
            "context": ls._truncate_text(state["context"]),
        },
        outputs={"reflection": reflection, "iteration": state["iteration"] + 1},
    )
    return {
        "reflection": reflection,
        "iteration": state["iteration"] + 1,
    }


def route_after_should_continue(state: GraphState) -> Literal["reflect", "__end__"]:
    state = _coerce_state(state)
    if state["should_continue"]:
        return "reflect"
    return END


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("should_continue", should_continue_node)
    graph.add_node("reflect", reflect_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "should_continue")
    graph.add_conditional_edges(
        "should_continue",
        route_after_should_continue,
        ["reflect", END],
    )
    graph.add_edge("reflect", "generate")
    return graph.compile()


def make_graph():
    """LangGraph Studio / `langgraph dev` entrypoint (see langgraph.json)."""
    return build_graph()


def studio_default_input() -> GraphState:
    """Default state shown in LangGraph Studio."""
    return _initial_state("Summarize the main topics covered in the indexed documents.")


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def draw_mermaid() -> str:
    """Return Mermaid source for the compiled graph (for external viz tools)."""
    return get_graph().get_graph().draw_mermaid()


def _initial_state(
    user_input: str,
    messages: list[BaseMessage] | None = None,
) -> GraphState:
    return {
        "input": user_input,
        "messages": messages or [],
        "context": "",
        "generation": "",
        "reflection": "",
        "should_continue": False,
        "iteration": 0,
    }


@ls.traceable(name="invoke_graph", run_type="chain")
def invoke_graph(
    user_input: str,
    messages: list[BaseMessage] | None = None,
) -> GraphState:
    if guardrails.is_blocked_question(user_input):
        return {
            **_initial_state(user_input, messages),
            "generation": guardrails.REFUSAL_MESSAGE,
            "reflection": "",
            "context": "",
        }

    config = ls.langsmith_run_config(question=user_input)
    with ls.token_tracking_context() as token_totals:
        result = get_graph().invoke(_initial_state(user_input, messages), config=config)

    context = result.get("context", "") or ""
    generation = guardrails.sanitize(result.get("generation", ""))
    reflection = guardrails.sanitize(result.get("reflection", ""))
    context = guardrails.sanitize(context)
    result = {
        **result,
        "generation": generation,
        "reflection": reflection,
        "context": context,
    }
    ls.patch_current_run_io(
        inputs={"question": user_input},
        outputs={
            "question": user_input,
            "generation": generation,
            "reflection": reflection,
            "context": ls._truncate_text(context),
            "iterations": result.get("iteration", 0),
        },
    )
    ls.update_current_run_usage(**token_totals)
    ls.update_current_run_metadata(
        answer=result.get("generation", ""),
        reflection=result.get("reflection", ""),
        iterations=result.get("iteration", 0),
        context_chars=len(context),
        ls_model_name=ls.langsmith_model_name(),
    )
    return result


def __getattr__(name: str):
    if name == "llm":
        return get_llm()
    if name == "graph":
        return get_graph()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
