"""LangGraph workflows with RAG over Analytics Engineer .pdf (same index as service.py)."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

import service
from service import observe

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
        "Answer also using the provided context from the "
        "Analytics Engineer job description PDF when context is available.",
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


def ensure_pdf_index() -> int:
    """Load and index Analytics Engineer .pdf if not already indexed."""
    if not service.chunks:
        return service.load_pdf_index()
    return len(service.chunks)


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
    return service._run_llm_generation(name, text, lambda: llm.invoke(text))


def _langfuse_callbacks() -> list[Any]:
    if not service.langfuse_enabled():
        return []
    try:
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    except Exception:
        return []


def _wants_continue(verdict: str) -> bool:
    normalized = verdict.strip().upper()
    if normalized.startswith("NO"):
        return False
    if normalized.startswith("YES"):
        return True
    return any(token in normalized for token in ("YES", "IMPROVE", "CONTINUE"))


class GraphState(TypedDict):
    input: str
    messages: Annotated[list[BaseMessage], add_messages]
    context: str
    generation: str
    reflection: str
    should_continue: bool
    iteration: int


@observe(name="langgraph_retrieve", as_type="span")
def retrieve_node(state: GraphState) -> dict[str, str]:
    ensure_pdf_index()
    context = service.retrieve_context(state["input"], top_k=RAG_TOP_K)
    if service.langfuse_enabled():
        try:
            from langfuse import get_client

            get_client().update_current_span(
                metadata={"top_k": RAG_TOP_K, "context_chars": len(context)},
            )
        except Exception:
            pass
    return {"context": context}


@observe(name="langgraph_generate", as_type="span")
def generate_node(state: GraphState) -> dict[str, str]:
    user_input = state["input"]
    reflection = state.get("reflection", "")
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
            "context": state.get("context", ""),
            "messages": state.get("messages", []),
        },
        name="langgraph_generate",
    )
    return {"generation": generation}


@observe(name="langgraph_should_continue", as_type="span")
def should_continue_node(state: GraphState) -> dict[str, bool]:
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return {"should_continue": False}

    verdict = invoke_chat_prompt(
        should_continue_prompt,
        {
            "input": state["input"],
            "context": state.get("context", ""),
            "generation": state["generation"],
        },
        name="langgraph_should_continue",
    )
    should_continue = _wants_continue(verdict)
    if service.langfuse_enabled():
        try:
            from langfuse import get_client

            get_client().update_current_span(
                metadata={"verdict": verdict, "should_continue": should_continue},
            )
        except Exception:
            pass
    return {"should_continue": should_continue}


@observe(name="langgraph_reflect", as_type="span")
def reflect_node(state: GraphState) -> dict[str, str | int]:
    reflection_input = (
        f"Context:\n{state.get('context', '')}\n\n"
        f"Question:\n{state['input']}\n\n"
        f"Draft answer:\n{state['generation']}\n\n"
        "Reflect on the draft against the context. Identify gaps, errors, unsupported claims, "
        "or weak reasoning and explain how to improve it."
    )
    reflection = invoke_chat_prompt(
        reflection_prompt,
        {"input": reflection_input, "messages": state.get("messages", [])},
        name="langgraph_reflect",
    )
    return {
        "reflection": reflection,
        "iteration": state.get("iteration", 0) + 1,
    }


def route_after_should_continue(state: GraphState) -> Literal["reflect", "__end__"]:
    if state.get("should_continue", False):
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


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def draw_mermaid() -> str:
    """Return Mermaid source for the compiled graph (for external viz tools)."""
    return get_graph().get_graph().draw_mermaid()


@observe(name="invoke_graph", as_type="span")
def invoke_graph(
    user_input: str,
    messages: list[BaseMessage] | None = None,
) -> GraphState:
    config: dict[str, Any] = {}
    callbacks = _langfuse_callbacks()
    if callbacks:
        config["callbacks"] = callbacks

    result = get_graph().invoke(
        {
            "input": user_input,
            "messages": messages or [],
            "context": "",
            "generation": "",
            "reflection": "",
            "should_continue": False,
            "iteration": 0,
        },
        config=config,
    )

    if service.langfuse_enabled():
        try:
            from langfuse import get_client

            get_client().update_current_span(
                input={"question": user_input},
                output={
                    "answer": result.get("generation", ""),
                    "reflection": result.get("reflection", ""),
                    "iterations": result.get("iteration", 0),
                    "context_chars": len(result.get("context", "")),
                },
            )
        except Exception:
            pass

    return result


def __getattr__(name: str):
    if name == "llm":
        return get_llm()
    if name == "graph":
        return get_graph()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
