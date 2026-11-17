from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context
from semantic_memory_agent.embeddings import Embedder

from shared.errors import AgentError
from shared.structured import StructuredModel, request

from .config import Settings
from .context import assemble, payload
from .models import Context, Query, Reply, Result, Retrieval
from .prompts import SYSTEM_PROMPT
from .retrieval import Stores, recheck, retrieve
from .validation import validate


class State(TypedDict, total=False):
    memory: dict[str, Any]
    context: dict[str, Any]
    reply: dict[str, Any]


def run(
    query: Query,
    stores: Stores,
    model: StructuredModel | None,
    embedder: Embedder | None,
    settings: Settings,
    *,
    output: Literal["answer", "preview", "retrieve"] = "answer",
) -> Result:
    if output == "preview" and query.mode != "lexical":
        raise AgentError("Preview is local-only; explicitly use lexical mode.")

    def recall(state: State) -> State:
        return {"memory": retrieve(query, stores, embedder, settings).model_dump(mode="json")}

    def pack(state: State) -> State:
        memory = Retrieval.model_validate(state["memory"])
        context = assemble(query, memory, settings)
        recheck(query, memory, context, stores)
        return {"context": context.model_dump()}

    def answer(state: State) -> State:
        if model is None:
            raise AgentError("Answer generation requires a model.")
        context = Context.model_validate(state["context"])
        reply = request(
            model, SYSTEM_PROMPT, payload(query, context), Reply, settings.max_context_chars
        )
        return {"reply": reply.model_dump()}

    def check(state: State) -> State:
        context = Context.model_validate(state["context"])
        validate(Reply.model_validate(state["reply"]), context)
        recheck(query, Retrieval.model_validate(state["memory"]), context, stores)
        return {}

    graph = StateGraph(State)
    graph.add_node("retrieve", recall)
    graph.add_node("assemble", pack)
    graph.add_node("answer", answer)
    graph.add_node("validate", check)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "assemble")
    graph.add_conditional_edges(
        "assemble",
        lambda s: "answer" if output == "answer" and s["context"]["sources"] else END,
        {"answer": "answer", END: END},
    )
    graph.add_edge("answer", "validate")
    graph.add_edge("validate", END)
    with tracing_context(enabled=False):
        state = graph.compile().invoke({}, {"recursion_limit": 6})
    memory = Retrieval.model_validate(state["memory"])
    context = Context.model_validate(state["context"])
    status: Literal["preview", "retrieved", "answered", "no_evidence"]
    if output == "preview":
        status = "preview"
    elif output == "retrieve":
        status = "retrieved"
    else:
        status = "answered" if context.sources else "no_evidence"
    supplied_fact_keys = {s.key for s in context.sources if s.kind == "asserted_fact"}
    return Result(
        status=status,
        mode=query.mode,
        context=context,
        fact_ranking=[row for row in memory.facts if row.fact.key in supplied_fact_keys],
        embedding_tokens=memory.embedding_tokens,
        reply=Reply.model_validate(state["reply"]) if "reply" in state else None,
    )
