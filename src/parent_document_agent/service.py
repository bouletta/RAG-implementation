from time import perf_counter

from basic_rag_agent.context import pack_sources, payload
from basic_rag_agent.corpus import sources
from basic_rag_agent.models import Query, Reply
from basic_rag_agent.prompts import SYSTEM_PROMPT
from basic_rag_agent.validation import validate_reply
from chroma_rag_agent.models import Answer
from context_compression_agent.compression import compress
from multi_query_rag_agent.service import run as retrieve
from semantic_memory_agent.embeddings import Embedder

from shared.errors import AgentError
from shared.structured import StructuredModel, fingerprint, request

from .config import Settings
from .hierarchy import expand
from .models import Result
from .store import ParentStore


def run(
    store: ParentStore,
    name: str,
    query: Query,
    settings: Settings,
    embedder: Embedder,
    *,
    expansion_model: StructuredModel | None = None,
    compression_model: StructuredModel | None = None,
    answer_model: StructuredModel | None = None,
    provided: list[str] | None = None,
    retrieve_only: bool = False,
    live: bool = False,
) -> Result:
    started = perf_counter()
    hierarchy = store.load_hierarchy(name, settings)
    selection_calls = int(settings.compress_parents and settings.compression_mode == "llm")
    required = (
        int(provided is None and settings.max_variants > 0)
        + selection_calls
        + int(not retrieve_only)
    )
    if required > settings.max_model_calls:
        raise AgentError("Parent retrieval model call budget exceeded before provider requests.")
    if selection_calls and compression_model is None:
        raise AgentError("Parent compression requires a selection model.")
    if not retrieve_only and answer_model is None:
        raise AgentError("Generation requires a model; use retrieval-only explicitly.")
    retrieved = retrieve(
        store,
        name,
        query,
        settings,
        embedder,
        expansion_model=expansion_model,
        provided=provided,
        retrieve_only=True,
        live=live,
    )
    if hierarchy.sha256 != store.load_hierarchy(name, settings).sha256:
        raise AgentError("Hierarchy changed during retrieval.")
    hits = expand(hierarchy, [r.source_id for r in retrieved.fusion], settings.max_parents)
    lookup = {s.id: s for s in sources(hierarchy.parents.corpus)}
    parents = [lookup[hit.parent_id] for hit in hits]
    compressed = (
        compress(query, parents, settings, compression_model) if settings.compress_parents else None
    )
    context = (
        compressed.context
        if compressed
        else pack_sources(query, parents, settings.max_context_chars)
    )
    reply = None
    if context.sources and not retrieve_only:
        assert answer_model is not None
        reply = request(
            answer_model,
            SYSTEM_PROMPT,
            payload(query, context.sources),
            Reply,
            settings.max_context_chars,
        )
        validate_reply(reply, context)
    reason = (
        "no_matches"
        if not hits
        else "no_selected_units"
        if compressed and not compressed.selected_ids
        else "context_budget"
        if not context.sources
        else "retrieval_only"
        if retrieve_only
        else "answered"
        if reply is not None and reply.claims
        else "insufficient_evidence"
    )
    answer = Answer(
        mode="live" if live else "injected_provider",
        status="answered"
        if reason == "answered"
        else "retrieved"
        if reason == "retrieval_only"
        else "abstained",
        reason=reason,
        snapshot=name,
        corpus_sha256=hierarchy.parents.manifest.corpus_sha256,
        retrieval_method="child_search_parent_context_v1",
        matches=retrieved.answer.matches,
        context=context,
        reply=reply,
        embedding_usage=retrieved.answer.embedding_usage,
        logical_generation_calls=int(reply is not None),
        generation_model=settings.model if reply is not None else None,
        prompt_sha256=fingerprint(SYSTEM_PROMPT),
        elapsed_ms=(perf_counter() - started) * 1000,
    )
    return Result(
        hierarchy_sha256=hierarchy.sha256,
        retrieval=retrieved,
        parents=hits,
        compression=compressed,
        answer=answer,
        logical_model_calls=retrieved.logical_model_calls
        + (compressed.logical_model_calls if compressed else 0)
        + answer.logical_generation_calls,
    )
