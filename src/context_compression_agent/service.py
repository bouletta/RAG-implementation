from time import perf_counter

from basic_rag_agent.context import payload
from basic_rag_agent.corpus import sources
from basic_rag_agent.models import Query, Reply
from basic_rag_agent.prompts import SYSTEM_PROMPT
from basic_rag_agent.validation import validate_reply
from chroma_rag_agent.models import Answer
from chroma_rag_agent.store import Store
from multi_query_rag_agent.service import run as retrieve
from semantic_memory_agent.embeddings import Embedder

from shared.errors import AgentError
from shared.structured import StructuredModel, fingerprint, request

from .compression import compress
from .config import Settings
from .models import Result


def run(
    store: Store,
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
    generated = provided is None and settings.max_variants > 0
    required = int(generated) + int(settings.compression_mode == "llm") + int(not retrieve_only)
    if required > settings.max_model_calls:
        raise AgentError("Combined model call budget exceeded before any provider requests.")
    if settings.compression_mode == "llm" and compression_model is None:
        raise AgentError("LLM compression requires a selection model.")
    if not retrieve_only and answer_model is None:
        raise AgentError("Answer generation requires a model; use retrieval-only explicitly.")
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
    snapshot = store.load(name, settings)
    if snapshot.build.manifest.corpus_sha256 != retrieved.answer.corpus_sha256:
        raise AgentError("Snapshot changed after retrieval.")
    lookup = {source.id: source for source in sources(snapshot.build.corpus)}
    parents = [lookup[row.source_id] for row in retrieved.fusion]
    compressed = compress(query, parents, settings, compression_model)
    context = compressed.context
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
        if not parents
        else "no_selected_units"
        if not compressed.selected_ids
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
        corpus_sha256=retrieved.answer.corpus_sha256,
        retrieval_method="multi_query_hybrid_extract_v1",
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
        retrieval=retrieved,
        compression=compressed,
        answer=answer,
        logical_model_calls=retrieved.logical_model_calls
        + compressed.logical_model_calls
        + answer.logical_generation_calls,
    )
