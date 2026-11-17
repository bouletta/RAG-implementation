from time import perf_counter

from basic_rag_agent.context import pack_sources, payload
from basic_rag_agent.corpus import sources
from basic_rag_agent.models import Query, Reply
from basic_rag_agent.validation import validate_reply
from embedding_rag_agent.embeddings import embed_checked
from embedding_rag_agent.models import Match, Usage
from semantic_memory_agent.embeddings import Embedder

from shared.errors import AgentError
from shared.structured import StructuredModel, fingerprint, request

from .config import Settings
from .models import Answer, Snapshot
from .prompts import SYSTEM_PROMPT
from .store import Store


def answer(
    snapshot: Snapshot,
    query: Query,
    ids: list[str],
    settings: Settings,
    *,
    matches: list[Match],
    usage: Usage,
    method: str,
    model: StructuredModel | None,
    retrieve_only: bool,
    started: float,
    live: bool = False,
) -> Answer:
    lookup = {source.id: source for source in sources(snapshot.build.corpus)}
    if len(ids) != len(set(ids)) or len(ids) > settings.top_k or not set(ids) <= lookup.keys():
        raise AgentError("Retrieval returned invalid passage IDs.")
    context = pack_sources(query, [lookup[id_] for id_ in ids], settings.max_context_chars)
    reply = None
    if context.sources and not retrieve_only:
        if model is None:
            raise AgentError("Generation requires a model; use retrieval-only explicitly.")
        reply = request(
            model, SYSTEM_PROMPT, payload(query, context.sources), Reply, settings.max_context_chars
        )
        validate_reply(reply, context)
    reason = (
        "no_matches"
        if not ids
        else "context_budget"
        if not context.sources
        else "retrieval_only"
        if retrieve_only
        else "answered"
        if reply is not None and reply.claims
        else "insufficient_evidence"
    )
    return Answer(
        mode="live" if live else "injected_provider",
        status="answered"
        if reason == "answered"
        else "retrieved"
        if reason == "retrieval_only"
        else "abstained",
        reason=reason,
        snapshot=snapshot.name,
        corpus_sha256=snapshot.build.manifest.corpus_sha256,
        retrieval_method=method,
        matches=matches,
        context=context,
        reply=reply,
        embedding_usage=usage,
        logical_generation_calls=int(reply is not None),
        generation_model=settings.model if reply is not None else None,
        prompt_sha256=fingerprint(SYSTEM_PROMPT),
        elapsed_ms=(perf_counter() - started) * 1000,
    )


def run(
    store: Store,
    name: str,
    query: Query,
    settings: Settings,
    embedder: Embedder,
    *,
    model: StructuredModel | None = None,
    retrieve_only: bool = False,
    live: bool = False,
) -> Answer:
    started = perf_counter()
    snapshot = store.load(name, settings)
    pack_sources(query, [], settings.max_context_chars)
    vectors, usage = embed_checked([query.question], embedder, settings)
    matches = store.search(snapshot, vectors[0], settings.top_k, settings.min_similarity)
    return answer(
        snapshot,
        query,
        [m.source_id for m in matches],
        settings,
        matches=matches,
        usage=usage,
        method="chroma_cosine",
        model=model,
        retrieve_only=retrieve_only,
        started=started,
        live=live,
    )
