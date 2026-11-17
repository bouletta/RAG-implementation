from collections.abc import Callable
from time import perf_counter
from typing import Literal

from basic_rag_agent.context import pack_sources, payload
from basic_rag_agent.corpus import sources
from basic_rag_agent.models import Query, Reply
from basic_rag_agent.prompts import SYSTEM_PROMPT
from basic_rag_agent.validation import validate_reply
from chunking_agent.models import Build
from semantic_memory_agent.embeddings import Embedder

from shared.errors import AgentError
from shared.structured import StructuredModel, fingerprint, request

from .config import Settings
from .embeddings import embed_checked, preflight, prepare
from .models import Answer, VectorData
from .search import ExactIndex, SearchIndex


def run(
    built: Build,
    query: Query,
    settings: Settings,
    embedder: Embedder,
    *,
    model: StructuredModel | None = None,
    retrieve_only: bool = False,
    live: bool = False,
    index_factory: Callable[[VectorData], SearchIndex] = ExactIndex,
) -> Answer:
    start = perf_counter()
    if built.manifest.corpus_sha256 != fingerprint(built.corpus.model_dump(mode="json")):
        raise AgentError("Corpus no longer matches its chunk manifest.")
    pack_sources(query, [], settings.max_context_chars)
    preflight(built.corpus, [query.question], settings)
    data, usage = prepare(built.corpus, embedder, settings)
    index = index_factory(data)
    indexed = perf_counter()
    query_vectors, query_usage = embed_checked([query.question], embedder, settings)
    usage.logical_calls += query_usage.logical_calls
    usage.tokens += query_usage.tokens
    embedded = perf_counter()
    matches = index.search(query_vectors[0], settings.top_k, settings.min_similarity)
    lookup = {source.id: source for source in sources(built.corpus)}
    ids = [match.source_id for match in matches]
    if len(ids) != len(set(ids)) or len(ids) > settings.top_k or not set(ids) <= lookup.keys():
        raise AgentError("Index returned invalid passage IDs or exceeded top-k.")
    if any(match.similarity < settings.min_similarity for match in matches):
        raise AgentError("Index returned a result below the requested threshold.")
    context = pack_sources(query, [lookup[id_] for id_ in ids], settings.max_context_chars)
    packed = perf_counter()
    reply = None
    generation_calls = 0
    if not retrieve_only and context.sources:
        if model is None:
            raise AgentError("Generation requires a model; request retrieval-only explicitly.")
        reply = request(
            model, SYSTEM_PROMPT, payload(query, context.sources), Reply, settings.max_context_chars
        )
        validate_reply(reply, context)
        generation_calls = 1
    status: Literal["retrieved", "answered", "abstained"] = (
        "retrieved"
        if retrieve_only
        else "answered"
        if reply is not None and reply.claims
        else "abstained"
    )
    reason: Literal[
        "retrieval_only", "no_matches", "context_budget", "answered", "insufficient_evidence"
    ] = (
        "retrieval_only"
        if retrieve_only
        else "no_matches"
        if not matches
        else "context_budget"
        if not context.sources
        else "answered"
        if reply is not None and reply.claims
        else "insufficient_evidence"
    )
    return Answer(
        status=status,
        reason=reason,
        backend=index.name,
        mode="live" if live else "injected_provider",
        manifest=built.manifest,
        index_sha256=data.sha256,
        embedding_model=settings.embedding_model,
        dimensions=settings.dimensions,
        top_k=settings.top_k,
        min_similarity=settings.min_similarity,
        matches=matches,
        context=context,
        reply=reply,
        embedding_usage=usage,
        logical_generation_calls=generation_calls,
        generation_model=settings.model if generation_calls else None,
        prompt_sha256=fingerprint(SYSTEM_PROMPT),
        timing_ms={
            "document_embedding_and_index": (indexed - start) * 1000,
            "query_embedding": (embedded - indexed) * 1000,
            "search_and_pack": (packed - embedded) * 1000,
            "generation_and_validation": (perf_counter() - packed) * 1000,
        },
    )
