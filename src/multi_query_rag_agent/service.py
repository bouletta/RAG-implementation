import logging
from time import perf_counter

from basic_rag_agent.models import Query
from chroma_rag_agent.service import answer
from chroma_rag_agent.store import Store
from embedding_rag_agent.embeddings import embed_checked
from embedding_rag_agent.models import Usage
from hybrid_rag_agent.retrieval import retrieve_hybrid
from semantic_memory_agent.embeddings import Embedder

from shared.errors import AgentError
from shared.structured import StructuredModel

from .config import Settings
from .expansion import expand, preflight
from .models import Branch, Result
from .retrieval import fuse_queries

logger = logging.getLogger(__name__)


def run(
    store: Store,
    name: str,
    query: Query,
    settings: Settings,
    embedder: Embedder,
    *,
    expansion_model: StructuredModel | None = None,
    answer_model: StructuredModel | None = None,
    provided: list[str] | None = None,
    retrieve_only: bool = False,
    live: bool = False,
) -> Result:
    started = perf_counter()
    snapshot = store.load(name, settings)
    generated = provided is None and settings.max_variants > 0
    if not retrieve_only and answer_model is None:
        raise AgentError("Answer generation requires a model; use retrieval-only explicitly.")
    count = 1 + (settings.max_variants if generated else len(provided or []))
    preflight(query, settings, generated=generated, retrieve_only=retrieve_only, query_count=count)
    plan = expand(query, settings, expansion_model, provided=provided)
    vectors: list[tuple[float, ...]] = []
    usage = Usage()
    for start in range(0, len(plan.queries), settings.embedding_batch_size):
        rows, batch = embed_checked(
            plan.queries[start : start + settings.embedding_batch_size], embedder, settings
        )
        vectors.extend(rows)
        usage.logical_calls += batch.logical_calls
        usage.tokens += batch.tokens
    branch_settings = settings.model_copy(update={"top_k": settings.candidate_window})
    branches = []
    for i, (question, vector) in enumerate(zip(plan.queries, vectors, strict=True)):
        lexical, semantic, ranking = retrieve_hybrid(
            store, snapshot, Query(question=question), vector, branch_settings
        )
        branches.append(
            Branch(query_index=i, lexical_ids=lexical, semantic_matches=semantic, ranking=ranking)
        )
    fused = fuse_queries(
        [[row.source_id for row in branch.ranking] for branch in branches],
        settings.rrf_k,
        settings.top_k,
    )
    logger.info("multi_query_retrieval_completed")
    result = answer(
        snapshot,
        query,
        [row.source_id for row in fused],
        settings,
        matches=branches[0].semantic_matches,
        usage=usage,
        method="multi_query_hybrid_rrf_v1",
        model=answer_model,
        retrieve_only=retrieve_only,
        started=started,
        live=live,
    )
    return Result(
        answer=result,
        expansion=plan,
        branches=branches,
        fusion=fused,
        logical_model_calls=plan.logical_model_calls + result.logical_generation_calls,
        candidate_window=settings.candidate_window,
        rrf_k=settings.rrf_k,
    )
