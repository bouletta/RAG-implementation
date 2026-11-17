from time import perf_counter

from basic_rag_agent.context import pack_sources
from basic_rag_agent.models import Query
from chroma_rag_agent.service import answer
from chroma_rag_agent.store import Store
from embedding_rag_agent.embeddings import embed_checked
from semantic_memory_agent.embeddings import Embedder

from shared.structured import StructuredModel

from .config import Settings
from .models import HybridAnswer
from .retrieval import retrieve_hybrid


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
) -> HybridAnswer:
    started = perf_counter()
    snapshot = store.load(name, settings)
    pack_sources(query, [], settings.max_context_chars)
    vectors, usage = embed_checked([query.question], embedder, settings)
    lexical, semantic, fused = retrieve_hybrid(store, snapshot, query, vectors[0], settings)
    result = answer(
        snapshot,
        query,
        [row.source_id for row in fused],
        settings,
        matches=semantic,
        usage=usage,
        method="rrf_unique_terms_chroma_v1",
        model=model,
        retrieve_only=retrieve_only,
        started=started,
        live=live,
    )
    return HybridAnswer(
        answer=result,
        lexical_ids=lexical,
        semantic_ids=[row.source_id for row in semantic],
        fusion=fused,
        candidate_window=settings.candidate_window,
        rrf_k=settings.rrf_k,
    )
