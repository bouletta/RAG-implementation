from basic_rag_agent.models import Query
from basic_rag_agent.retrieval import retrieve
from chroma_rag_agent.models import Snapshot
from chroma_rag_agent.store import Store
from embedding_rag_agent.models import Match
from hybrid_memory_agent.ranking import fuse

from .config import Settings
from .models import FusedMatch


def retrieve_hybrid(
    store: Store,
    snapshot: Snapshot,
    query: Query,
    vector: tuple[float, ...],
    settings: Settings,
) -> tuple[list[str], list[Match], list[FusedMatch]]:
    lexical_settings = settings.model_copy(update={"top_k": settings.candidate_window})
    lexical = [hit.source.id for hit in retrieve(snapshot.build.corpus, query, lexical_settings)]
    semantic = store.search(snapshot, vector, settings.candidate_window, settings.min_similarity)
    fused = fuse(lexical, [match.source_id for match in semantic], settings.rrf_k)
    return (
        lexical,
        semantic,
        [
            FusedMatch(
                source_id=row.key,
                rrf_score=row.score,
                lexical_rank=row.lexical_rank,
                semantic_rank=row.semantic_rank,
            )
            for row in fused[: settings.top_k]
        ],
    )
