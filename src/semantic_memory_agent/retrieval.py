"""Exact cosine search over at most 100 active facts; no persistent vector cache."""

from long_term_memory_agent.models import Fact, Scope
from long_term_memory_agent.storage import FactStore, MemoryConflict, active

from shared.errors import AgentError

from .config import Settings
from .embeddings import Embedder, normalize
from .models import Match, Query, SearchResult


def candidates(store: FactStore, scope: Scope) -> list[Fact]:
    now = store.clock()
    return [fact for fact in store.list_facts(scope) if active(fact, now)]


def retrieve(
    query: Query, store: FactStore, embedder: Embedder, settings: Settings
) -> SearchResult:
    facts = candidates(store, query)
    if not facts:
        return SearchResult(
            matches=[],
            embedding_model=settings.embedding_model,
            dimensions=settings.dimensions,
            embedding_tokens=0,
        )
    embedded = embedder.embed([query.question, *[f"{f.key}: {f.value}" for f in facts]])
    if len(embedded.vectors) != len(facts) + 1:
        raise AgentError("Embedding count does not match the selected facts.")
    vectors = [normalize(vector, settings.dimensions) for vector in embedded.vectors]
    current = candidates(store, query)
    if [(f.key, f.revision) for f in facts] != [(f.key, f.revision) for f in current]:
        raise MemoryConflict("Memory changed during retrieval; retry with a fresh snapshot.")
    matches = []
    for fact, vector in zip(facts, vectors[1:], strict=True):
        score = max(-1.0, min(1.0, sum(a * b for a, b in zip(vectors[0], vector, strict=True))))
        if score >= settings.min_similarity:
            matches.append(Match(fact=fact, similarity=score))
    matches.sort(key=lambda item: (-item.similarity, item.fact.key))
    return SearchResult(
        matches=matches[: settings.top_k],
        embedding_model=settings.embedding_model,
        dimensions=settings.dimensions,
        embedding_tokens=embedded.tokens,
    )
