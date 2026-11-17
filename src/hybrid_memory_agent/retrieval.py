from dataclasses import dataclass

from episodic_memory_agent.models import Query as EpisodeQuery
from episodic_memory_agent.storage import EpisodeStore
from long_term_memory_agent.models import Scope, Selection
from long_term_memory_agent.storage import FactStore, MemoryConflict
from profile_agent.storage import ProfileStore
from semantic_memory_agent.embeddings import Embedder
from semantic_memory_agent.models import Query as SemanticQuery
from semantic_memory_agent.retrieval import candidates
from semantic_memory_agent.retrieval import retrieve as semantic_retrieve

from shared.errors import AgentError

from .config import Settings
from .models import Context, Query, RankedFact, Retrieval
from .ranking import fuse, lexical_rank


@dataclass(frozen=True)
class Stores:
    facts: FactStore
    episodes: EpisodeStore
    profiles: ProfileStore


def retrieve(
    query: Query, stores: Stores, embedder: Embedder | None, settings: Settings
) -> Retrieval:
    scope = Scope(profile=query.profile)
    facts = candidates(stores.facts, scope)
    lexical = lexical_rank(query.question, facts, settings.top_k)
    semantic: list[str] = []
    tokens = 0
    if query.mode == "hybrid" and facts:
        if embedder is None:
            raise AgentError("Hybrid retrieval requires an embedding provider.")
        search = semantic_retrieve(
            SemanticQuery(profile=query.profile, question=query.question),
            stores.facts,
            embedder,
            settings,
        )
        semantic = [match.fact.key for match in search.matches]
        tokens = search.embedding_tokens
        expected = {fact.key: fact.revision for fact in facts}
        if any(expected.get(m.fact.key) != m.fact.revision for m in search.matches):
            raise MemoryConflict("Semantic results do not match the lexical snapshot.")
        current = candidates(stores.facts, scope)
        if [(f.key, f.revision) for f in facts] != [(f.key, f.revision) for f in current]:
            raise MemoryConflict("Facts changed between lexical and semantic retrieval.")
    by_key = {fact.key: fact for fact in facts}
    ranked = [
        RankedFact(
            fact=by_key[item.key],
            score=item.score,
            lexical_rank=item.lexical_rank,
            semantic_rank=item.semantic_rank,
        )
        for item in fuse(lexical, semantic, settings.rrf_k)[: settings.fact_limit]
    ]
    return Retrieval(
        facts=ranked,
        episodes=stores.episodes.search(
            EpisodeQuery(profile=query.profile, question=query.question), settings.episode_limit
        ),
        preferences=stores.profiles.get(query.profile),
        mode=query.mode,
        embedding_tokens=tokens,
    )


def recheck(query: Query, memory: Retrieval, context: Context, stores: Stores) -> None:
    """Recheck supplied records only; these reads are not a cross-database transaction."""
    fact_sources = [s for s in context.sources if s.kind == "asserted_fact"]
    if fact_sources:
        current = stores.facts.select(
            Selection(profile=query.profile, keys=[source.key for source in fact_sources])
        )
        if {f.key: f.revision for f in current} != {s.key: s.revision for s in fact_sources}:
            raise MemoryConflict("Supplied facts changed; discard the hybrid result.")
    episode_keys = {s.key for s in context.sources if s.kind == "historical_episode"}
    stores.episodes.recheck([e for e in memory.episodes if e.episode.id in episode_keys])
    if stores.profiles.get(query.profile).revision != context.preferences.revision:
        raise MemoryConflict("Approved preferences changed; discard the hybrid result.")
