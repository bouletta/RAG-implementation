from itertools import zip_longest
from typing import Any

from episodic_memory_agent.storage import document

from shared.errors import AgentError
from shared.structured import canonical, fingerprint

from .config import Settings
from .models import Context, Query, Retrieval, Source
from .prompts import SYSTEM_PROMPT


def sources(query: Query, memory: Retrieval) -> tuple[list[Source], list[Source]]:
    facts = [
        Source(
            id="f_" + fingerprint([query.profile, row.fact.key, row.fact.revision])[:32],
            kind="asserted_fact",
            key=row.fact.key,
            revision=row.fact.revision,
            recorded_at=row.fact.updated_at.isoformat(),
            text=row.fact.value,
        )
        for row in memory.facts
    ]
    episodes = [
        Source(
            id="e_" + fingerprint([query.profile, row.episode.id, row.revision])[:32],
            kind="historical_episode",
            key=row.episode.id,
            revision=row.revision,
            recorded_at=row.episode.happened_at.isoformat(),
            text=document(row.episode),
        )
        for row in memory.episodes
    ]
    return facts, episodes


def payload(query: Query, context: Context) -> dict[str, Any]:
    return {
        "question": query.question,
        "approved_preferences": context.preferences.preferences.model_dump(exclude_none=True),
        "sources": [source.model_dump() for source in context.sources],
    }


def assemble(query: Query, memory: Retrieval, settings: Settings) -> Context:
    context = Context(
        sources=[],
        preferences=memory.preferences,
        dropped_source_ids=[],
        input_chars=0,
        payload_sha256="",
    )

    def size() -> int:
        return len(SYSTEM_PROMPT) + len(canonical(payload(query, context)))

    if size() > settings.max_context_chars:
        raise AgentError("Question and preferences exceed the context budget.")
    # Interleave types without pretending their relevance scores are comparable.
    facts, episodes = sources(query, memory)
    for pair in zip_longest(facts, episodes):
        for source in pair:
            if source is None:
                continue
            context.sources.append(source)
            if size() > settings.max_context_chars:
                context.sources.pop()
                context.dropped_source_ids.append(source.id)
    if len({source.id for source in context.sources}) != len(context.sources):
        raise AgentError("Source identity collision; no answer generated.")
    context.input_chars = size()
    context.payload_sha256 = fingerprint(payload(query, context))
    return context
