from typing import Any

from shared.errors import AgentError
from shared.structured import canonical, fingerprint

from .models import Context, Hit, Query, Source
from .prompts import SYSTEM_PROMPT


def payload(query: Query, selected: list[Source]) -> dict[str, Any]:
    return {
        "question": query.question,
        "sources": [source.model_dump(mode="json") for source in selected],
    }


def pack(query: Query, hits: list[Hit], max_chars: int) -> Context:
    return pack_sources(query, [hit.source for hit in hits], max_chars)


def pack_sources(query: Query, sources: list[Source], max_chars: int) -> Context:
    def size(selected: list[Source]) -> int:
        return len(SYSTEM_PROMPT) + len(canonical(payload(query, selected)))

    if size([]) > max_chars:
        raise AgentError("Question and instructions exceed the context budget.")
    selected: list[Source] = []
    dropped = []
    for source in sources:
        if size([*selected, source]) <= max_chars:
            selected.append(source)
        else:
            dropped.append(source.id)
    return Context(
        sources=selected,
        dropped_ids=dropped,
        input_chars=size(selected),
        payload_sha256=fingerprint(payload(query, selected)),
    )
