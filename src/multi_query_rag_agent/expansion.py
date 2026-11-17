import math

from basic_rag_agent.context import pack_sources
from basic_rag_agent.models import Query
from embedding_rag_agent.embeddings import validate_inputs

from shared.errors import AgentError
from shared.structured import StructuredModel, canonical, fingerprint, request

from .config import Settings
from .models import Expansion, Variants
from .prompts import EXPANSION_PROMPT


def preflight(
    query: Query, settings: Settings, *, generated: bool, retrieve_only: bool, query_count: int
) -> None:
    required_models = int(generated) + int(not retrieve_only)
    if required_models > settings.max_model_calls:
        raise AgentError("Model call budget exceeded before any provider requests.")
    if math.ceil(query_count / settings.embedding_batch_size) > settings.max_embedding_calls:
        raise AgentError("Embedding call budget exceeded before any provider requests.")
    validate_inputs([query.question])
    pack_sources(query, [], settings.max_context_chars)
    if generated:
        content = {"question": query.question, "max_variants": settings.max_variants}
        if len(EXPANSION_PROMPT) + len(canonical(content)) > settings.expansion_context_chars:
            raise AgentError("Query expansion input exceeds its context budget.")


def expand(
    query: Query,
    settings: Settings,
    model: StructuredModel | None,
    *,
    provided: list[str] | None = None,
) -> Expansion:
    generated = provided is None and settings.max_variants > 0
    if generated:
        if model is None:
            raise AgentError("Generated expansion requires a model.")
        variants = request(
            model,
            EXPANSION_PROMPT,
            {"question": query.question, "max_variants": settings.max_variants},
            Variants,
            settings.expansion_context_chars,
        ).queries
    else:
        variants = provided if provided is not None else []
    if len(variants) > settings.max_variants:
        raise AgentError("Query expansion exceeded the configured variant limit.")
    unique = [query.question]
    seen = {" ".join(query.question.split()).casefold()}
    duplicates = 0
    for variant in variants:
        value = Query(question=variant).question.strip()
        key = " ".join(value.split()).casefold()
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(value)
    validate_inputs(unique)
    return Expansion(
        mode="generated" if generated else "provided" if provided is not None else "disabled",
        queries=unique,
        discarded_duplicates=duplicates,
        logical_model_calls=int(generated),
        model=settings.model if generated else None,
        prompt_sha256=fingerprint(EXPANSION_PROMPT) if generated else None,
    )
