import logging

from basic_rag_agent.context import pack_sources
from basic_rag_agent.models import Query, Source
from basic_rag_agent.retrieval import terms

from shared.errors import AgentError
from shared.structured import StructuredModel, canonical, fingerprint, request

from .config import Settings
from .models import Compression, Selection
from .prompts import COMPRESSION_PROMPT
from .units import segment

logger = logging.getLogger(__name__)


def compress(
    query: Query,
    parents: list[Source],
    settings: Settings,
    model: StructuredModel | None = None,
) -> Compression:
    baseline = pack_sources(query, parents, settings.max_context_chars)
    units = segment(parents, settings)
    calls = 0
    input_chars = 0
    if settings.compression_mode == "llm" and units:
        if model is None:
            raise AgentError("LLM compression requires a selection model.")
        payload = {
            "question": query.question,
            "max_selected_units": settings.max_selected_units,
            "units": [unit.model_dump(mode="json") for unit in units],
        }
        input_chars = len(COMPRESSION_PROMPT) + len(canonical(payload))
        selected = request(
            model, COMPRESSION_PROMPT, payload, Selection, settings.compression_context_chars
        ).unit_ids
        calls = 1
    else:
        query_terms = terms(query.question)
        scored = [(unit.source.id, len(query_terms & terms(unit.source.text))) for unit in units]
        selected = [
            id_ for id_, score in sorted(scored, key=lambda row: (-row[1], row[0])) if score > 0
        ][: settings.max_selected_units]
    if len(selected) > settings.max_selected_units or len(set(selected)) != len(selected):
        raise AgentError("Compression selection has duplicate IDs or exceeds the selection budget.")
    known = {unit.source.id for unit in units}
    if not set(selected) <= known:
        raise AgentError("Compression selected an unknown excerpt ID.")
    # Restore retrieval/source order rather than letting the selector rearrange evidence.
    selected_units = [unit for unit in units if unit.source.id in selected]
    context = pack_sources(
        query, [unit.source for unit in selected_units], settings.max_context_chars
    )
    retrieved_chars = sum(len(parent.text) for parent in parents)
    packed_chars = sum(len(source.text) for source in context.sources)
    logger.info("context_compression_validated")
    return Compression(
        mode=settings.compression_mode,
        context=context,
        units=units,
        selected_ids=[unit.source.id for unit in selected_units],
        omitted_ids=[unit.source.id for unit in units if unit.source.id not in selected],
        logical_model_calls=calls,
        model=settings.model if calls else None,
        prompt_sha256=fingerprint(COMPRESSION_PROMPT) if calls else None,
        selector_input_chars=input_chars,
        baseline_input_chars=baseline.input_chars,
        retrieved_text_chars=retrieved_chars,
        selected_text_chars=sum(len(unit.source.text) for unit in selected_units),
        packed_text_chars=packed_chars,
        text_reduction_fraction=1 - packed_chars / retrieved_chars if retrieved_chars else None,
        packed_input_char_delta=context.input_chars - baseline.input_chars,
    )
