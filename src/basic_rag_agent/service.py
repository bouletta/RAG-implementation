import logging
from time import perf_counter

from shared.errors import AgentError
from shared.structured import StructuredModel, fingerprint, request

from .config import Settings
from .context import pack, payload
from .models import Corpus, Query, Ranking, Reply, Result
from .prompts import SYSTEM_PROMPT
from .retrieval import retrieve
from .validation import validate_reply

logger = logging.getLogger(__name__)


def run(
    corpus: Corpus,
    query: Query,
    settings: Settings,
    *,
    model: StructuredModel | None = None,
    preview: bool = False,
) -> Result:
    start = perf_counter()
    hits = retrieve(corpus, query, settings)
    retrieved = perf_counter()
    context = pack(query, hits, settings.max_context_chars)
    packed = perf_counter()
    result = Result(
        status="preview" if preview else "abstained",
        reason="preview" if preview else "no_matches" if not hits else "context_budget",
        corpus_sha256=fingerprint(corpus.model_dump(mode="json")),
        prompt_sha256=fingerprint(SYSTEM_PROMPT),
        generation_model=None,
        top_k=settings.top_k,
        min_matched_terms=settings.min_matched_terms,
        context=context,
        ranking=[Ranking(source_id=h.source.id, matched_terms=h.matched_terms) for h in hits],
        reply=None,
        logical_model_calls=0,
        timing_ms={
            "retrieval": (retrieved - start) * 1000,
            "packing": (packed - retrieved) * 1000,
        },
    )
    logger.info("rag_retrieval_completed")
    if not preview and context.sources:
        if model is None:
            raise AgentError("Answer generation requires a model; use preview for local retrieval.")
        generated = perf_counter()
        reply = request(
            model, SYSTEM_PROMPT, payload(query, context.sources), Reply, settings.max_context_chars
        )
        validate_reply(reply, context)
        result.timing_ms["generation_and_validation"] = (perf_counter() - generated) * 1000
        result.status = "answered" if reply.claims else "abstained"
        result.reason = "answered" if reply.claims else "insufficient_evidence"
        result.reply = reply
        result.logical_model_calls = 1
        result.generation_model = settings.model
        logger.info("rag_answer_validated")
    result.timing_ms["pipeline_total"] = (perf_counter() - start) * 1000
    return result
