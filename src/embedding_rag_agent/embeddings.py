import logging
import math

from basic_rag_agent.models import Corpus
from semantic_memory_agent.embeddings import Embedder, normalize

from shared.errors import AgentError

from .config import Settings
from .models import Usage, VectorData

logger = logging.getLogger(__name__)


def validate_inputs(texts: list[str]) -> None:
    if any(not text.strip() or len(text.encode("utf-8")) > 8000 for text in texts):
        raise AgentError("Embedding text must be nonblank and at most 8000 UTF-8 bytes.")


def preflight(corpus: Corpus, questions: list[str], settings: Settings) -> None:
    validate_inputs([p.text for p in corpus.passages] + questions)
    calls = math.ceil(len(corpus.passages) / settings.embedding_batch_size) + len(questions)
    if calls > settings.max_embedding_calls:
        raise AgentError("Embedding call budget exceeded before any provider requests.")


def embed_checked(
    texts: list[str], embedder: Embedder, settings: Settings
) -> tuple[list[tuple[float, ...]], Usage]:
    if not 1 <= len(texts) <= settings.embedding_batch_size:
        raise AgentError("Embedding batch exceeds the configured input count.")
    validate_inputs(texts)
    logger.info("embedding_batch_started")
    result = embedder.embed(texts)
    if len(result.vectors) != len(texts) or result.tokens < 0:
        raise AgentError("Embedding response has invalid row count or usage.")
    vectors = [tuple(normalize(row, settings.dimensions)) for row in result.vectors]
    logger.info("embedding_batch_validated")
    return vectors, Usage(
        logical_calls=1,
        tokens=result.tokens,
    )


def prepare(corpus: Corpus, embedder: Embedder, settings: Settings) -> tuple[VectorData, Usage]:
    preflight(corpus, [], settings)
    vectors: list[tuple[float, ...]] = []
    usage = Usage()
    texts = [passage.text for passage in corpus.passages]
    for start in range(0, len(texts), settings.embedding_batch_size):
        rows, batch = embed_checked(
            texts[start : start + settings.embedding_batch_size], embedder, settings
        )
        vectors.extend(rows)
        usage.logical_calls += batch.logical_calls
        usage.tokens += batch.tokens
    return VectorData(
        corpus=corpus,
        embedding_model=settings.embedding_model,
        dimensions=settings.dimensions,
        vectors=tuple(vectors),
    ), usage
