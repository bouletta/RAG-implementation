"""Evaluate lexical and embedding retrieval against the same source evidence."""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from basic_rag_agent.context import pack_sources
from basic_rag_agent.corpus import sources
from basic_rag_agent.models import Query
from basic_rag_agent.retrieval import retrieve
from chunking_agent.chunking import build
from chunking_agent.evaluation import Case, CaseScore, gold_spans, intact, rate
from chunking_agent.ingestion import load_documents
from chunking_agent.models import Document, Policy
from pydantic import BaseModel, TypeAdapter
from semantic_memory_agent.embeddings import Embedder, OpenAIEmbedder

from shared.cli import execute
from shared.errors import AgentError
from shared.files import read_limited
from shared.structured import fingerprint

from .config import Settings
from .embeddings import embed_checked, preflight, prepare
from .models import Usage, VectorData
from .search import ExactIndex, SearchIndex


class Summary(BaseModel):
    backend: str
    cases: list[CaseScore]
    intact_retrieval_rate: float | None
    intact_context_rate: float | None
    false_positive_rate: float | None
    packed_false_positive_rate: float | None


class Evaluation(BaseModel):
    mode: Literal["lexical_only", "live", "injected_provider"]
    corpus_sha256: str
    cases_sha256: str
    index_sha256: str | None
    embedding_model: str | None
    dimensions: int | None
    top_k: int
    min_matched_terms: int
    chunk_policy: Policy
    embedding_batch_size: int
    max_embedding_calls: int
    min_similarity: float
    max_context_chars: int
    embedding_usage: Usage
    summaries: list[Summary]


def evaluate(
    documents: tuple[Document, ...],
    cases: list[Case],
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    live: bool = False,
    index_factory: Callable[[VectorData], SearchIndex] = ExactIndex,
) -> Evaluation:
    targets = gold_spans(documents, cases)
    built = build(documents, settings.policy, settings.max_chunks)
    for case in cases:
        pack_sources(Query(question=case.question), [], settings.max_context_chars)
    if live and embedder is None:
        raise AgentError("Live evaluation requires a provider.")
    index = None
    data = None
    usage = Usage()
    if embedder is not None:
        preflight(built.corpus, [case.question for case in cases], settings)
        data, usage = prepare(built.corpus, embedder, settings)
        index = index_factory(data)
    by_chunk = {c.id: c for c in built.manifest.chunks}
    by_source = {s.id: s for s in sources(built.corpus)}
    rows: dict[str, list[CaseScore]] = {"lexical": []}
    if index is not None:
        rows[index.name] = []
    for case in cases:
        query = Query(question=case.question)
        rankings = {"lexical": [hit.source.id for hit in retrieve(built.corpus, query, settings)]}
        if index is not None and embedder is not None:
            vectors, batch = embed_checked([case.question], embedder, settings)
            usage.tokens += batch.tokens
            usage.logical_calls += batch.logical_calls
            rankings[index.name] = [
                m.source_id
                for m in index.search(
                    vectors[0],
                    settings.top_k,
                    settings.min_similarity,
                )
            ]
        target = targets[case.id]
        for name, ids in rankings.items():
            if (
                len(ids) != len(set(ids))
                or len(ids) > settings.top_k
                or not set(ids) <= by_source.keys()
            ):
                raise AgentError("Evaluation index returned invalid source IDs.")
            context = pack_sources(
                query, [by_source[id_] for id_ in ids], settings.max_context_chars
            )
            packed = [source.id for source in context.sources]
            rows[name].append(
                CaseScore(
                    case_id=case.id,
                    intact_in_corpus=intact(list(by_chunk.values()), target),
                    intact_in_retrieval=intact([by_chunk[id_] for id_ in ids], target),
                    intact_in_context=intact([by_chunk[id_] for id_ in packed], target),
                    false_positive=bool(ids) if target is None else None,
                    packed_false_positive=bool(packed) if target is None else None,
                )
            )
    return Evaluation(
        mode="live" if live else "injected_provider" if embedder is not None else "lexical_only",
        corpus_sha256=built.manifest.corpus_sha256,
        cases_sha256=fingerprint([case.model_dump(mode="json") for case in cases]),
        index_sha256=data.sha256 if data is not None else None,
        embedding_model=settings.embedding_model if embedder is not None else None,
        dimensions=settings.dimensions if embedder is not None else None,
        top_k=settings.top_k,
        min_matched_terms=settings.min_matched_terms,
        chunk_policy=settings.policy,
        embedding_batch_size=settings.embedding_batch_size,
        max_embedding_calls=settings.max_embedding_calls,
        min_similarity=settings.min_similarity,
        max_context_chars=settings.max_context_chars,
        embedding_usage=usage,
        summaries=[
            Summary(
                backend=name,
                cases=values,
                intact_retrieval_rate=rate([row.intact_in_retrieval for row in values]),
                intact_context_rate=rate([row.intact_in_context for row in values]),
                false_positive_rate=rate([row.false_positive for row in values]),
                packed_false_positive_rate=rate([row.packed_false_positive for row in values]),
            )
            for name, values in rows.items()
        ],
    )


def main(
    settings_type: type[Settings] = Settings,
    index_factory: Callable[[VectorData], SearchIndex] = ExactIndex,
    check_backend: Callable[[], None] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument(
        "--live", action="store_true", help="Send passages and queries to the embedding provider."
    )
    args = parser.parse_args()

    def action() -> Evaluation:
        settings = settings_type()
        cases = TypeAdapter(list[Case]).validate_json(read_limited(args.cases, 200_000))
        documents = load_documents(args.files, settings)
        # Validate labels before constructing a provider; no payload is sent here.
        gold_spans(documents, cases)
        if args.live and check_backend is not None:
            check_backend()
        provider = OpenAIEmbedder(settings.transport_settings()) if args.live else None
        try:
            return evaluate(
                documents,
                cases,
                settings,
                embedder=provider,
                live=args.live,
                index_factory=index_factory,
            )
        finally:
            if provider is not None:
                provider.close()

    return execute(action)


if __name__ == "__main__":
    raise SystemExit(main())
