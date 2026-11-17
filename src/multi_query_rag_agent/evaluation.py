"""Compare original-query and expanded retrieval without generating answers."""

import argparse
import math
from pathlib import Path

from basic_rag_agent.context import pack_sources
from basic_rag_agent.corpus import sources
from basic_rag_agent.llm import LazyModel
from basic_rag_agent.models import Query
from chroma_rag_agent.llm import LazyEmbedder
from chroma_rag_agent.store import Store
from chunking_agent.evaluation import Case
from embedding_rag_agent.models import Usage
from hybrid_rag_agent.evaluation import Score
from pydantic import BaseModel, TypeAdapter
from semantic_memory_agent.embeddings import Embedder

from shared.cli import execute
from shared.errors import AgentError
from shared.files import read_limited
from shared.structured import StructuredModel, fingerprint

from .config import Settings
from .expansion import preflight
from .models import Result
from .service import run


class Comparison(BaseModel):
    mode: str
    snapshot: str
    corpus_sha256: str
    cases_sha256: str
    settings: dict[str, int | float]
    embedding_usage: Usage
    logical_expansion_calls: int
    scores: list[Score]
    summaries: dict[str, dict[str, float | None]]
    runs: list[Result]


def evaluate(
    store: Store,
    name: str,
    cases: list[Case],
    settings: Settings,
    embedder: Embedder,
    expansion_model: StructuredModel | None,
    *,
    live: bool = False,
) -> Comparison:
    if not 1 <= len(cases) <= 32 or len({c.id for c in cases}) != len(cases):
        raise AgentError("Evaluation requires 1–32 cases with unique IDs.")
    generated = settings.max_variants > 0
    required_models = len(cases) * int(generated)
    required_embeddings = len(cases) * math.ceil(
        (1 + settings.max_variants) / settings.embedding_batch_size
    )
    if required_models > settings.max_model_calls:
        raise AgentError("Evaluation model call budget exceeded before any requests.")
    if required_embeddings > settings.max_embedding_calls:
        raise AgentError("Evaluation embedding call budget exceeded before any requests.")
    snapshot = store.load(name, settings)
    lookup = {s.id: s for s in sources(snapshot.build.corpus)}
    targets = {}
    for case in cases:
        preflight(
            case,
            settings,
            generated=generated,
            retrieve_only=True,
            query_count=1 + settings.max_variants,
        )
        targets[case.id] = {
            p.id
            for p in snapshot.build.corpus.passages
            if case.evidence is not None
            and p.title == case.evidence.document_name
            and case.evidence.quote in p.text
        }
        if case.evidence is not None and (not case.evidence.quote.strip() or not targets[case.id]):
            raise AgentError("Gold evidence is blank, missing, or split across stored passages.")
    scores = []
    runs = []
    usage = Usage()
    for case in cases:
        query = Query(question=case.question)
        result = run(
            store,
            name,
            query,
            settings,
            embedder,
            expansion_model=expansion_model,
            retrieve_only=True,
            live=live,
        )
        if result.answer.corpus_sha256 != snapshot.build.manifest.corpus_sha256:
            raise AgentError("Snapshot changed during evaluation.")
        runs.append(result)
        usage.logical_calls += result.answer.embedding_usage.logical_calls
        usage.tokens += result.answer.embedding_usage.tokens
        rankings = {
            "single_query": [row.source_id for row in result.branches[0].ranking][: settings.top_k],
            "multi_query": [row.source_id for row in result.fusion],
        }
        for method, ids in rankings.items():
            context = pack_sources(query, [lookup[id_] for id_ in ids], settings.max_context_chars)
            ranks = [rank for rank, id_ in enumerate(ids, 1) if id_ in targets[case.id]]
            positive = case.evidence is not None
            scores.append(
                Score(
                    case_id=case.id,
                    method=method,
                    hit_at_k=bool(ranks) if positive else None,
                    reciprocal_rank=(1 / min(ranks) if ranks else 0.0) if positive else None,
                    intact_in_context=bool({s.id for s in context.sources} & targets[case.id])
                    if positive
                    else None,
                    false_positive=bool(ids) if not positive else None,
                )
            )
    summaries: dict[str, dict[str, float | None]] = {}
    for method in ("single_query", "multi_query"):
        summaries[method] = {}
        for metric in ("hit_at_k", "reciprocal_rank", "intact_in_context", "false_positive"):
            values = [
                float(value)
                for score in scores
                if score.method == method and (value := getattr(score, metric)) is not None
            ]
            summaries[method][metric] = sum(values) / len(values) if values else None
    return Comparison(
        mode="live" if live else "injected_provider",
        snapshot=name,
        corpus_sha256=snapshot.build.manifest.corpus_sha256,
        cases_sha256=fingerprint([c.model_dump(mode="json") for c in cases]),
        settings={
            key: getattr(settings, key)
            for key in (
                "max_variants",
                "top_k",
                "candidate_window",
                "rrf_k",
                "min_similarity",
                "min_matched_terms",
                "max_context_chars",
                "max_model_calls",
                "max_embedding_calls",
            )
        },
        embedding_usage=usage,
        logical_expansion_calls=sum(r.expansion.logical_model_calls for r in runs),
        scores=scores,
        summaries=summaries,
        runs=runs,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired single-/multi-query retrieval evaluation.")
    parser.add_argument("name")
    parser.add_argument("cases", type=Path)
    parser.add_argument("--store", type=Path)
    parser.add_argument(
        "--live",
        action="store_true",
        required=True,
        help="Allow paid query generation and embeddings; no final answers.",
    )
    args = parser.parse_args()

    def action() -> Comparison:
        settings = Settings()
        cases = TypeAdapter(list[Case]).validate_json(read_limited(args.cases, 100_000))
        store = Store(args.store or settings.store_path)
        embedding = LazyEmbedder(settings)
        expansion = LazyModel(
            settings.model_copy(
                update={
                    "max_output_tokens": settings.expansion_output_tokens,
                }
            )
        )
        try:
            return evaluate(store, args.name, cases, settings, embedding, expansion, live=True)
        finally:
            embedding.close()
            expansion.close()

    return execute(action)


if __name__ == "__main__":
    raise SystemExit(main())
