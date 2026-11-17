"""Paired evidence-retention evaluation; compression is not judged by size alone."""

import argparse
import math
from pathlib import Path
from statistics import mean

from basic_rag_agent.context import pack_sources
from basic_rag_agent.llm import LazyModel
from basic_rag_agent.models import Query, Source
from chroma_rag_agent.llm import LazyEmbedder
from chroma_rag_agent.store import Store
from chunking_agent.evaluation import Case, Gold
from embedding_rag_agent.models import Usage
from multi_query_rag_agent.expansion import preflight
from pydantic import BaseModel, TypeAdapter
from semantic_memory_agent.embeddings import Embedder

from shared.cli import execute
from shared.errors import AgentError
from shared.files import read_limited
from shared.structured import StructuredModel, fingerprint

from .config import Settings
from .models import Result
from .service import run


class CaseScore(BaseModel):
    case_id: str
    baseline_gold: bool | None
    compressed_gold: bool | None
    baseline_context_false_positive: bool | None
    compressed_context_false_positive: bool | None
    baseline_input_chars: int
    compressed_input_chars: int
    text_reduction_fraction: float | None


class Comparison(BaseModel):
    mode: str
    corpus_sha256: str
    cases_sha256: str
    policy: dict[str, str | int | float]
    embedding_usage: Usage
    logical_model_calls: int
    scores: list[CaseScore]
    summary: dict[str, float | None]
    runs: list[Result]


def contains(sources: list[Source], gold: Gold) -> bool:
    return any(
        source.title == gold.document_name and gold.quote in source.text for source in sources
    )


def summarize(scores: list[CaseScore]) -> dict[str, float | None]:
    result = {}
    for metric in (
        "baseline_gold",
        "compressed_gold",
        "baseline_context_false_positive",
        "compressed_context_false_positive",
        "text_reduction_fraction",
    ):
        values = [float(value) for score in scores if (value := getattr(score, metric)) is not None]
        result[metric] = mean(values) if values else None
    retained = [
        float(score.compressed_gold is True) for score in scores if score.baseline_gold is True
    ]
    result["gold_loss_rate_given_baseline"] = 1 - mean(retained) if retained else None
    return result


def evaluate(
    store: Store,
    name: str,
    cases: list[Case],
    settings: Settings,
    embedder: Embedder,
    *,
    expansion_model: StructuredModel | None = None,
    compression_model: StructuredModel | None = None,
    single_query: bool = False,
    live: bool = False,
) -> Comparison:
    if not 1 <= len(cases) <= 32 or len({c.id for c in cases}) != len(cases):
        raise AgentError("Evaluation requires 1–32 cases with unique IDs.")
    expanded = not single_query and settings.max_variants > 0
    model_calls = len(cases) * (int(expanded) + int(settings.compression_mode == "llm"))
    embedding_calls = len(cases) * math.ceil(
        (1 + settings.max_variants * int(expanded)) / settings.embedding_batch_size
    )
    if model_calls > settings.max_model_calls or embedding_calls > settings.max_embedding_calls:
        raise AgentError("Evaluation call budget exceeded before any provider requests.")
    snapshot = store.load(name, settings)
    for case in cases:
        pack_sources(case, [], settings.max_context_chars)
        preflight(
            case,
            settings,
            generated=expanded,
            retrieve_only=True,
            query_count=1 + settings.max_variants * int(expanded),
        )
        if case.evidence is not None and (
            not case.evidence.quote.strip()
            or not any(
                p.title == case.evidence.document_name and case.evidence.quote in p.text
                for p in snapshot.build.corpus.passages
            )
        ):
            raise AgentError("Gold evidence is blank, missing, or split across stored passages.")
    scores, runs = [], []
    usage = Usage()
    for case in cases:
        result = run(
            store,
            name,
            Query(question=case.question),
            settings,
            embedder,
            expansion_model=expansion_model,
            compression_model=compression_model,
            provided=[] if single_query else None,
            retrieve_only=True,
            live=live,
        )
        if result.answer.corpus_sha256 != snapshot.build.manifest.corpus_sha256:
            raise AgentError("Snapshot changed during evaluation.")
        runs.append(result)
        baseline = result.retrieval.answer.context
        compressed = result.answer.context
        scores.append(
            CaseScore(
                case_id=case.id,
                baseline_gold=contains(baseline.sources, case.evidence) if case.evidence else None,
                compressed_gold=contains(compressed.sources, case.evidence)
                if case.evidence
                else None,
                baseline_context_false_positive=bool(baseline.sources)
                if not case.evidence
                else None,
                compressed_context_false_positive=bool(compressed.sources)
                if not case.evidence
                else None,
                baseline_input_chars=baseline.input_chars,
                compressed_input_chars=compressed.input_chars,
                text_reduction_fraction=result.compression.text_reduction_fraction,
            )
        )
        usage.logical_calls += result.answer.embedding_usage.logical_calls
        usage.tokens += result.answer.embedding_usage.tokens
    return Comparison(
        mode="live" if live else "injected_provider",
        corpus_sha256=snapshot.build.manifest.corpus_sha256,
        cases_sha256=fingerprint([c.model_dump(mode="json") for c in cases]),
        policy={
            key: getattr(settings, key)
            for key in (
                "compression_mode",
                "unit_chars",
                "max_units",
                "max_selected_units",
                "compression_context_chars",
                "compression_output_tokens",
                "max_context_chars",
                "model",
                "embedding_model",
                "dimensions",
                "top_k",
                "candidate_window",
                "rrf_k",
                "min_similarity",
                "min_matched_terms",
                "max_variants",
            )
        },
        embedding_usage=usage,
        logical_model_calls=sum(r.logical_model_calls for r in runs),
        scores=scores,
        summary=summarize(scores),
        runs=runs,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare packed evidence before/after compression."
    )
    parser.add_argument("name")
    parser.add_argument("cases", type=Path)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--single-query", action="store_true")
    parser.add_argument("--lexical-compression", action="store_true")
    parser.add_argument(
        "--live",
        action="store_true",
        required=True,
        help="Allow provider calls for retrieval/compression, but no final answers.",
    )
    args = parser.parse_args()

    def action() -> Comparison:
        settings = Settings()
        if args.lexical_compression:
            settings.compression_mode = "lexical"
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
        compression = LazyModel(
            settings.model_copy(
                update={
                    "max_output_tokens": settings.compression_output_tokens,
                }
            )
        )
        try:
            return evaluate(
                store,
                args.name,
                cases,
                settings,
                embedding,
                expansion_model=expansion,
                compression_model=compression,
                single_query=args.single_query,
                live=True,
            )
        finally:
            embedding.close()
            expansion.close()
            compression.close()

    return execute(action)


if __name__ == "__main__":
    raise SystemExit(main())
