"""Compare child and parent evidence using the same retrieved child hits."""

import argparse
from pathlib import Path
from statistics import mean

from basic_rag_agent.context import pack_sources
from basic_rag_agent.models import Query
from chroma_rag_agent.llm import LazyEmbedder
from chunking_agent.evaluation import Case
from context_compression_agent.evaluation import contains
from embedding_rag_agent.models import Usage
from pydantic import BaseModel, TypeAdapter
from semantic_memory_agent.embeddings import Embedder

from shared.cli import execute
from shared.errors import AgentError
from shared.files import read_limited
from shared.structured import fingerprint

from .config import Settings
from .service import run
from .store import ParentStore


class Score(BaseModel):
    case_id: str
    child_gold: bool | None
    parent_gold: bool | None
    child_context_false_positive: bool | None
    parent_context_false_positive: bool | None
    child_input_chars: int
    parent_input_chars: int
    child_ids: list[str]
    parent_ids: list[str]


class Comparison(BaseModel):
    mode: str
    hierarchy_sha256: str
    cases_sha256: str
    policy: dict[str, str | int | float]
    embedding_usage: Usage
    logical_model_calls: int = 0
    scores: list[Score]
    summary: dict[str, float | None]


def evaluate(
    store: ParentStore,
    name: str,
    cases: list[Case],
    settings: Settings,
    embedder: Embedder,
    *,
    live: bool = False,
) -> Comparison:
    if not 1 <= len(cases) <= 32 or len({case.id for case in cases}) != len(cases):
        raise AgentError("Evaluation requires 1–32 cases with unique IDs.")
    if len(cases) > settings.max_embedding_calls:
        raise AgentError("Evaluation embedding budget exceeded before provider requests.")
    if settings.compress_parents:
        raise AgentError("This comparison measures parent expansion before compression.")
    hierarchy = store.load_hierarchy(name, settings)
    for case in cases:
        pack_sources(case, [], settings.max_context_chars)
        if case.evidence and (
            not case.evidence.quote.strip()
            or not any(
                p.title == case.evidence.document_name and case.evidence.quote in p.text
                for p in hierarchy.parents.corpus.passages
            )
        ):
            raise AgentError("Gold evidence must exist within a stored parent.")
    scores: list[Score] = []
    usage = Usage()
    for case in cases:
        result = run(
            store,
            name,
            Query(question=case.question),
            settings,
            embedder,
            provided=[],
            retrieve_only=True,
            live=live,
        )
        if result.hierarchy_sha256 != hierarchy.sha256:
            raise AgentError("Hierarchy changed during evaluation.")
        child, parent = result.retrieval.answer.context, result.answer.context
        scores.append(
            Score(
                case_id=case.id,
                child_gold=contains(child.sources, case.evidence) if case.evidence else None,
                parent_gold=contains(parent.sources, case.evidence) if case.evidence else None,
                child_context_false_positive=bool(child.sources) if not case.evidence else None,
                parent_context_false_positive=bool(parent.sources) if not case.evidence else None,
                child_input_chars=child.input_chars,
                parent_input_chars=parent.input_chars,
                child_ids=[hit.source_id for hit in result.retrieval.fusion],
                parent_ids=[hit.parent_id for hit in result.parents],
            )
        )
        usage.logical_calls += result.answer.embedding_usage.logical_calls
        usage.tokens += result.answer.embedding_usage.tokens
    summary: dict[str, float | None] = {}
    for metric in (
        "child_gold",
        "parent_gold",
        "child_context_false_positive",
        "parent_context_false_positive",
    ):
        values = [float(value) for score in scores if (value := getattr(score, metric)) is not None]
        summary[metric] = mean(values) if values else None
    retained = [float(score.parent_gold is True) for score in scores if score.child_gold is True]
    summary["gold_loss_given_child"] = 1 - mean(retained) if retained else None
    summary["mean_input_character_delta"] = mean(
        score.parent_input_chars - score.child_input_chars for score in scores
    )
    return Comparison(
        mode="live" if live else "injected_provider",
        hierarchy_sha256=hierarchy.sha256,
        cases_sha256=fingerprint([case.model_dump(mode="json") for case in cases]),
        policy={
            key: getattr(settings, key)
            for key in (
                "top_k",
                "candidate_window",
                "rrf_k",
                "min_similarity",
                "min_matched_terms",
                "max_parents",
                "max_context_chars",
                "embedding_model",
                "dimensions",
            )
        },
        embedding_usage=usage,
        scores=scores,
        summary=summary,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paired child/parent evidence comparison; no LLM calls."
    )
    parser.add_argument("name")
    parser.add_argument("cases", type=Path)
    parser.add_argument("--store", type=Path)
    parser.add_argument(
        "--live",
        action="store_true",
        required=True,
        help="Allow one live question embedding per case.",
    )
    args = parser.parse_args()

    def action() -> Comparison:
        settings = Settings()
        cases = TypeAdapter(list[Case]).validate_json(read_limited(args.cases, 100_000))
        embedder = LazyEmbedder(settings)
        try:
            return evaluate(
                ParentStore(args.store or settings.store_path),
                args.name,
                cases,
                settings,
                embedder,
                live=True,
            )
        finally:
            embedder.close()

    return execute(action)


if __name__ == "__main__":
    raise SystemExit(main())
