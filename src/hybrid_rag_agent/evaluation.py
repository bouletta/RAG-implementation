"""Paired retrieval evaluation; no generated answers and no LLM judge."""

import argparse
from pathlib import Path

from basic_rag_agent.context import pack_sources
from basic_rag_agent.corpus import sources
from basic_rag_agent.models import Query
from chroma_rag_agent.llm import LazyEmbedder
from chroma_rag_agent.store import Store
from chunking_agent.evaluation import Case
from embedding_rag_agent.embeddings import preflight
from embedding_rag_agent.models import Usage
from pydantic import BaseModel, TypeAdapter
from semantic_memory_agent.embeddings import Embedder

from shared.cli import execute
from shared.errors import AgentError
from shared.files import read_limited
from shared.structured import fingerprint

from .config import Settings
from .service import run


class Score(BaseModel):
    case_id: str
    method: str
    hit_at_k: bool | None
    reciprocal_rank: float | None
    intact_in_context: bool | None
    false_positive: bool | None


class Report(BaseModel):
    snapshot: str
    corpus_sha256: str
    cases_sha256: str
    mode: str
    top_k: int
    candidate_window: int
    min_similarity: float
    min_matched_terms: int
    rrf_k: int
    embedding_usage: Usage
    scores: list[Score]
    summaries: dict[str, dict[str, float | None]]


def evaluate(
    store: Store,
    name: str,
    cases: list[Case],
    settings: Settings,
    embedder: Embedder,
    *,
    live: bool = False,
) -> Report:
    snapshot = store.load(name, settings)
    if not 1 <= len(cases) <= 100 or len({c.id for c in cases}) != len(cases):
        raise AgentError("Evaluation requires 1–100 cases with unique IDs.")
    corpus = snapshot.build.corpus
    # Document embeddings already exist; only query calls count against this run's budget.
    from basic_rag_agent.models import Corpus

    preflight(Corpus(passages=()), [case.question for case in cases], settings)
    targets: dict[str, set[str]] = {}
    lookup = {source.id: source for source in sources(corpus)}
    for case in cases:
        pack_sources(Query(question=case.question), [], settings.max_context_chars)
        targets[case.id] = {
            p.id
            for p in corpus.passages
            if case.evidence is not None
            and p.title == case.evidence.document_name
            and case.evidence.quote in p.text
        }
        if case.evidence is not None and not targets[case.id]:
            raise AgentError("A gold quote is absent or split across chunks in this snapshot.")
    scores: list[Score] = []
    usage = Usage()
    for case in cases:
        query = Query(question=case.question)
        result = run(store, name, query, settings, embedder, retrieve_only=True, live=live)
        usage.logical_calls += result.answer.embedding_usage.logical_calls
        usage.tokens += result.answer.embedding_usage.tokens
        rankings = {
            "lexical": result.lexical_ids[: settings.top_k],
            "semantic": result.semantic_ids[: settings.top_k],
            "hybrid": [row.source_id for row in result.fusion],
        }
        for method, ids in rankings.items():
            context = pack_sources(query, [lookup[id_] for id_ in ids], settings.max_context_chars)
            ranks = [i for i, id_ in enumerate(ids, 1) if id_ in targets[case.id]]
            positive = case.evidence is not None
            scores.append(
                Score(
                    case_id=case.id,
                    method=method,
                    hit_at_k=bool(ranks) if positive else None,
                    reciprocal_rank=1 / min(ranks) if ranks else 0.0 if positive else None,
                    intact_in_context=bool({s.id for s in context.sources} & targets[case.id])
                    if positive
                    else None,
                    false_positive=bool(ids) if not positive else None,
                )
            )
    summaries: dict[str, dict[str, float | None]] = {}
    for method in ("lexical", "semantic", "hybrid"):
        summaries[method] = {}
        for metric in ("hit_at_k", "reciprocal_rank", "intact_in_context", "false_positive"):
            values = [
                float(value)
                for score in scores
                if score.method == method and (value := getattr(score, metric)) is not None
            ]
            summaries[method][metric] = sum(values) / len(values) if values else None
    return Report(
        snapshot=name,
        corpus_sha256=snapshot.build.manifest.corpus_sha256,
        cases_sha256=fingerprint([case.model_dump(mode="json") for case in cases]),
        mode="live" if live else "injected_provider",
        top_k=settings.top_k,
        candidate_window=settings.candidate_window,
        min_similarity=settings.min_similarity,
        min_matched_terms=settings.min_matched_terms,
        rrf_k=settings.rrf_k,
        embedding_usage=usage,
        scores=scores,
        summaries=summaries,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate three retrieval channels on one snapshot."
    )
    parser.add_argument("name")
    parser.add_argument("cases", type=Path)
    parser.add_argument("--store", type=Path)
    parser.add_argument(
        "--live",
        action="store_true",
        required=True,
        help="Acknowledges paid query embeddings; no generation.",
    )
    args = parser.parse_args()

    def action() -> Report:
        settings = Settings()
        cases = TypeAdapter(list[Case]).validate_json(read_limited(args.cases, 100_000))
        store = Store(args.store or settings.store_path)
        provider = LazyEmbedder(settings)
        try:
            return evaluate(store, args.name, cases, settings, provider, live=True)
        finally:
            provider.close()

    return execute(action)


if __name__ == "__main__":
    raise SystemExit(main())
