"""Compare policies using source evidence spans rather than policy-dependent chunk IDs."""

import argparse
from pathlib import Path
from statistics import mean
from typing import Literal

from basic_rag_agent.models import Query
from basic_rag_agent.service import run as run_rag
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from shared.cli import execute
from shared.errors import AgentError
from shared.evidence import EvidenceID
from shared.files import read_limited
from shared.structured import fingerprint

from .chunking import build
from .config import Settings
from .ingestion import load_documents
from .models import Chunk, Document, Policy


class Gold(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_name: str = Field(min_length=1, max_length=200)
    quote: str = Field(min_length=1, max_length=8000)


class Case(Query):
    id: EvidenceID
    evidence: Gold | None


class CaseScore(BaseModel):
    case_id: str
    intact_in_corpus: bool | None
    intact_in_retrieval: bool | None
    intact_in_context: bool | None
    false_positive: bool | None
    packed_false_positive: bool | None


class PolicyReport(BaseModel):
    policy: Policy
    corpus_sha256: str
    chunk_count: int
    duplicated_characters: int
    emitted_characters: int
    cases: list[CaseScore]
    intact_corpus_rate: float | None
    intact_retrieval_rate: float | None
    intact_context_rate: float | None
    false_positive_rate: float | None
    packed_false_positive_rate: float | None


class Comparison(BaseModel):
    scope: Literal["single_span_containment"] = "single_span_containment"
    documents_sha256: str
    cases_sha256: str
    top_k: int
    min_matched_terms: int
    max_context_chars: int
    policies: list[PolicyReport]


def gold_spans(
    documents: tuple[Document, ...], cases: list[Case]
) -> dict[str, tuple[str, int, int] | None]:
    if not 1 <= len(cases) <= 100 or len({c.id for c in cases}) != len(cases):
        raise AgentError("Evaluation requires 1–100 uniquely identified cases.")
    by_name = {d.name: d for d in documents}
    if len(by_name) != len(documents):
        raise AgentError("Document names must be unique for evaluation.")
    targets: dict[str, tuple[str, int, int] | None] = {}
    for case in cases:
        if case.evidence is None:
            targets[case.id] = None
            continue
        gold = case.evidence
        document = by_name.get(gold.document_name)
        if document is None or not gold.quote.strip():
            raise AgentError("Gold evidence must reference a known document and nonblank quote.")
        start = document.text.find(gold.quote)
        if start < 0 or document.text.find(gold.quote, start + 1) >= 0:
            raise AgentError("Gold quote must occur exactly once in the original document.")
        targets[case.id] = (document.id, start, start + len(gold.quote))
    return targets


def intact(chunks: list[Chunk], target: tuple[str, int, int] | None) -> bool | None:
    if target is None:
        return None
    document_id, start, end = target
    return any(
        chunk.document_id == document_id and chunk.start <= start and chunk.end >= end
        for chunk in chunks
    )


def rate(values: list[bool | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return mean(defined) if defined else None


def evaluate(documents: tuple[Document, ...], cases: list[Case], settings: Settings) -> Comparison:
    targets = gold_spans(documents, cases)
    reports = []
    strategies: list[Literal["fixed", "boundary"]] = ["fixed", "boundary"]
    for strategy in strategies:
        policy = Policy(strategy=strategy, size=settings.chunk_size, overlap=settings.chunk_overlap)
        built = build(documents, policy, settings.max_chunks)
        by_id = {chunk.id: chunk for chunk in built.manifest.chunks}
        rows = []
        for case in cases:
            result = run_rag(built.corpus, Query(question=case.question), settings, preview=True)
            retrieved = [r.source_id for r in result.ranking]
            packed = [s.id for s in result.context.sources]
            target = targets[case.id]

            rows.append(
                CaseScore(
                    case_id=case.id,
                    intact_in_corpus=intact(list(by_id.values()), target),
                    intact_in_retrieval=intact([by_id[id_] for id_ in retrieved], target),
                    intact_in_context=intact([by_id[id_] for id_ in packed], target),
                    false_positive=bool(retrieved) if target is None else None,
                    packed_false_positive=bool(packed) if target is None else None,
                )
            )

        reports.append(
            PolicyReport(
                policy=policy,
                corpus_sha256=built.manifest.corpus_sha256,
                chunk_count=len(by_id),
                duplicated_characters=built.manifest.duplicated_characters,
                emitted_characters=built.manifest.emitted_characters,
                cases=rows,
                intact_corpus_rate=rate([row.intact_in_corpus for row in rows]),
                intact_retrieval_rate=rate([row.intact_in_retrieval for row in rows]),
                intact_context_rate=rate([row.intact_in_context for row in rows]),
                false_positive_rate=rate([row.false_positive for row in rows]),
                packed_false_positive_rate=rate([row.packed_false_positive for row in rows]),
            )
        )
    return Comparison(
        documents_sha256=fingerprint([{"name": d.name, "sha256": d.sha256} for d in documents]),
        cases_sha256=fingerprint([c.model_dump(mode="json") for c in cases]),
        top_k=settings.top_k,
        min_matched_terms=settings.min_matched_terms,
        max_context_chars=settings.max_context_chars,
        policies=reports,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args()

    def action() -> Comparison:
        settings = Settings()
        cases = TypeAdapter(list[Case]).validate_json(read_limited(args.cases, 200_000))
        return evaluate(load_documents(args.files, settings), cases, settings)

    return execute(action)


if __name__ == "__main__":
    raise SystemExit(main())
