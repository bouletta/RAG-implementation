"""Offline retrieval and packing evaluation; never an answer-quality benchmark."""

import argparse
from pathlib import Path

from memory_benchmark_agent.metrics import aggregate, score
from memory_benchmark_agent.models import Aggregate
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from shared.cli import execute
from shared.errors import AgentError
from shared.evidence import EvidenceID
from shared.files import read_limited
from shared.structured import fingerprint

from .config import Settings
from .corpus import load_corpus
from .models import Corpus, Query
from .service import run


class EvalCase(Query):
    id: EvidenceID
    relevant_ids: list[EvidenceID] = Field(max_length=200)


class Evaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scope: str = "offline_retrieval_and_packing"
    corpus_sha256: str
    cases_sha256: str
    top_k: int
    min_matched_terms: int
    max_context_chars: int
    retrieval: Aggregate
    packed_context: Aggregate


def evaluate(corpus: Corpus, cases: list[EvalCase], settings: Settings) -> Evaluation:
    if not 1 <= len(cases) <= 100 or len({c.id for c in cases}) != len(cases):
        raise AgentError("Evaluation requires 1–100 uniquely identified cases.")
    known = {p.id for p in corpus.passages}
    for case in cases:
        if (
            len(case.relevant_ids) != len(set(case.relevant_ids))
            or not set(case.relevant_ids) <= known
        ):
            raise AgentError("Evaluation labels must be unique known passage IDs.")
    retrieval_rows, packed_rows = [], []
    for case in cases:
        result = run(corpus, Query(question=case.question), settings, preview=True)
        relevant = set(case.relevant_ids)
        retrieval_rows.append(
            score([r.source_id for r in result.ranking], relevant, settings.top_k)
        )
        packed_rows.append(score([s.id for s in result.context.sources], relevant, settings.top_k))
    return Evaluation(
        corpus_sha256=fingerprint(corpus.model_dump(mode="json")),
        cases_sha256=fingerprint([c.model_dump(mode="json") for c in cases]),
        top_k=settings.top_k,
        min_matched_terms=settings.min_matched_terms,
        max_context_chars=settings.max_context_chars,
        retrieval=aggregate(retrieval_rows),
        packed_context=aggregate(packed_rows),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("cases", type=Path)
    args = parser.parse_args()

    def action() -> Evaluation:
        settings = Settings()
        corpus = load_corpus(args.corpus, settings.max_corpus_bytes)
        cases = TypeAdapter(list[EvalCase]).validate_json(read_limited(args.cases, 200_000))
        return evaluate(corpus, cases, settings)

    return execute(action)


if __name__ == "__main__":
    raise SystemExit(main())
