"""Fuse ranks within one fact corpus, never raw cosine and BM25 scores."""

import re
from dataclasses import dataclass

from long_term_memory_agent.models import Fact


@dataclass(frozen=True)
class FusedRank:
    key: str
    score: float
    lexical_rank: int | None
    semantic_rank: int | None


def lexical_rank(question: str, facts: list[Fact], window: int) -> list[str]:
    terms = set(re.findall(r"\w+", question.casefold()))
    scored = []
    for fact in facts:
        words = set(re.findall(r"\w+", f"{fact.key} {fact.value}".casefold()))
        overlap = len(terms & words)
        if overlap:
            scored.append((fact.key, overlap))
    return [key for key, _ in sorted(scored, key=lambda row: (-row[1], row[0]))[:window]]


def fuse(lexical: list[str], semantic: list[str], k: int) -> list[FusedRank]:
    if k < 1:
        raise ValueError("RRF constant must be positive.")
    if len(lexical) != len(set(lexical)) or len(semantic) != len(set(semantic)):
        raise ValueError("Rank lists must not contain duplicate keys.")
    left = {key: rank for rank, key in enumerate(lexical, 1)}
    right = {key: rank for rank, key in enumerate(semantic, 1)}
    ranked = [
        FusedRank(
            key=key,
            score=sum(1 / (k + ranks[key]) for ranks in (left, right) if key in ranks),
            lexical_rank=left.get(key),
            semantic_rank=right.get(key),
        )
        for key in left.keys() | right.keys()
    ]
    return sorted(ranked, key=lambda item: (-item.score, item.key))
