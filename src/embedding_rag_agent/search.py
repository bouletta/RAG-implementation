import math
from typing import Protocol

from semantic_memory_agent.embeddings import normalize

from shared.errors import AgentError

from .models import Match, VectorData


class SearchIndex(Protocol):
    name: str

    def search(self, query: tuple[float, ...], top_k: int, minimum: float) -> list[Match]: ...


def validate_search(top_k: int, minimum: float) -> None:
    if not 1 <= top_k <= 10 or not math.isfinite(minimum) or not -1 <= minimum <= 1:
        raise AgentError("Invalid search cutoff or similarity threshold.")


def ranked(scores: list[tuple[str, float]], top_k: int, minimum: float) -> list[Match]:
    validate_search(top_k, minimum)
    if len({id_ for id_, _ in scores}) != len(scores) or any(
        not math.isfinite(s) for _, s in scores
    ):
        raise AgentError("Search returned duplicate IDs or nonfinite scores.")
    matches = [Match(source_id=id_, similarity=max(-1.0, min(1.0, score))) for id_, score in scores]
    return sorted(
        [match for match in matches if match.similarity >= minimum],
        key=lambda match: (-match.similarity, match.source_id),
    )[:top_k]


class ExactIndex:
    name = "python_exact_cosine"

    def __init__(self, data: VectorData) -> None:
        self.data = data

    def search(self, query: tuple[float, ...], top_k: int, minimum: float) -> list[Match]:
        validate_search(top_k, minimum)
        unit = normalize(list(query), self.data.dimensions)
        return ranked(
            [
                (passage.id, math.fsum(a * b for a, b in zip(unit, vector, strict=True)))
                for passage, vector in zip(
                    self.data.corpus.passages, self.data.vectors, strict=True
                )
            ],
            top_k,
            minimum,
        )
