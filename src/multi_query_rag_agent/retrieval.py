"""Fuse query-level rank lists; each query gives a passage at most one vote."""

from collections import defaultdict

from shared.errors import AgentError

from .models import MultiMatch, Vote


def fuse_queries(rankings: list[list[str]], k: int, top_k: int) -> list[MultiMatch]:
    if not 1 <= len(rankings) <= 4 or k < 1 or not 1 <= top_k <= 10:
        raise AgentError("Invalid multi-query fusion bounds.")
    votes: dict[str, list[Vote]] = defaultdict(list)
    for index, ids in enumerate(rankings):
        if len(ids) != len(set(ids)):
            raise AgentError("A query ranking contains duplicate passage IDs.")
        for rank, id_ in enumerate(ids, 1):
            votes[id_].append(Vote(query_index=index, rank=rank))
    matches = [
        MultiMatch(
            source_id=id_, rrf_score=sum(1 / (k + vote.rank) for vote in evidence), votes=evidence
        )
        for id_, evidence in votes.items()
    ]
    return sorted(matches, key=lambda row: (-row.rrf_score, row.source_id))[:top_k]
