"""Exact evidence references, not a claim of semantic entailment."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from shared.errors import AgentError

EvidenceID = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,40}$")]


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: EvidenceID
    quote: str = Field(min_length=1, max_length=2000)


def validate_evidence(items: list[Evidence], sources: dict[str, str]) -> None:
    seen: set[tuple[str, str]] = set()
    for item in items:
        pair = (item.source_id, item.quote)
        if not item.quote.strip() or pair in seen:
            raise AgentError("Evidence must be nonblank and nonduplicated.")
        if item.source_id not in sources or item.quote not in sources[item.source_id]:
            raise AgentError("Evidence references an absent source or invented quote.")
        seen.add(pair)
