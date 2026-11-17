from typing import Literal

from episodic_memory_agent.models import Episode
from long_term_memory_agent.models import Fact
from profile_agent.models import Profile
from pydantic import BaseModel, ConfigDict, Field, field_validator

from shared.evidence import Evidence, EvidenceID


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile: EvidenceID
    question: str = Field(min_length=1, max_length=1000)
    mode: Literal["hybrid", "lexical"] = "hybrid"

    @field_validator("question")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must not be blank.")
        return value


class RankedFact(BaseModel):
    fact: Fact
    score: float
    lexical_rank: int | None
    semantic_rank: int | None


class Retrieval(BaseModel):
    facts: list[RankedFact]
    episodes: list[Episode]
    preferences: Profile
    mode: Literal["hybrid", "lexical"]
    embedding_tokens: int


class Source(BaseModel):
    id: EvidenceID
    kind: Literal["asserted_fact", "historical_episode"]
    key: str
    revision: str
    recorded_at: str
    text: str


class Context(BaseModel):
    sources: list[Source]
    preferences: Profile
    dropped_source_ids: list[str]
    input_chars: int
    payload_sha256: str


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=800)
    evidence: list[Evidence] = Field(min_length=1, max_length=4)


class Reply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[Claim] = Field(max_length=6)
    uncertainty: str = Field(min_length=1, max_length=1000)


class Result(BaseModel):
    status: Literal["preview", "retrieved", "answered", "no_evidence"]
    mode: Literal["hybrid", "lexical"]
    context: Context
    fact_ranking: list[RankedFact]
    embedding_tokens: int
    reply: Reply | None
    memory_written: Literal[False] = False
