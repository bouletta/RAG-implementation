from typing import Annotated, Literal

from chroma_rag_agent.models import Answer
from embedding_rag_agent.models import Match
from hybrid_rag_agent.models import FusedMatch
from pydantic import BaseModel, ConfigDict, Field

SearchText = Annotated[str, Field(min_length=1, max_length=1000)]


class Variants(BaseModel):
    model_config = ConfigDict(extra="forbid")
    queries: list[SearchText] = Field(max_length=3)


class Expansion(BaseModel):
    mode: Literal["generated", "provided", "disabled"]
    queries: list[str]
    discarded_duplicates: int
    logical_model_calls: int
    model: str | None
    prompt_sha256: str | None
    input_tokens: int | None = None
    output_tokens: int | None = None


class Branch(BaseModel):
    query_index: int
    lexical_ids: list[str]
    semantic_matches: list[Match]
    ranking: list[FusedMatch]


class Vote(BaseModel):
    query_index: int
    rank: int


class MultiMatch(BaseModel):
    source_id: str
    rrf_score: float
    votes: list[Vote]


class Result(BaseModel):
    answer: Answer
    expansion: Expansion
    branches: list[Branch]
    fusion: list[MultiMatch]
    logical_model_calls: int
    candidate_window: int
    rrf_k: int
