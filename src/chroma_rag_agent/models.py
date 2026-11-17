from typing import Literal

from basic_rag_agent.models import Context, Reply
from chunking_agent.models import Build
from embedding_rag_agent.models import Match, Usage
from pydantic import BaseModel, ConfigDict, Field


class Snapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    name: str
    build: Build
    embedding_model: str
    dimensions: int = Field(ge=16, le=1536)
    normalization: Literal["l2_float32_v1"] = "l2_float32_v1"
    vectors_sha256: str


class IngestResult(BaseModel):
    snapshot: str
    corpus_sha256: str
    passages: int
    reused: bool
    embedding_usage: Usage


class Answer(BaseModel):
    mode: Literal["live", "injected_provider"]
    status: Literal["retrieved", "answered", "abstained"]
    reason: str
    snapshot: str
    corpus_sha256: str
    retrieval_method: str
    matches: list[Match]
    context: Context
    reply: Reply | None
    embedding_usage: Usage
    logical_generation_calls: int
    generation_model: str | None
    input_tokens: int | None = None
    output_tokens: int | None = None
    prompt_sha256: str
    elapsed_ms: float
