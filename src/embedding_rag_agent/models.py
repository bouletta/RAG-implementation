import math
from typing import Literal

from basic_rag_agent.models import Context, Corpus, Reply
from chunking_agent.models import Manifest
from pydantic import BaseModel, ConfigDict, Field, model_validator

from shared.structured import fingerprint


class VectorData(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    corpus: Corpus
    embedding_model: str
    dimensions: int = Field(ge=16, le=1536)
    vectors: tuple[tuple[float, ...], ...]
    normalization: Literal["l2_v1"] = "l2_v1"

    @model_validator(mode="after")
    def aligned_unit_vectors(self) -> "VectorData":
        if len(self.vectors) != len(self.corpus.passages):
            raise ValueError("Vector rows must match passage rows exactly.")
        for vector in self.vectors:
            if len(vector) != self.dimensions or not all(math.isfinite(x) for x in vector):
                raise ValueError("Vector dimensions and values must be valid.")
            if not math.isclose(math.hypot(*vector), 1, abs_tol=1e-6):
                raise ValueError("Stored vectors must be unit normalized.")
        return self

    @property
    def sha256(self) -> str:
        return fingerprint(self.model_dump(mode="json"))


class Match(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_id: str
    similarity: float = Field(ge=-1, le=1, allow_inf_nan=False)


class Usage(BaseModel):
    logical_calls: int = 0
    tokens: int = 0


class Answer(BaseModel):
    schema_version: Literal[1] = 1
    status: Literal["retrieved", "answered", "abstained"]
    reason: Literal[
        "retrieval_only", "no_matches", "context_budget", "insufficient_evidence", "answered"
    ]
    backend: str
    mode: Literal["live", "injected_provider"]
    manifest: Manifest
    index_sha256: str
    embedding_model: str
    dimensions: int
    top_k: int
    min_similarity: float
    matches: list[Match]
    context: Context
    reply: Reply | None
    embedding_usage: Usage
    logical_generation_calls: int
    generation_model: str | None
    generation_input_tokens: int | None = None
    generation_output_tokens: int | None = None
    prompt_sha256: str
    timing_ms: dict[str, float]
