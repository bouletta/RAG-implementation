from typing import Literal

from basic_rag_agent.models import Context, Source
from chroma_rag_agent.models import Answer
from multi_query_rag_agent.models import Result as RetrievalResult
from pydantic import BaseModel, ConfigDict, Field

from shared.evidence import EvidenceID


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unit_ids: list[EvidenceID] = Field(max_length=24)


class Unit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source: Source
    parent_id: str
    parent_sha256: str
    parent_locator: str
    start: int
    end: int


class Compression(BaseModel):
    mode: Literal["llm", "lexical"]
    context: Context
    units: list[Unit]
    selected_ids: list[str]
    omitted_ids: list[str]
    logical_model_calls: int
    model: str | None
    prompt_sha256: str | None
    selector_input_chars: int
    baseline_input_chars: int
    retrieved_text_chars: int
    selected_text_chars: int
    packed_text_chars: int
    text_reduction_fraction: float | None
    packed_input_char_delta: int
    input_tokens: int | None = None
    output_tokens: int | None = None


class Result(BaseModel):
    retrieval: RetrievalResult
    compression: Compression
    answer: Answer
    logical_model_calls: int
