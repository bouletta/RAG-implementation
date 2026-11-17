from typing import Literal

from chroma_rag_agent.models import Answer
from chunking_agent.models import Build
from context_compression_agent.models import Compression
from multi_query_rag_agent.models import Result as RetrievalResult
from pydantic import BaseModel, ConfigDict, Field

from shared.structured import fingerprint


class Link(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    child_id: str
    parent_id: str


class Hierarchy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    parents: Build
    children: Build
    links: list[Link] = Field(max_length=200)

    @property
    def sha256(self) -> str:
        return fingerprint(self.model_dump(mode="json"))


class ParentHit(BaseModel):
    parent_id: str
    child_ids: list[str]


class Result(BaseModel):
    hierarchy_sha256: str
    retrieval: RetrievalResult
    parents: list[ParentHit]
    compression: Compression | None
    answer: Answer
    logical_model_calls: int
