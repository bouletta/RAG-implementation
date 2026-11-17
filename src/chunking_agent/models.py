import hashlib
from typing import Literal

from basic_rag_agent.models import Corpus, Result
from pydantic import BaseModel, ConfigDict, Field, model_validator

from shared.evidence import EvidenceID
from shared.structured import fingerprint


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["character_chunks_v1"] = "character_chunks_v1"
    strategy: Literal["fixed", "boundary"] = "boundary"
    size: int = Field(default=600, ge=64, le=4000)
    overlap: int = Field(default=80, ge=0)

    @model_validator(mode="after")
    def bounded_overlap(self) -> "Policy":
        if self.overlap > self.size // 2:
            raise ValueError("Overlap must not exceed half the chunk size.")
        return self


class Document(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: EvidenceID
    name: str = Field(min_length=1, max_length=200)
    sha256: str
    text: str

    @model_validator(mode="after")
    def consistent_identity(self) -> "Document":
        if not self.text.strip() or "\x00" in self.text:
            raise ValueError("Document text must be nonblank and contain no NUL characters.")
        digest = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        expected = "d_" + fingerprint({"name": self.name, "sha256": digest})[:32]
        if self.sha256 != digest or self.id != expected:
            raise ValueError("Document identity must match its exact name and UTF-8 content.")
        return self


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: EvidenceID
    document_id: EvidenceID
    start: int
    end: int


class DocumentInfo(BaseModel):
    id: str
    name: str
    sha256: str
    characters: int


class Manifest(BaseModel):
    schema_version: Literal[1] = 1
    offset_unit: Literal["unicode_code_points"] = "unicode_code_points"
    policy: Policy
    documents: list[DocumentInfo]
    chunks: list[Chunk]
    source_characters: int
    emitted_characters: int
    unique_covered_characters: int
    duplicated_characters: int
    corpus_sha256: str


class Build(BaseModel):
    corpus: Corpus
    manifest: Manifest


class QuoteLocation(BaseModel):
    claim_index: int
    source_id: str
    document_id: str
    document_sha256: str
    start: int
    end: int
    ambiguous_in_chunk: bool


class Answer(BaseModel):
    manifest: Manifest
    rag: Result
    quote_locations: list[QuoteLocation]
