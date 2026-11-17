from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from shared.evidence import Evidence, EvidenceID


class Passage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: EvidenceID
    document_id: EvidenceID
    title: str = Field(min_length=1, max_length=200)
    locator: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=8000)

    @field_validator("title", "locator", "text")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Passage fields must not be blank.")
        return value


class Corpus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    passages: tuple[Passage, ...] = Field(max_length=200)

    @model_validator(mode="after")
    def unique_sources(self) -> "Corpus":
        if len({p.id for p in self.passages}) != len(self.passages):
            raise ValueError("Passage IDs must be unique.")
        locations = {(p.document_id, p.locator) for p in self.passages}
        if len(locations) != len(self.passages):
            raise ValueError("Document locations must be unique.")
        titles: dict[str, str] = {}
        for passage in self.passages:
            if passage.document_id in titles and titles[passage.document_id] != passage.title:
                raise ValueError("A document must have one consistent title.")
            titles[passage.document_id] = passage.title
        return self


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    question: str = Field(min_length=1, max_length=1000)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must not be blank.")
        return value


class Source(Passage):
    sha256: str


class Hit(BaseModel):
    source: Source
    matched_terms: int


class Context(BaseModel):
    sources: list[Source]
    dropped_ids: list[str]
    input_chars: int
    payload_sha256: str


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=800)
    evidence: list[Evidence] = Field(min_length=1, max_length=4)


class Reply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[Claim] = Field(max_length=6)
    limitation: str = Field(min_length=1, max_length=1000)


class Ranking(BaseModel):
    source_id: str
    matched_terms: int


class Result(BaseModel):
    schema_version: Literal[1] = 1
    status: Literal["preview", "answered", "abstained"]
    reason: Literal["preview", "answered", "no_matches", "context_budget", "insufficient_evidence"]
    retrieval_method: Literal["unique_term_overlap_v1"] = "unique_term_overlap_v1"
    corpus_sha256: str
    prompt_sha256: str
    generation_model: str | None
    top_k: int
    min_matched_terms: int
    context: Context
    ranking: list[Ranking]
    reply: Reply | None
    logical_model_calls: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    timing_ms: dict[str, float]
