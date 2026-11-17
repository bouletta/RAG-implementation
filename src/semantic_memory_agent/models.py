from long_term_memory_agent.models import Answer, Fact, Scope
from pydantic import BaseModel, Field, field_validator


class Query(Scope):
    question: str = Field(min_length=1, max_length=1000)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question cannot be blank.")
        return value


class Match(BaseModel):
    fact: Fact
    similarity: float


class SearchResult(BaseModel):
    matches: list[Match]
    embedding_model: str
    dimensions: int
    embedding_tokens: int


class Result(BaseModel):
    retrieval: SearchResult
    answer: Answer | None
