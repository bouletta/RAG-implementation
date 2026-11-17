from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

ProfileID = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
MemoryKey = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.-]{0,47}$")]
Revision = Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]


class Scope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    profile: ProfileID


class Remember(Scope):
    key: MemoryKey
    value: str = Field(min_length=1, max_length=500)
    expected_revision: Revision | None = None
    expires_at: AwareDatetime | None = None

    @field_validator("value")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Memory must contain text.")
        return value


class Forget(Scope):
    key: MemoryKey
    expected_revision: Revision


class Selection(Scope):
    keys: list[MemoryKey] = Field(max_length=8)

    @field_validator("keys")
    @classmethod
    def unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Select each memory key only once.")
        return value


class Question(Selection):
    question: str = Field(min_length=1, max_length=8000)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must contain text.")
        return value


class Fact(Scope):
    key: MemoryKey
    value: str = Field(min_length=1, max_length=500)
    revision: Revision
    created_at: AwareDatetime
    updated_at: AwareDatetime
    expires_at: AwareDatetime | None
    source: Literal["explicit_user"] = "explicit_user"


class Reply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(min_length=1, max_length=8000)
    used_keys: list[str] = Field(max_length=8)


class Answer(BaseModel):
    reply: Reply
    supplied_revisions: dict[str, str]
    model: str
