"""Provider-independent runtime contracts."""

from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


@dataclass(frozen=True)
class Call:
    id: str
    name: str
    arguments: str


@dataclass
class Turn:
    text: str = ""
    calls: list[Call] = field(default_factory=list)
    replay: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class Model(Protocol):
    def generate(
        self,
        instructions: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Turn: ...


class Step(BaseModel):
    tool: str
    succeeded: bool
    result: dict[str, Any]


class RunResult(BaseModel):
    answer: str = Field(min_length=1)
    steps: list[Step]
    input_tokens: int = 0
    output_tokens: int = 0
