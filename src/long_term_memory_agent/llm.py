"""Bound user-data payloads and keep memory outside system instructions."""

import json
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from shared.errors import AgentError

from .models import Fact, Question
from .prompts import SYSTEM_PROMPT

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class StructuredModel(Protocol):
    def structured(
        self, instructions: str, content: list[dict[str, Any]], schema: type[ResponseT]
    ) -> ResponseT: ...


def build_content(question: Question, facts: list[Fact], max_chars: int) -> list[dict[str, Any]]:
    payload = json.dumps(
        {
            "question": question.question,
            "remembered_facts": [
                {"key": fact.key, "value": fact.value, "source": fact.source} for fact in facts
            ],
        },
        ensure_ascii=True,
    )
    if len(payload) + len(SYSTEM_PROMPT) > max_chars:
        raise AgentError("Selected memories and question exceed the context budget.")
    return [{"type": "input_text", "text": payload}]
