"""Typed, bounded model calls for graph nodes."""

import hashlib
import json
import logging
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from shared.errors import AgentError

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


class StructuredModel(Protocol):
    def structured(
        self, instructions: str, content: list[dict[str, Any]], schema: type[T]
    ) -> T: ...


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def request(
    model: StructuredModel, prompt: str, payload: Any, schema: type[T], max_chars: int
) -> T:
    text = canonical(payload)
    if len(text) + len(prompt) > max_chars:
        raise AgentError("Model input exceeds the configured context budget.")
    logger.info("structured_request_started")
    result = model.structured(prompt, [{"type": "input_text", "text": text}], schema)
    logger.info("structured_request_completed")
    return result
