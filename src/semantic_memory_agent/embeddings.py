"""Explicit embedding transport and numerically checked vectors."""

import math
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI, OpenAIError

from shared.errors import AgentError

from .config import Settings


@dataclass
class Embedded:
    vectors: list[list[float]]
    tokens: int = 0


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> Embedded: ...


def normalize(vector: list[float], dimensions: int) -> list[float]:
    if len(vector) != dimensions or not all(math.isfinite(value) for value in vector):
        raise AgentError("Embedding has invalid dimensions or nonfinite values.")
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or norm == 0:
        raise AgentError("Embedding must have a finite nonzero norm.")
    return [value / norm for value in vector]


class OpenAIEmbedder:
    def __init__(self, settings: Settings) -> None:
        if settings.api_key is None or not settings.api_key.get_secret_value().strip():
            raise AgentError("Set OPENAI_API_KEY for semantic retrieval.")
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.api_key.get_secret_value(),
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
        )

    def close(self) -> None:
        self.client.close()

    def embed(self, texts: list[str]) -> Embedded:
        if not 1 <= len(texts) <= 101 or any(
            not text.strip() or len(text.encode("utf-8")) > 8000 for text in texts
        ):
            raise AgentError("Embedding inputs exceed the batch or text budget.")
        try:
            response = self.client.embeddings.create(
                model=self.settings.embedding_model,
                dimensions=self.settings.dimensions,
                input=texts,
                encoding_format="float",
            )
        except OpenAIError as exc:
            raise AgentError(
                "Embedding request failed; check quota, credentials or connectivity."
            ) from exc
        rows = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in rows] != list(range(len(texts))):
            raise AgentError("Embedding response omitted or duplicated an input.")
        return Embedded(
            vectors=[item.embedding for item in rows], tokens=response.usage.total_tokens
        )
