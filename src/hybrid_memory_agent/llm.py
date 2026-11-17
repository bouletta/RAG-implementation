"""Lazy provider ownership keeps empty and preview paths credential-free."""

from typing import Any, TypeVar

from pydantic import BaseModel
from semantic_memory_agent.embeddings import Embedded, OpenAIEmbedder

from shared.llm import OpenAIModel

from .config import Settings

T = TypeVar("T", bound=BaseModel)


class Providers:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._embedding: OpenAIEmbedder | None = None
        self._generation: OpenAIModel | None = None

    def embed(self, texts: list[str]) -> Embedded:
        if self._embedding is None:
            self._embedding = OpenAIEmbedder(self.settings)
        return self._embedding.embed(texts)

    def structured(self, instructions: str, content: list[dict[str, Any]], schema: type[T]) -> T:
        if self._generation is None:
            self._generation = OpenAIModel(self.settings)
        return self._generation.structured(instructions, content, schema)

    def close(self) -> None:
        if self._generation is not None:
            self._generation.close()
        if self._embedding is not None:
            self._embedding.close()
