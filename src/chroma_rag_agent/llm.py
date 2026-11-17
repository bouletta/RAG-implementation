"""Reuse provider transports without requiring a key for snapshot inspection or reuse."""

from semantic_memory_agent.embeddings import Embedded, OpenAIEmbedder

from .config import Settings


class LazyEmbedder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._provider: OpenAIEmbedder | None = None

    def embed(self, texts: list[str]) -> Embedded:
        if self._provider is None:
            self._provider = OpenAIEmbedder(self.settings.transport_settings())
        return self._provider.embed(texts)

    def close(self) -> None:
        if self._provider is not None:
            self._provider.close()
