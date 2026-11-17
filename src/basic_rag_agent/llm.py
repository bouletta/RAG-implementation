from typing import Any, TypeVar

from pydantic import BaseModel

from shared.llm import OpenAIModel

from .config import Settings

T = TypeVar("T", bound=BaseModel)


class LazyModel:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: OpenAIModel | None = None

    def structured(self, instructions: str, content: list[dict[str, Any]], schema: type[T]) -> T:
        if self._client is None:
            self._client = OpenAIModel(self.settings)
        return self._client.structured(instructions, content, schema)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
