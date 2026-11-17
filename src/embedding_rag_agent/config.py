from pathlib import Path
from typing import Literal

from chunking_agent.config import Settings as ChunkSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict
from semantic_memory_agent.config import Settings as EmbeddingSettings


class _TransportSettings(EmbeddingSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore", populate_by_name=True)


class Settings(ChunkSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        extra="ignore",
        populate_by_name=True,
    )
    embedding_model: Literal["text-embedding-3-small", "text-embedding-3-large"] = (
        "text-embedding-3-small"
    )
    dimensions: int = Field(default=512, ge=16, le=1536)
    embedding_batch_size: int = Field(default=32, ge=1, le=32)
    max_embedding_calls: int = Field(default=8, ge=1, le=64)
    min_similarity: float = Field(default=0.2, ge=-1, le=1)

    def transport_settings(self) -> EmbeddingSettings:
        return _TransportSettings(
            api_key=self.api_key,
            embedding_model=self.embedding_model,
            dimensions=self.dimensions,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
        )
