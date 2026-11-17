from pathlib import Path
from typing import Literal

from basic_rag_agent.config import Settings as RAGSettings
from pydantic import Field, model_validator
from pydantic_settings import SettingsConfigDict

from .models import Policy


class Settings(RAGSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        extra="ignore",
        populate_by_name=True,
    )
    chunk_strategy: Literal["fixed", "boundary"] = "boundary"
    chunk_size: int = Field(default=600, ge=64, le=4000)
    chunk_overlap: int = Field(default=80, ge=0)
    max_document_bytes: int = Field(default=250_000, ge=1, le=1_000_000)
    max_total_bytes: int = Field(default=1_000_000, ge=1, le=5_000_000)
    max_chunks: int = Field(default=200, ge=1, le=200)

    @property
    def policy(self) -> Policy:
        return Policy(
            strategy=self.chunk_strategy, size=self.chunk_size, overlap=self.chunk_overlap
        )

    @model_validator(mode="after")
    def valid_policy(self) -> "Settings":
        _ = self.policy
        return self
