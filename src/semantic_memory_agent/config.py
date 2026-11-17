from pathlib import Path
from typing import Literal

from long_term_memory_agent.config import Settings as FactSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class Settings(FactSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        extra="ignore",
        populate_by_name=True,
    )
    embedding_model: Literal["text-embedding-3-small", "text-embedding-3-large"] = (
        "text-embedding-3-small"
    )
    dimensions: int = Field(default=512, ge=16, le=1536)
    top_k: int = Field(default=3, ge=1, le=8)
    min_similarity: float = Field(default=0.2, ge=-1, le=1)
