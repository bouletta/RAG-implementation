from pathlib import Path

from chroma_rag_agent.config import Settings as ChromaSettings
from pydantic import Field, model_validator
from pydantic_settings import SettingsConfigDict


class Settings(ChromaSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        extra="ignore",
        populate_by_name=True,
    )
    candidate_window: int = Field(default=10, ge=1, le=10)
    rrf_k: int = Field(default=60, ge=1, le=1000)

    @model_validator(mode="after")
    def sufficient_candidates(self) -> "Settings":
        if self.candidate_window < self.top_k:
            raise ValueError("Candidate window must be at least top_k.")
        return self
