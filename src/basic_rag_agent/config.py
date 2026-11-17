from pathlib import Path

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from shared.config import RuntimeSettings


class Settings(RuntimeSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        extra="ignore",
        populate_by_name=True,
    )
    top_k: int = Field(default=4, ge=1, le=10)
    min_matched_terms: int = Field(default=1, ge=1, le=20)
    max_corpus_bytes: int = Field(default=1_000_000, ge=1000, le=5_000_000)
    max_context_chars: int = Field(default=16000, ge=1500, le=50000)
    max_output_tokens: int = Field(default=2000, ge=100, le=8000)
