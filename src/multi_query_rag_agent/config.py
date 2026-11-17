from pathlib import Path

from hybrid_rag_agent.config import Settings as HybridSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class Settings(HybridSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        extra="ignore",
        populate_by_name=True,
    )
    max_variants: int = Field(default=3, ge=0, le=3)
    max_model_calls: int = Field(default=2, ge=0, le=32)
    expansion_context_chars: int = Field(default=8000, ge=1000, le=16000)
    expansion_output_tokens: int = Field(default=600, ge=100, le=1200)
