from pathlib import Path
from typing import Literal

from multi_query_rag_agent.config import Settings as RetrievalSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class Settings(RetrievalSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        extra="ignore",
        populate_by_name=True,
    )
    max_model_calls: int = Field(default=3, ge=0, le=32)
    compression_mode: Literal["llm", "lexical"] = "llm"
    unit_chars: int = Field(default=800, ge=64, le=2000)
    max_units: int = Field(default=80, ge=1, le=128)
    max_selected_units: int = Field(default=12, ge=1, le=24)
    compression_context_chars: int = Field(default=32000, ge=1000, le=100000)
    compression_output_tokens: int = Field(default=1200, ge=100, le=2400)
