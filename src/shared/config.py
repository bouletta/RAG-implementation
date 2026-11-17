"""Validated execution budgets; secrets are required only for live model calls."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    model: str = Field(default="gpt-4.1-mini", validation_alias="OPENAI_MODEL")
    timeout_seconds: float = Field(default=30, gt=0, le=120)
    max_retries: int = Field(default=2, ge=0, le=3)
    max_output_tokens: int = Field(default=1200, ge=100, le=8000)
    max_tool_calls: int = Field(default=5, ge=1, le=10)
    max_tool_chars: int = Field(default=16000, ge=500, le=50000)
    max_context_chars: int = Field(default=100000, ge=1000, le=200000)
