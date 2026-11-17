from pathlib import Path

from chroma_rag_agent.store import validate_name
from parent_document_agent.config import Settings as ParentSettings
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import SettingsConfigDict


class Settings(ParentSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        extra="ignore",
        populate_by_name=True,
    )
    service_key: SecretStr
    snapshot_name: str
    expected_hierarchy_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    max_concurrent: int = Field(default=2, ge=1, le=8)
    requests_per_minute: int = Field(default=30, ge=1, le=600)
    max_body_bytes: int = Field(default=8192, ge=1024, le=32768)
    body_timeout_seconds: float = Field(default=5, gt=0, le=30)
    response_deadline_seconds: float = Field(default=30, gt=0, le=300)
    compress_parents: bool = True

    @model_validator(mode="after")
    def valid_service(self) -> "Settings":
        if len(self.service_key.get_secret_value().strip()) < 32:
            raise ValueError("SERVICE_KEY must contain at least 32 characters.")
        validate_name(self.snapshot_name)
        return self
