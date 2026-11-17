from pathlib import Path

from context_compression_agent.config import Settings as CompressionSettings
from pydantic import Field, model_validator
from pydantic_settings import SettingsConfigDict


class Settings(CompressionSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        extra="ignore",
        populate_by_name=True,
    )
    store_path: Path = Path(__file__).resolve().parents[1] / "data" / "chroma"
    parent_size: int = Field(default=2000, ge=128, le=4000)
    child_size: int = Field(default=300, ge=64, le=1000)
    child_overlap: int = Field(default=40, ge=0)
    max_parents: int = Field(default=4, ge=1, le=10)
    compress_parents: bool = False

    @model_validator(mode="after")
    def valid_hierarchy_policy(self) -> "Settings":
        if self.child_size >= self.parent_size or self.child_overlap > self.child_size // 2:
            raise ValueError("Children must be smaller than parents, with overlap at most half.")
        return self
