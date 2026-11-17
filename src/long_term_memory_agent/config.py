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
    database: Path = Field(
        default=Path(__file__).resolve().parents[1] / "data/facts.sqlite3",
        validation_alias="FACTS_DATABASE",
    )
    context_chars: int = Field(default=16000, ge=1000, le=50000)
