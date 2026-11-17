from pathlib import Path

from pydantic import Field
from pydantic_settings import SettingsConfigDict
from semantic_memory_agent.config import Settings as SemanticSettings

AGENTS = Path(__file__).resolve().parents[2]


class Settings(SemanticSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        extra="ignore",
        populate_by_name=True,
    )
    episodes_database: Path = AGENTS / "027_episodic_memory/data/episodes.sqlite3"
    profiles_database: Path = AGENTS / "028_user_profiles/data/profiles.sqlite3"
    top_k: int = Field(default=8, ge=1, le=8)
    fact_limit: int = Field(default=3, ge=1, le=5)
    episode_limit: int = Field(default=2, ge=1, le=3)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    max_context_chars: int = Field(default=16000, ge=1500, le=60000)
