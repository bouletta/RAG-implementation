from pathlib import Path

from embedding_rag_agent.config import Settings as EmbeddingSettings
from pydantic_settings import SettingsConfigDict


class Settings(EmbeddingSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        extra="ignore",
        populate_by_name=True,
    )
    store_path: Path = Path(__file__).resolve().parents[1] / "data" / "chroma"
