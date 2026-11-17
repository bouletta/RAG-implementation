from typing import Protocol

from basic_rag_agent.llm import LazyModel
from basic_rag_agent.models import Query
from chroma_rag_agent.llm import LazyEmbedder
from parent_document_agent.models import Result
from parent_document_agent.service import run
from parent_document_agent.store import ParentStore

from shared.errors import AgentError

from .config import Settings


class Engine(Protocol):
    def start(self) -> None: ...
    def query(self, question: str, retrieve_only: bool) -> Result: ...


class RetrievalEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = ParentStore(settings.store_path)

    def _verify(self) -> None:
        hierarchy = self.store.load_hierarchy(self.settings.snapshot_name, self.settings)
        if hierarchy.sha256 != self.settings.expected_hierarchy_sha256:
            raise AgentError("Snapshot differs from the operator-approved hierarchy.")

    def start(self) -> None:
        self._verify()
        if not self.settings.api_key or not self.settings.api_key.get_secret_value().strip():
            raise AgentError("OPENAI_API_KEY is required to start the live service.")

    def query(self, question: str, retrieve_only: bool) -> Result:
        self._verify()
        settings = self.settings
        embedding = LazyEmbedder(settings)
        expansion = LazyModel(
            settings.model_copy(
                update={
                    "max_output_tokens": settings.expansion_output_tokens,
                }
            )
        )
        compression = LazyModel(
            settings.model_copy(
                update={
                    "max_output_tokens": settings.compression_output_tokens,
                }
            )
        )
        generation = LazyModel(settings)
        try:
            result = run(
                self.store,
                settings.snapshot_name,
                Query(question=question),
                settings,
                embedding,
                expansion_model=expansion,
                compression_model=compression,
                answer_model=generation,
                retrieve_only=retrieve_only,
                live=True,
            )
            if result.hierarchy_sha256 != settings.expected_hierarchy_sha256:
                raise AgentError("Snapshot changed during the request.")
            return result
        finally:
            embedding.close()
            expansion.close()
            compression.close()
            generation.close()
