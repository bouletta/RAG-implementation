from chroma_rag_agent.models import IngestResult
from chroma_rag_agent.store import Store, validate_name
from pydantic import ValidationError
from semantic_memory_agent.embeddings import Embedder

from shared.errors import AgentError

from .config import Settings
from .hierarchy import validate
from .models import Hierarchy


class ParentStore(Store):
    def load_hierarchy(self, name: str, settings: Settings) -> Hierarchy:
        snapshot = self.load(name, settings)
        metadata = self._collection(name).metadata or {}
        raw = metadata.get("hierarchy")
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > 8_000_000:
            raise AgentError("Parent hierarchy is missing or exceeds its size limit.")
        try:
            hierarchy = Hierarchy.model_validate_json(raw)
        except ValidationError as exc:
            raise AgentError("Parent hierarchy metadata is invalid.") from exc
        validate(hierarchy)
        if (
            hierarchy.children != snapshot.build
            or metadata.get("hierarchy_sha256") != hierarchy.sha256
        ):
            raise AgentError("Parent hierarchy does not match its published child snapshot.")
        return hierarchy

    def ingest_hierarchy(
        self,
        name: str,
        hierarchy: Hierarchy,
        settings: Settings,
        embedder: Embedder,
    ) -> IngestResult:
        validate_name(name)
        validate(hierarchy)
        raw = hierarchy.model_dump_json()
        if len(raw.encode("utf-8")) > 8_000_000:
            raise AgentError("Parent hierarchy exceeds its size limit.")
        from chromadb.errors import NotFoundError

        try:
            existing = self.client.get_collection(name, embedding_function=None)
        except NotFoundError:
            existing = None
        except Exception as exc:
            raise AgentError("Could not inspect hierarchy publication state.") from exc
        if existing is not None:
            if self.load_hierarchy(name, settings) != hierarchy:
                raise AgentError("Snapshot belongs to another hierarchy; use a new name.")
            return self.ingest(name, hierarchy.children, settings, embedder)
        receipt = self.ingest(name, hierarchy.children, settings, embedder)
        if receipt.reused:
            raise AgentError("Another writer published this name; inspect before retrying.")
        try:
            collection = self._collection(name)
            collection.modify(
                metadata={
                    **(collection.metadata or {}),
                    "hierarchy": raw,
                    "hierarchy_sha256": hierarchy.sha256,
                }
            )
        except Exception as exc:
            raise AgentError("Parent publication failed; rebuild with a new name.") from exc
        return receipt
