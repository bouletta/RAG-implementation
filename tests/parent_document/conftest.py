import pytest
from chunking_agent.ingestion import load_documents
from parent_document_agent.config import Settings
from parent_document_agent.hierarchy import create
from parent_document_agent.store import ParentStore
from semantic_memory_agent.embeddings import Embedded


class Provider:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(texts)
        return Embedded(vectors=[[1.0] + [0.0] * 15 for _ in texts], tokens=10)


@pytest.fixture
def settings():
    return Settings(
        _env_file=None,
        api_key=None,
        dimensions=16,
        parent_size=256,
        child_size=64,
        child_overlap=16,
    )


@pytest.fixture
def provider():
    return Provider()


@pytest.fixture
def hierarchy(tmp_path, settings):
    path = tmp_path / "guide.txt"
    path.write_text(
        "Timeouts protect resources. " * 3
        + "Bulk requests have a separate timeout of sixty seconds. "
        + "Never log credentials. " * 8
    )
    return create(load_documents([path], settings), settings)


@pytest.fixture
def ready(tmp_path, settings, hierarchy, provider):
    store = ParentStore(tmp_path / "chroma")
    store.ingest_hierarchy("fixture-v1", hierarchy, settings, provider)
    provider.calls.clear()
    return store
