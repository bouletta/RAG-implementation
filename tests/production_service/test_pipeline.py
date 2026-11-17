import json

import pytest
from basic_rag_agent.models import Reply
from chunking_agent.ingestion import load_documents
from fastapi.testclient import TestClient
from parent_document_agent.hierarchy import create
from parent_document_agent.store import ParentStore
from production_rag_agent.app import create_app
from production_rag_agent.config import Settings
from production_rag_agent.engine import RetrievalEngine
from semantic_memory_agent.embeddings import Embedded

from shared.errors import AgentError

KEY = "x" * 40


class Embedding:
    calls = 0

    def __init__(self, settings=None):
        pass

    def embed(self, texts):
        type(self).calls += 1
        return Embedded(vectors=[[1.0] + [0.0] * 15 for _ in texts], tokens=10)

    def close(self):
        pass


class Model:
    def __init__(self, settings):
        pass

    def structured(self, instructions, content, schema):
        source = json.loads(content[0]["text"])["sources"][0]
        return Reply(
            claims=[
                {
                    "text": "Timeouts are bounded.",
                    "evidence": [
                        {"source_id": source["id"], "quote": source["text"][:20]},
                    ],
                }
            ],
            limitation="Offline injected provider integration test.",
        )

    def close(self):
        pass


@pytest.fixture
def prepared(tmp_path):
    settings = Settings(
        _env_file=None,
        api_key="offline-test-not-a-real-key",
        dimensions=16,
        service_key=KEY,
        snapshot_name="fixture-v1",
        expected_hierarchy_sha256="a" * 64,
        store_path=tmp_path / "chroma",
        max_variants=0,
        compress_parents=False,
    )
    path = tmp_path / "guide.txt"
    path.write_text("Timeouts protect resources. Never log credentials.")
    hierarchy = create(load_documents([path], settings), settings)
    ParentStore(settings.store_path).ingest_hierarchy(
        "fixture-v1", hierarchy, settings, Embedding()
    )
    settings.expected_hierarchy_sha256 = hierarchy.sha256
    return settings


def test_real_http_to_chroma_to_citation_pipeline(prepared, monkeypatch):
    import production_rag_agent.engine as engine_module

    monkeypatch.setattr(engine_module, "LazyEmbedder", Embedding)
    monkeypatch.setattr(engine_module, "LazyModel", Model)
    with TestClient(create_app(prepared)) as client:
        response = client.post(
            "/v1/query", headers={"Authorization": "Bearer " + KEY}, json={"question": "Timeouts"}
        )
    assert response.status_code == 200
    answer = response.json()["answer"]
    assert answer["status"] == "answered"
    assert answer["reply"]["claims"][0]["evidence"][0]["source_id"].startswith("p_")


def test_wrong_operator_pin_fails_without_provider_calls(prepared):
    prepared.expected_hierarchy_sha256 = "b" * 64
    calls = Embedding.calls
    with pytest.raises(AgentError, match="operator-approved"):
        RetrievalEngine(prepared).start()
    assert Embedding.calls == calls


def test_missing_provider_key_fails_startup(prepared):
    prepared.api_key = None
    with pytest.raises(AgentError, match="OPENAI_API_KEY"):
        RetrievalEngine(prepared).start()
