import json

import pytest
from basic_rag_agent.models import Query, Reply
from parent_document_agent.service import run

from shared.errors import AgentError


def test_generation_receives_parent_ids(ready, settings, provider):
    class Model:
        def structured(self, instructions, content, schema):
            source = json.loads(content[0]["text"])["sources"][0]
            assert source["id"].startswith("p_")
            return Reply(
                claims=[
                    {
                        "text": "Timeouts protect resources.",
                        "evidence": [{"source_id": source["id"], "quote": source["text"][:20]}],
                    }
                ],
                limitation="Only provided evidence.",
            )

    result = run(
        ready,
        "fixture-v1",
        Query(question="timeout"),
        settings,
        provider,
        provided=[],
        answer_model=Model(),
    )
    assert result.answer.status == "answered"
    assert result.logical_model_calls == 1
    assert result.parents
    assert len(provider.calls) == 1


def test_budget_fails_before_embedding(ready, settings, provider):
    settings.max_model_calls = 0
    with pytest.raises(AgentError, match="budget"):
        run(ready, "fixture-v1", Query(question="timeout"), settings, provider)
    assert provider.calls == []


def test_retrieval_only_and_lexical_compression(ready, settings, provider):
    settings.compress_parents = True
    settings.compression_mode = "lexical"
    result = run(
        ready,
        "fixture-v1",
        Query(question="Timeouts"),
        settings,
        provider,
        provided=[],
        retrieve_only=True,
    )
    assert result.logical_model_calls == 0
    assert result.compression is not None
    assert all(s.id.startswith("u_") for s in result.answer.context.sources)


def test_child_citation_rejected_after_parent_expansion(ready, settings, provider):
    child_id = ready.load_hierarchy("fixture-v1", settings).children.corpus.passages[0].id

    class Model:
        def structured(self, instructions, content, schema):
            return Reply(
                claims=[
                    {"text": "Claim", "evidence": [{"source_id": child_id, "quote": "Timeouts"}]}
                ],
                limitation="Test",
            )

    with pytest.raises(AgentError):
        run(
            ready,
            "fixture-v1",
            Query(question="Timeouts"),
            settings,
            provider,
            provided=[],
            answer_model=Model(),
        )
