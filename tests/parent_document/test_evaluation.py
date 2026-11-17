import pytest
from chunking_agent.evaluation import Case, Gold
from parent_document_agent.evaluation import evaluate

from shared.errors import AgentError


def test_same_hits_recover_boundary_evidence(ready, hierarchy, settings, provider):
    parent = hierarchy.parents.corpus.passages[0]
    cases = [
        Case(
            id="boundary",
            question="Timeouts",
            evidence=Gold(document_name=parent.title, quote=parent.text[:100]),
        )
    ]
    result = evaluate(ready, "fixture-v1", cases, settings, provider)
    assert result.scores[0].child_gold is False
    assert result.scores[0].parent_gold is True
    assert result.logical_model_calls == 0
    assert result.embedding_usage.logical_calls == 1
    assert provider.calls == [["Timeouts"]]


def test_unanswerable_context_is_not_answer_quality(ready, settings, provider):
    result = evaluate(
        ready,
        "fixture-v1",
        [Case(id="none", question="unknown", evidence=None)],
        settings,
        provider,
    )
    assert result.scores[0].child_gold is None
    assert result.scores[0].parent_context_false_positive is True


@pytest.mark.parametrize("failure", ["budget", "gold", "duplicates", "compression"])
def test_evaluation_preflight_without_paid_calls(ready, settings, provider, failure):
    cases = [Case(id="case", question="Timeouts", evidence=None)]
    if failure == "budget":
        settings.max_embedding_calls = 0
    elif failure == "gold":
        cases[0] = Case(
            id="case",
            question="Timeouts",
            evidence=Gold(document_name="guide.txt", quote="invented gold"),
        )
    elif failure == "duplicates":
        cases *= 2
    else:
        settings.compress_parents = True
    with pytest.raises(AgentError):
        evaluate(ready, "fixture-v1", cases, settings, provider)
    assert provider.calls == []
