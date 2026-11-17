import pytest
from parent_document_agent.hierarchy import expand, validate

from shared.errors import AgentError


def test_exact_parent_slices_and_deterministic_expansion(hierarchy):
    validate(hierarchy)
    parents = {p.id: p for p in hierarchy.parents.corpus.passages}
    children = {p.id: p for p in hierarchy.children.corpus.passages}
    assert all(p.startswith("p_") for p in parents)
    assert all(c.startswith("c_") for c in children)
    for link in hierarchy.links:
        assert children[link.child_id].text in parents[link.parent_id].text
    ids = [link.child_id for link in hierarchy.links]
    hits = expand(hierarchy, ids, 1)
    assert len(hits) == 1
    assert len(hits[0].child_ids) > 1
    assert hits[0].parent_id == hierarchy.links[0].parent_id


@pytest.mark.parametrize("ids", [["missing"], ["duplicate", "duplicate"]])
def test_unknown_and_duplicate_hits_rejected(hierarchy, ids):
    with pytest.raises(AgentError):
        expand(hierarchy, ids, 4)


def test_missing_mapping_rejected(hierarchy):
    hierarchy.links.pop()
    with pytest.raises(AgentError):
        validate(hierarchy)


def test_swapped_parent_rejected(hierarchy):
    link = hierarchy.links[0]
    other = hierarchy.parents.corpus.passages[-1].id
    hierarchy.links[0] = link.model_copy(update={"parent_id": other})
    with pytest.raises(AgentError):
        validate(hierarchy)


def test_parent_recovers_quote_crossing_child_boundary(hierarchy):
    parent = hierarchy.parents.corpus.passages[0]
    # A quote wider than a child cannot be supplied by child-only retrieval.
    quote = parent.text[:100]
    assert not any(quote in c.text for c in hierarchy.children.corpus.passages)
    hit = expand(hierarchy, [hierarchy.links[0].child_id], 1)[0]
    assert hit.parent_id == parent.id and quote in parent.text


def test_reopening_and_idempotent_ingestion(ready, settings, hierarchy, provider):
    assert ready.load_hierarchy("fixture-v1", settings) == hierarchy
    assert ready.ingest_hierarchy("fixture-v1", hierarchy, settings, provider).reused
    assert provider.calls == []


def test_corrupt_metadata_rejected(ready, settings):
    collection = ready.client.get_collection("fixture-v1", embedding_function=None)
    collection.modify(metadata={**collection.metadata, "hierarchy_sha256": "0" * 64})
    with pytest.raises(AgentError):
        ready.load_hierarchy("fixture-v1", settings)
