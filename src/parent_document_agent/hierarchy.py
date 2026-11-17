from basic_rag_agent.models import Corpus, Passage
from chroma_rag_agent.store import validate_build
from chunking_agent.chunking import build, spans
from chunking_agent.models import Build, Chunk, Document, Policy

from shared.errors import AgentError
from shared.structured import fingerprint

from .config import Settings
from .models import Hierarchy, Link, ParentHit


def validate(hierarchy: Hierarchy) -> None:
    validate_build(hierarchy.parents)
    validate_build(hierarchy.children)
    if hierarchy.parents.manifest.documents != hierarchy.children.manifest.documents:
        raise AgentError("Parent and child document identities differ.")
    documents = {document.id: document for document in hierarchy.parents.manifest.documents}
    if len(documents) != len(hierarchy.parents.manifest.documents):
        raise AgentError("Duplicate document identities in hierarchy metadata.")
    for built, prefix in ((hierarchy.parents, "p_"), (hierarchy.children, "c_")):
        for passage, chunk in zip(built.corpus.passages, built.manifest.chunks, strict=True):
            document = documents.get(passage.document_id)
            expected_id = (
                prefix
                + fingerprint(
                    {
                        "document_id": passage.document_id,
                        "start": chunk.start,
                        "end": chunk.end,
                    }
                )[:32]
            )
            if (
                document is None
                or not 0 <= chunk.start < chunk.end <= document.characters
                or passage.id != expected_id
                or passage.title != document.name
                or passage.locator != f"characters [{chunk.start}, {chunk.end})"
            ):
                raise AgentError(
                    "Hierarchy provenance does not match the declared document ranges."
                )
    parents = {p.id: p for p in hierarchy.parents.corpus.passages}
    children = {p.id: p for p in hierarchy.children.corpus.passages}
    parent_ranges = {c.id: c for c in hierarchy.parents.manifest.chunks}
    child_ranges = {c.id: c for c in hierarchy.children.manifest.chunks}
    if (
        len(hierarchy.links) != len(children)
        or {link.child_id for link in hierarchy.links} != children.keys()
    ):
        raise AgentError("Every child requires exactly one parent mapping.")
    if {link.parent_id for link in hierarchy.links} != parents.keys():
        raise AgentError("Hierarchy contains missing or unreferenced parents.")
    for link in hierarchy.links:
        child, parent = children[link.child_id], parents[link.parent_id]
        small, large = child_ranges[child.id], parent_ranges[parent.id]
        if (
            child.document_id != parent.document_id
            or small.document_id != child.document_id
            or large.document_id != parent.document_id
            or not 0 <= large.start <= small.start < small.end <= large.end
            or large.end - large.start != len(parent.text)
            or small.end - small.start != len(child.text)
            or child.text != parent.text[small.start - large.start : small.end - large.start]
        ):
            raise AgentError("Child ranges or text do not match the declared parent.")


def create(documents: tuple[Document, ...], settings: Settings) -> Hierarchy:
    parents = build(documents, Policy(size=settings.parent_size, overlap=0), settings.max_chunks)
    passages = tuple(p.model_copy(update={"id": "p_" + p.id[2:]}) for p in parents.corpus.passages)
    parent_corpus = Corpus(passages=passages)
    parents = Build(
        corpus=parent_corpus,
        manifest=parents.manifest.model_copy(
            update={
                "chunks": [
                    c.model_copy(update={"id": "p_" + c.id[2:]}) for c in parents.manifest.chunks
                ],
                "corpus_sha256": fingerprint(parent_corpus.model_dump(mode="json")),
            }
        ),
    )
    policy = Policy(size=settings.child_size, overlap=settings.child_overlap)
    children: list[Passage] = []
    chunks: list[Chunk] = []
    links: list[Link] = []
    covered = 0
    for parent, parent_range in zip(parents.corpus.passages, parents.manifest.chunks, strict=True):
        previous_end = 0
        for left, right in spans(parent.text, policy):
            if len(children) >= settings.max_chunks:
                raise AgentError("Child chunk budget exceeded; no partial hierarchy returned.")
            start, end = parent_range.start + left, parent_range.start + right
            id_ = (
                "c_"
                + fingerprint(
                    {
                        "document_id": parent.document_id,
                        "start": start,
                        "end": end,
                    }
                )[:32]
            )
            children.append(
                Passage(
                    id=id_,
                    document_id=parent.document_id,
                    title=parent.title,
                    locator=f"characters [{start}, {end})",
                    text=parent.text[left:right],
                )
            )
            chunks.append(Chunk(id=id_, document_id=parent.document_id, start=start, end=end))
            links.append(Link(child_id=id_, parent_id=parent.id))
            covered += right - max(left, previous_end)
            previous_end = right
    corpus = Corpus(passages=tuple(children))
    emitted = sum(len(p.text) for p in children)
    child_build = Build(
        corpus=corpus,
        manifest=parents.manifest.model_copy(
            update={
                "policy": policy,
                "chunks": chunks,
                "emitted_characters": emitted,
                "unique_covered_characters": covered,
                "duplicated_characters": emitted - covered,
                "corpus_sha256": fingerprint(corpus.model_dump(mode="json")),
            }
        ),
    )
    result = Hierarchy(parents=parents, children=child_build, links=links)
    validate(result)
    return result


def expand(hierarchy: Hierarchy, child_ids: list[str], limit: int) -> list[ParentHit]:
    validate(hierarchy)
    if not 1 <= limit <= 10 or len(child_ids) != len(set(child_ids)):
        raise AgentError("Invalid parent expansion bounds or duplicate child hits.")
    mapping = {link.child_id: link.parent_id for link in hierarchy.links}
    hits: dict[str, ParentHit] = {}
    for child_id in child_ids:
        if child_id not in mapping:
            raise AgentError("Retrieved child is absent from the hierarchy.")
        parent_id = mapping[child_id]
        if parent_id not in hits:
            hits[parent_id] = ParentHit(parent_id=parent_id, child_ids=[])
        hits[parent_id].child_ids.append(child_id)
    return list(hits.values())[:limit]
