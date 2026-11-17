import re
from collections.abc import Iterator

from basic_rag_agent.models import Corpus, Passage

from shared.errors import AgentError
from shared.structured import fingerprint

from .models import Build, Chunk, Document, DocumentInfo, Manifest, Policy


def spans(text: str, policy: Policy) -> Iterator[tuple[int, int]]:
    """Yield exact half-open character ranges, preserving all non-whitespace input."""
    start = 0
    while start < len(text):
        end = min(start + policy.size, len(text))
        if policy.strategy == "boundary" and end < len(text):
            lower = max(start + policy.size // 2, start + policy.overlap + 1)
            for pattern in (r"\r?\n\r?\n", r"\r?\n", r"\s"):
                boundaries = [m.end() for m in re.finditer(pattern, text[start:end])]
                viable = [start + point for point in boundaries if start + point >= lower]
                if viable:
                    end = viable[-1]
                    break
        if text[start:end].strip():
            yield start, end
        if end == len(text):
            break
        start = end - policy.overlap


def build(documents: tuple[Document, ...], policy: Policy, max_chunks: int = 200) -> Build:
    if not 1 <= len(documents) <= 20:
        raise AgentError("Build requires between 1 and 20 documents.")
    if not 1 <= max_chunks <= 200:
        raise AgentError("Chunk cap must be between 1 and 200.")
    if len({d.id for d in documents}) != len(documents):
        raise AgentError("Duplicate document identities are not allowed.")
    if len({d.name for d in documents}) != len(documents):
        raise AgentError("Document names must be unique.")
    passages: list[Passage] = []
    chunks: list[Chunk] = []
    covered = 0
    for document in sorted(documents, key=lambda item: item.id):
        previous_end = 0
        for start, end in spans(document.text, policy):
            if len(chunks) >= max_chunks:
                raise AgentError("Chunk budget exceeded; no partial corpus was returned.")
            chunk_id = (
                "c_"
                + fingerprint(
                    {
                        "document_id": document.id,
                        "start": start,
                        "end": end,
                    }
                )[:32]
            )
            chunks.append(Chunk(id=chunk_id, document_id=document.id, start=start, end=end))
            passages.append(
                Passage(
                    id=chunk_id,
                    document_id=document.id,
                    title=document.name,
                    locator=f"characters [{start}, {end})",
                    text=document.text[start:end],
                )
            )
            covered += end - max(start, previous_end)
            previous_end = end
    corpus = Corpus(passages=tuple(passages))
    emitted = sum(len(p.text) for p in passages)
    return Build(
        corpus=corpus,
        manifest=Manifest(
            policy=policy,
            documents=[
                DocumentInfo(id=d.id, name=d.name, sha256=d.sha256, characters=len(d.text))
                for d in sorted(documents, key=lambda item: item.id)
            ],
            chunks=chunks,
            source_characters=sum(len(d.text) for d in documents),
            emitted_characters=emitted,
            unique_covered_characters=covered,
            duplicated_characters=emitted - covered,
            corpus_sha256=fingerprint(corpus.model_dump(mode="json")),
        ),
    )
