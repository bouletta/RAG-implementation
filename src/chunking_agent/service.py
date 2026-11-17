from basic_rag_agent.models import Query, Result
from basic_rag_agent.service import run as run_rag

from shared.structured import StructuredModel

from .chunking import build
from .config import Settings
from .models import Answer, Build, Document, QuoteLocation


def locate_quotes(result: Result, built: Build) -> list[QuoteLocation]:
    chunks = {chunk.id: chunk for chunk in built.manifest.chunks}
    documents = {document.id: document for document in built.manifest.documents}
    texts = {source.id: source.text for source in result.context.sources}
    locations = []
    if result.reply is not None:
        for index, claim in enumerate(result.reply.claims):
            for evidence in claim.evidence:
                chunk = chunks[evidence.source_id]
                text = texts[evidence.source_id]
                local = text.index(evidence.quote)
                locations.append(
                    QuoteLocation(
                        claim_index=index,
                        source_id=chunk.id,
                        document_id=chunk.document_id,
                        document_sha256=documents[chunk.document_id].sha256,
                        start=chunk.start + local,
                        end=chunk.start + local + len(evidence.quote),
                        ambiguous_in_chunk=text.find(evidence.quote, local + 1) >= 0,
                    )
                )
    return locations


def run(
    documents: tuple[Document, ...],
    query: Query,
    settings: Settings,
    *,
    model: StructuredModel | None = None,
    preview: bool = False,
) -> Answer:
    built = build(documents, settings.policy, settings.max_chunks)
    result = run_rag(built.corpus, query, settings, model=model, preview=preview)
    return Answer(manifest=built.manifest, rag=result, quote_locations=locate_quotes(result, built))
