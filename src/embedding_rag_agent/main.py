import argparse
from collections.abc import Callable
from pathlib import Path

from basic_rag_agent.llm import LazyModel
from basic_rag_agent.models import Query
from chunking_agent.chunking import build
from chunking_agent.ingestion import load_documents
from chunking_agent.models import Answer as LexicalAnswer
from chunking_agent.service import run as lexical_run
from semantic_memory_agent.embeddings import OpenAIEmbedder

from shared.cli import execute

from .config import Settings
from .models import Answer, VectorData
from .search import ExactIndex, SearchIndex
from .service import run


def main(
    settings_type: type[Settings] = Settings,
    index_factory: Callable[[VectorData], SearchIndex] = ExactIndex,
    check_backend: Callable[[], None] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Embedding retrieval over traceable document chunks."
    )
    parser.add_argument("question")
    parser.add_argument("files", type=Path, nargs="+")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--lexical-preview", action="store_true", help="No-key 032 preview, not semantic search."
    )
    modes.add_argument(
        "--retrieve-only", action="store_true", help="Live embeddings without answer generation."
    )
    args = parser.parse_args()

    def action() -> Answer | LexicalAnswer:
        settings = settings_type()
        query = Query(question=args.question)
        documents = load_documents(args.files, settings)
        if args.lexical_preview:
            return lexical_run(documents, query, settings, preview=True)
        if check_backend is not None:
            check_backend()
        built = build(documents, settings.policy, settings.max_chunks)
        from .embeddings import preflight

        preflight(built.corpus, [query.question], settings)
        embedding = OpenAIEmbedder(settings.transport_settings())
        model = LazyModel(settings)
        try:
            return run(
                built,
                query,
                settings,
                embedding,
                model=model,
                retrieve_only=args.retrieve_only,
                live=True,
                index_factory=index_factory,
            )
        finally:
            embedding.close()
            model.close()

    return execute(action)


if __name__ == "__main__":
    raise SystemExit(main())
