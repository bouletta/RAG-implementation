import argparse
from pathlib import Path

from basic_rag_agent.llm import LazyModel
from basic_rag_agent.models import Query
from chroma_rag_agent.llm import LazyEmbedder
from chroma_rag_agent.models import IngestResult
from chunking_agent.ingestion import load_documents

from shared.cli import execute

from .config import Settings
from .hierarchy import create
from .models import Result
from .service import run
from .store import ParentStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Search children; provide bounded parent context.")
    parser.add_argument("--store", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("name")
    ingest.add_argument("files", type=Path, nargs="+")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("name")
    query = commands.add_parser("query")
    query.add_argument("name")
    query.add_argument("question")
    query.add_argument("--single-query", action="store_true")
    query.add_argument("--compress", action="store_true")
    query.add_argument("--retrieve-only", action="store_true")
    args = parser.parse_args()

    def action() -> Result | IngestResult | dict[str, str | int]:
        settings = Settings()
        store = ParentStore(args.store or settings.store_path)
        if args.command == "inspect":
            hierarchy = store.load_hierarchy(args.name, settings)
            return {
                "hierarchy_sha256": hierarchy.sha256,
                "parents": len(hierarchy.parents.corpus.passages),
                "children": len(hierarchy.children.corpus.passages),
            }
        embedding = LazyEmbedder(settings)
        expansion = LazyModel(
            settings.model_copy(update={"max_output_tokens": settings.expansion_output_tokens})
        )
        compression = LazyModel(
            settings.model_copy(update={"max_output_tokens": settings.compression_output_tokens})
        )
        generation = LazyModel(settings)
        try:
            if args.command == "ingest":
                return store.ingest_hierarchy(
                    args.name,
                    create(load_documents(args.files, settings), settings),
                    settings,
                    embedding,
                )
            if args.compress:
                settings.compress_parents = True
            return run(
                store,
                args.name,
                Query(question=args.question),
                settings,
                embedding,
                expansion_model=expansion,
                compression_model=compression,
                answer_model=generation,
                provided=[] if args.single_query else None,
                retrieve_only=args.retrieve_only,
                live=True,
            )
        finally:
            embedding.close()
            expansion.close()
            compression.close()
            generation.close()

    return execute(action)


if __name__ == "__main__":
    raise SystemExit(main())
