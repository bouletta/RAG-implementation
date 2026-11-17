import argparse
from pathlib import Path

from basic_rag_agent.llm import LazyModel
from basic_rag_agent.models import Query
from chunking_agent.chunking import build
from chunking_agent.ingestion import load_documents

from chroma_rag_agent.llm import LazyEmbedder
from shared.cli import execute

from .config import Settings
from .models import Answer, IngestResult
from .service import run
from .store import Store


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest once; query a persistent Chroma snapshot.")
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
    query.add_argument("--retrieve-only", action="store_true")
    args = parser.parse_args()

    def action() -> Answer | IngestResult | dict[str, object]:
        settings = Settings()
        store = Store(args.store or settings.store_path)
        if args.command == "inspect":
            snapshot = store.load(args.name, settings)
            return {
                "snapshot": snapshot.name,
                "passages": len(snapshot.build.corpus.passages),
                "corpus_sha256": snapshot.build.manifest.corpus_sha256,
                "embedding_model": snapshot.embedding_model,
                "dimensions": snapshot.dimensions,
            }
        provider = LazyEmbedder(settings)
        model = LazyModel(settings)
        try:
            if args.command == "ingest":
                built = build(
                    load_documents(args.files, settings), settings.policy, settings.max_chunks
                )
                return store.ingest(args.name, built, settings, provider)
            return run(
                store,
                args.name,
                Query(question=args.question),
                settings,
                provider,
                model=model,
                retrieve_only=args.retrieve_only,
                live=True,
            )
        finally:
            provider.close()
            model.close()

    return execute(action)


if __name__ == "__main__":
    raise SystemExit(main())
