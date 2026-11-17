import argparse
from pathlib import Path

from basic_rag_agent.llm import LazyModel
from basic_rag_agent.models import Query

from shared.cli import execute

from .chunking import build
from .config import Settings
from .ingestion import load_documents
from .models import Answer, Build
from .service import run


def main() -> int:
    parser = argparse.ArgumentParser(description="Traceable document chunking for RAG.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect", help="Print chunks and provenance without a model.")
    inspect.add_argument("files", type=Path, nargs="+")
    ask = subparsers.add_parser("ask", help="Answer using chunked documents.")
    ask.add_argument("question")
    ask.add_argument("files", type=Path, nargs="+")
    ask.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    def action() -> Answer | Build:
        settings = Settings()
        documents = load_documents(args.files, settings)
        if args.command == "inspect":
            return build(documents, settings.policy, settings.max_chunks)
        query = Query(question=args.question)
        model = LazyModel(settings)
        try:
            return run(documents, query, settings, model=model, preview=args.preview)
        finally:
            model.close()

    return execute(action)


if __name__ == "__main__":
    raise SystemExit(main())
