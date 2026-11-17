import argparse
from pathlib import Path

from basic_rag_agent.llm import LazyModel
from basic_rag_agent.models import Query
from chroma_rag_agent.llm import LazyEmbedder
from chroma_rag_agent.store import Store

from shared.cli import execute

from .config import Settings
from .models import HybridAnswer
from .service import run


def main() -> int:
    parser = argparse.ArgumentParser(description="Fuse keyword and semantic document rankings.")
    parser.add_argument("name")
    parser.add_argument("question")
    parser.add_argument("--store", type=Path)
    parser.add_argument("--retrieve-only", action="store_true")
    args = parser.parse_args()

    def action() -> HybridAnswer:
        settings = Settings()
        store = Store(args.store or settings.store_path)
        provider = LazyEmbedder(settings)
        model = LazyModel(settings)
        try:
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
