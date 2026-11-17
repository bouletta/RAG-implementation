import argparse
from pathlib import Path

from basic_rag_agent.llm import LazyModel
from basic_rag_agent.models import Query
from chroma_rag_agent.llm import LazyEmbedder
from chroma_rag_agent.store import Store

from shared.cli import execute

from .config import Settings
from .models import Result
from .service import run


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bounded multi-query retrieval over a ready snapshot."
    )
    parser.add_argument("name")
    parser.add_argument("question")
    parser.add_argument("--store", type=Path)
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Skip the final answer, not generated query expansion.",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--variant", action="append", help="Use explicit variations instead of an LLM."
    )
    modes.add_argument("--single-query", action="store_true", help="Disable query expansion.")
    args = parser.parse_args()

    def action() -> Result:
        settings = Settings()
        store = Store(args.store or settings.store_path)
        embedding = LazyEmbedder(settings)
        expansion = LazyModel(
            settings.model_copy(
                update={
                    "max_output_tokens": settings.expansion_output_tokens,
                }
            )
        )
        generation = LazyModel(settings)
        try:
            return run(
                store,
                args.name,
                Query(question=args.question),
                settings,
                embedding,
                expansion_model=expansion,
                answer_model=generation,
                provided=[] if args.single_query else args.variant,
                retrieve_only=args.retrieve_only,
                live=True,
            )
        finally:
            embedding.close()
            expansion.close()
            generation.close()

    return execute(action)


if __name__ == "__main__":
    raise SystemExit(main())
