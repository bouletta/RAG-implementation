import argparse
from typing import Any

from long_term_memory_agent.database import Database
from long_term_memory_agent.storage import FactStore
from pydantic import BaseModel

from shared.cli import execute
from shared.llm import OpenAIModel

from .config import Settings
from .embeddings import OpenAIEmbedder
from .llm import respond
from .models import Query, Result, SearchResult
from .retrieval import candidates, retrieve


def main() -> int:
    parser = argparse.ArgumentParser(description="Semantic retrieval over Agent 022 facts.")
    parser.add_argument("question")
    parser.add_argument("--profile", required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--preview", action="store_true", help="List eligible facts; no model calls."
    )
    modes.add_argument(
        "--search-only", action="store_true", help="Use embeddings but skip generation."
    )
    args = parser.parse_args()

    def action() -> BaseModel | dict[str, Any]:
        query = Query(profile=args.profile, question=args.question)
        settings = Settings()
        store = FactStore(Database(settings.database))
        eligible = candidates(store, query)
        if args.preview:
            return {
                "preview": True,
                "candidate_keys": [f.key for f in eligible],
                "embedding_model": settings.embedding_model,
                "dimensions": settings.dimensions,
            }
        if not eligible:
            return Result(
                retrieval=SearchResult(
                    matches=[],
                    embedding_model=settings.embedding_model,
                    dimensions=settings.dimensions,
                    embedding_tokens=0,
                ),
                answer=None,
            )
        embedder = OpenAIEmbedder(settings)
        try:
            search = retrieve(query, store, embedder, settings)
        finally:
            embedder.close()
        if args.search_only or not search.matches:
            return Result(retrieval=search, answer=None)
        model = OpenAIModel(settings)
        try:
            return respond(query, search, store, model, settings)
        finally:
            model.close()

    return execute(action)


if __name__ == "__main__":
    raise SystemExit(main())
