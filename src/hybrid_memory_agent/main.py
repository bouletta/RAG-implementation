import argparse
from typing import Literal

from episodic_memory_agent.storage import EpisodeStore
from long_term_memory_agent.database import Database
from long_term_memory_agent.storage import FactStore
from profile_agent.storage import ProfileStore

from shared.cli import execute

from .config import Settings
from .graph import run
from .llm import Providers
from .models import Query, Result
from .retrieval import Stores


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hybrid recall over facts, episodes and preferences."
    )
    parser.add_argument("question")
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--lexical-only", action="store_true", help="Disable fact embeddings explicitly."
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--preview", action="store_true", help="Local lexical preview; no provider calls."
    )
    modes.add_argument(
        "--retrieve-only", action="store_true", help="Skip generation; embeddings may cost."
    )
    args = parser.parse_args()

    def action() -> Result:
        query = Query(
            profile=args.profile,
            question=args.question,
            mode="lexical" if args.preview or args.lexical_only else "hybrid",
        )
        settings = Settings()
        stores = Stores(
            facts=FactStore(Database(settings.database)),
            episodes=EpisodeStore(settings.episodes_database),
            profiles=ProfileStore(settings.profiles_database),
        )
        output: Literal["answer", "preview", "retrieve"] = (
            "preview" if args.preview else "retrieve" if args.retrieve_only else "answer"
        )
        providers = Providers(settings)
        try:
            return run(query, stores, providers, providers, settings, output=output)
        finally:
            providers.close()

    return execute(action)


if __name__ == "__main__":
    raise SystemExit(main())
