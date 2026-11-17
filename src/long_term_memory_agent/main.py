import argparse
from typing import Any

from pydantic import BaseModel

from shared.cli import execute
from shared.llm import OpenAIModel

from .agent import answer
from .config import Settings
from .database import Database
from .llm import build_content
from .models import Forget, Question, Remember, Scope
from .storage import FactStore, active


def main() -> int:
    parser = argparse.ArgumentParser(description="Explicit long-term memory across conversations.")
    parser.add_argument("--profile", required=True, help="Local namespace, not authentication.")
    commands = parser.add_subparsers(dest="command", required=True)
    remember = commands.add_parser("remember", help="Explicitly store or replace one fact.")
    remember.add_argument("key")
    remember.add_argument("value")
    remember.add_argument("--revision", help="Required to replace an existing fact.")
    remember.add_argument("--expires-at", help="Optional ISO time with UTC offset.")
    commands.add_parser("list", help="Inspect records, including expired ones.")
    forget = commands.add_parser("forget", help="Delete one record with its current revision.")
    forget.add_argument("key")
    forget.add_argument("--revision", required=True)
    ask = commands.add_parser("ask", help="Answer without modifying memory.")
    ask.add_argument("question")
    ask.add_argument(
        "--memory", action="append", default=[], help="Explicit key; repeat up to 8 times."
    )
    ask.add_argument(
        "--preview", action="store_true", help="Show provider input without an API call."
    )
    args = parser.parse_args()

    def action() -> BaseModel | dict[str, Any]:
        settings = Settings()
        scope = Scope(profile=args.profile)
        store = FactStore(Database(settings.database))
        if args.command == "remember":
            return store.remember(
                Remember(
                    profile=scope.profile,
                    key=args.key,
                    value=args.value,
                    expected_revision=args.revision,
                    expires_at=args.expires_at,
                )
            )
        if args.command == "forget":
            store.forget(
                Forget(profile=scope.profile, key=args.key, expected_revision=args.revision)
            )
            return {"forgotten": args.key, "profile": scope.profile}
        if args.command == "list":
            now = store.clock()
            return {
                "facts": [
                    {**fact.model_dump(mode="json"), "active": active(fact, now)}
                    for fact in store.list_facts(scope)
                ]
            }
        question = Question(profile=scope.profile, question=args.question, keys=args.memory)
        if args.preview:
            facts = store.select(question)
            return {
                "preview": True,
                "content": build_content(question, facts, settings.context_chars),
                "supplied_revisions": {fact.key: fact.revision for fact in facts},
            }
        model = OpenAIModel(settings)
        try:
            return answer(question, store, model, settings)
        finally:
            model.close()

    return execute(action)


if __name__ == "__main__":
    raise SystemExit(main())
