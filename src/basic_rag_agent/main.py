import argparse
from pathlib import Path

from shared.cli import execute

from .config import Settings
from .corpus import load_corpus
from .llm import LazyModel
from .models import Query, Result
from .service import run


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retrieve passages and generate evidence-linked claims."
    )
    parser.add_argument("corpus", type=Path)
    parser.add_argument("question")
    parser.add_argument(
        "--preview", action="store_true", help="Local retrieval only; no provider calls."
    )
    args = parser.parse_args()

    def action() -> Result:
        settings = Settings()
        corpus = load_corpus(args.corpus, settings.max_corpus_bytes)
        query = Query(question=args.question)
        model = LazyModel(settings)
        try:
            return run(corpus, query, settings, model=model, preview=args.preview)
        finally:
            model.close()

    return execute(action)


if __name__ == "__main__":
    raise SystemExit(main())
