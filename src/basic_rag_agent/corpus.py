from pathlib import Path

from shared.files import read_limited
from shared.structured import fingerprint

from .models import Corpus, Source


def load_corpus(path: Path, max_bytes: int) -> Corpus:
    """Read one bounded snapshot; locators are labels, never fetched paths."""
    return Corpus.model_validate_json(read_limited(path, max_bytes))


def sources(corpus: Corpus) -> list[Source]:
    return [
        Source(**p.model_dump(), sha256=fingerprint(p.model_dump(mode="json")))
        for p in corpus.passages
    ]
