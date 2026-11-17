import re

from .config import Settings
from .corpus import sources
from .models import Corpus, Hit, Query


def terms(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))


def retrieve(corpus: Corpus, query: Query, settings: Settings) -> list[Hit]:
    """Rank distinct query/body term overlap, breaking ties by passage ID."""
    query_terms = terms(query.question)
    hits = []
    for source in sources(corpus):
        overlap = len(query_terms & terms(source.text))
        if overlap >= settings.min_matched_terms:
            hits.append(Hit(source=source, matched_terms=overlap))
    return sorted(hits, key=lambda hit: (-hit.matched_terms, hit.source.id))[: settings.top_k]
