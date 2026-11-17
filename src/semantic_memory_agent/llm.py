"""Reuse Agent 022's read-only answering boundary after semantic selection."""

from long_term_memory_agent.agent import answer
from long_term_memory_agent.llm import StructuredModel
from long_term_memory_agent.models import Question
from long_term_memory_agent.storage import FactStore, MemoryConflict

from .config import Settings
from .models import Query, Result, SearchResult


def respond(
    query: Query, search: SearchResult, store: FactStore, model: StructuredModel, settings: Settings
) -> Result:
    if not search.matches:
        return Result(retrieval=search, answer=None)
    result = answer(
        Question(
            profile=query.profile,
            question=query.question,
            keys=[m.fact.key for m in search.matches],
        ),
        store,
        model,
        settings,
    )
    if result.supplied_revisions != {m.fact.key: m.fact.revision for m in search.matches}:
        raise MemoryConflict("Memory changed between retrieval and answering; retry.")
    return Result(retrieval=search, answer=result)
