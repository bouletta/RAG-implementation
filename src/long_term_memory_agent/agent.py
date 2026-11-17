"""Read-only question answering over explicitly selected facts."""

from shared.errors import AgentError

from .config import Settings
from .llm import StructuredModel, build_content
from .models import Answer, Question, Reply
from .prompts import SYSTEM_PROMPT
from .storage import FactStore, MemoryConflict


def answer(
    question: Question, store: FactStore, model: StructuredModel, settings: Settings
) -> Answer:
    facts = store.select(question)
    content = build_content(question, facts, settings.context_chars)
    reply = model.structured(SYSTEM_PROMPT, content, Reply)
    supplied = {fact.key: fact.revision for fact in facts}
    if not reply.answer.strip() or len(set(reply.used_keys)) != len(reply.used_keys):
        raise AgentError("Model returned an empty answer or duplicate memory references.")
    if set(reply.used_keys) - supplied.keys():
        raise AgentError("Model cited memory that was not supplied.")
    current = store.select(question)
    if {fact.key: fact.revision for fact in current} != supplied:
        raise MemoryConflict("Selected memories changed during generation; ask again.")
    return Answer(reply=reply, supplied_revisions=supplied, model=settings.model)
