from shared.errors import AgentError
from shared.evidence import validate_evidence

from .models import Context, Reply


def validate_reply(reply: Reply, context: Context) -> None:
    """Validate provenance, not semantic entailment or factual truth."""
    if not reply.limitation.strip():
        raise AgentError("The response must state its limitations.")
    source_text = {source.id: source.text for source in context.sources}
    seen = set()
    for claim in reply.claims:
        normalized = " ".join(claim.text.casefold().split())
        if not normalized or normalized in seen:
            raise AgentError("Claims must be nonblank and nonduplicated.")
        seen.add(normalized)
        validate_evidence(claim.evidence, source_text)
