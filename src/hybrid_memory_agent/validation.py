from shared.errors import AgentError
from shared.evidence import validate_evidence

from .models import Context, Reply


def validate(reply: Reply, context: Context) -> None:
    if not reply.uncertainty.strip():
        raise AgentError("The answer must state uncertainty.")
    source_text = {source.id: source.text for source in context.sources}
    for claim in reply.claims:
        if not claim.text.strip():
            raise AgentError("Claims must not be blank.")
        validate_evidence(claim.evidence, source_text)
