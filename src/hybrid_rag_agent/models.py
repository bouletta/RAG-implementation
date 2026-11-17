from chroma_rag_agent.models import Answer
from pydantic import BaseModel


class FusedMatch(BaseModel):
    source_id: str
    rrf_score: float
    lexical_rank: int | None
    semantic_rank: int | None


class HybridAnswer(BaseModel):
    answer: Answer
    lexical_ids: list[str]
    semantic_ids: list[str]
    fusion: list[FusedMatch]
    candidate_window: int
    rrf_k: int
