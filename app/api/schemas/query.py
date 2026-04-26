from pydantic import BaseModel
from typing import List


class QueryRequest(BaseModel):
    question: str
    top_k:    int = 5


class Source(BaseModel):
    section:      str
    chapter:      str
    page:         int
    rrf_score:    float
    rerank_score: float


class QueryResponse(BaseModel):
    answer:           str
    sources:          List[Source]
    trace_id:         str
    context_used:     int
    total_latency_ms: int