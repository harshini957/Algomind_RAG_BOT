from fastapi import APIRouter, Depends, HTTPException
from app.api.schemas.query import QueryRequest, QueryResponse
from app.services.retrieval_service import RetrievalService
from app.dependencies import get_retrieval_service
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    svc:     RetrievalService = Depends(get_retrieval_service),
):
    """
    Ask a question about the indexed textbook.

    Pipeline:
        embed query (dense + sparse)
        → hybrid search Qdrant (RRF fusion)
        → rerank top-20 → top-5
        → expand parents
        → Groq LLM generation
        → return answer + sources + trace_id

    trace_id can be used to attach user feedback
    in Langfuse after the response is received.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:
        result = svc.query(
            question=request.question,
            top_k=request.top_k,
        )
        return QueryResponse(**result)

    except Exception as e:
        logger.error(f"[QueryRoute] Failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )