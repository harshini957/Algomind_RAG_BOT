from fastapi import APIRouter, Depends, HTTPException
from app.api.schemas.evaluate import EvalRequest, EvalResponse
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/evaluate", response_model=EvalResponse)
def evaluate(request: EvalRequest):
    """
    Run RAGAS evaluation on a batch of test cases.
    Skipped for now — returns placeholder response.
    Re-enable by wiring EvaluationService once
    RAGAS + Groq rate limit issue is resolved.
    """
    raise HTTPException(
        status_code=503,
        detail=(
            "Evaluation endpoint temporarily disabled. "
            "RAGAS requires n>1 completions which Groq free tier "
            "does not support. Upgrade to Groq Dev tier or "
            "provide an OpenAI key to enable this endpoint."
        )
    )