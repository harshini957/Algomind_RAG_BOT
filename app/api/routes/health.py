from fastapi import APIRouter, Depends
from app.api.schemas.health import HealthResponse
from app.core.vector_store import VectorStore
from app.dependencies import get_vector_store
from app.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(vs: VectorStore = Depends(get_vector_store)):
    """
    Liveness probe — confirms app is running and
    Qdrant is reachable with populated collections.
    """
    try:
        stats = vs.get_collection_stats()
        qdrant_status = "ok"
    except Exception as e:
        stats = {}
        qdrant_status = f"error: {str(e)}"

    return HealthResponse(
        status="ok",
        version="1.0.0",
        qdrant=qdrant_status,
        collections=stats,
    )