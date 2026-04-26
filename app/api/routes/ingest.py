import os
import shutil
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from app.api.schemas.ingest import IngestResponse
from app.services.ingestion_service import IngestionService
from app.dependencies import get_ingestion_service
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(
    file: UploadFile = File(...),
    svc:  IngestionService = Depends(get_ingestion_service),
):
    """
    Upload a PDF and run the full ingestion pipeline:
    chunk → embed (dense + sparse) → store in Qdrant.

    Accepts: multipart/form-data with a PDF file.
    Returns: counts of parent and child chunks stored.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    # save uploaded file to disk
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"[IngestRoute] Saved upload: {save_path}")

    result = svc.ingest_pdf(save_path)

    if result.get("status") == "error":
        raise HTTPException(
            status_code=500,
            detail=result.get("message", "Ingestion failed")
        )

    return IngestResponse(**result)