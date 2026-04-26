import logging
from fastapi import FastAPI
from app.api.routes import health, ingest, query, evaluate
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="CS RAG System",
        description=(
            "Production RAG system for The Algorithm Design Manual. "
            "Hybrid search (dense + sparse + RRF) with cross-encoder "
            "reranking, Langfuse observability, and Groq LLM generation."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # register all routes under /api/v1
    app.include_router(health.router,   prefix="/api/v1", tags=["Health"])
    app.include_router(ingest.router,   prefix="/api/v1", tags=["Ingestion"])
    app.include_router(query.router,    prefix="/api/v1", tags=["Query"])
    app.include_router(evaluate.router, prefix="/api/v1", tags=["Evaluation"])

    app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup():
        logger.info("CS RAG System starting up...")
        logger.info("Docs available at: http://localhost:8000/docs")

    return app


app = create_app()