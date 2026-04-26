from functools import lru_cache
from app.services.embedding_service import EmbeddingService
from app.core.vector_store import VectorStore
from app.core.cache import CacheService
from app.services.ingestion_service import IngestionService
from app.services.retrieval_service import RetrievalService


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    return VectorStore()


@lru_cache(maxsize=1)
def get_cache_service() -> CacheService:
    return CacheService()


@lru_cache(maxsize=1)
def get_ingestion_service() -> IngestionService:
    return IngestionService(
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
        cache_service=get_cache_service(),
    )


@lru_cache(maxsize=1)
def get_retrieval_service() -> RetrievalService:
    return RetrievalService(
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
        cache_service=get_cache_service(),
    )