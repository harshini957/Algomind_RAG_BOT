import logging
from app.core.chunker import DoclingChunker
from app.services.embedding_service import EmbeddingService
from app.core.vector_store import VectorStore
from app.core.cache import CacheService
from app.config import settings

logger = logging.getLogger(__name__)


class IngestionService:
    """
    Orchestrates the full ingestion pipeline:
        PDF → chunk → dense embed → sparse embed → store in Qdrant

    Cache invalidation happens at the start of every ingest.
    This ensures stale answers from a previous ingestion are
    never served after new content is indexed.

    Cache layers invalidated:
        - exact query cache    (hash(question) → answer)
        - semantic query cache (embedding → answer)
        - rerank cache         (hash(query+chunks) → scores)
        - parent chunk cache   (parent_id → chapter text)
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        cache_service: CacheService,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.cache = cache_service

    def ingest_pdf(self, pdf_path: str) -> dict:
        logger.info(f"[IngestionService] Starting: {pdf_path}")

        # -------------------------------------------------- #
        # Step 0 — Invalidate all caches before re-ingesting
        # If content changes, old cached answers become stale.
        # Always clear before indexing new content.
        # -------------------------------------------------- #
        try:
            self.cache.invalidate_all()
            logger.info("[IngestionService] All caches invalidated")
        except Exception as e:
            logger.warning(
                f"[IngestionService] Cache invalidation failed "
                f"(Redis may be down): {e}"
            )

        # -------------------------------------------------- #
        # Step 1 — Chunk the PDF
        # -------------------------------------------------- #
        chunker = DoclingChunker(
            child_max_tokens=settings.child_chunk_tokens,
            child_overlap=settings.child_overlap_tokens,
            source=pdf_path,
        )
        parents, children = chunker.chunk(pdf_path)

        if not parents or not children:
            return {
                "status":   "error",
                "message":  "Chunking produced no output",
                "parents":  0,
                "children": 0,
            }

        logger.info(
            f"[IngestionService] Chunked: "
            f"{len(parents)} parents, {len(children)} children"
        )

        child_texts = [c.text for c in children]

        # -------------------------------------------------- #
        # Step 2 — Dense embeddings
        # -------------------------------------------------- #
        logger.info("[IngestionService] Generating dense embeddings...")
        dense_embeddings = self.embedding_service.embed_documents(child_texts)
        logger.info(
            f"[IngestionService] Dense done: {len(dense_embeddings)} vectors"
        )

        # -------------------------------------------------- #
        # Step 3 — Sparse embeddings
        # -------------------------------------------------- #
        logger.info("[IngestionService] Generating sparse embeddings...")
        sparse_embeddings = self.embedding_service.sparse_embed_documents(child_texts)
        logger.info(
            f"[IngestionService] Sparse done: {len(sparse_embeddings)} vectors"
        )

        # -------------------------------------------------- #
        # Step 4 — Store parents in Qdrant
        # -------------------------------------------------- #
        logger.info("[IngestionService] Storing parents in Qdrant...")
        self.vector_store.store_parents(parents)

        # -------------------------------------------------- #
        # Step 5 — Store children with both vectors in Qdrant
        # -------------------------------------------------- #
        logger.info("[IngestionService] Storing children in Qdrant...")
        self.vector_store.store_children(
            children=children,
            dense_embeddings=dense_embeddings,
            sparse_embeddings=sparse_embeddings,
        )

        # -------------------------------------------------- #
        # Step 6 — Pre-warm parent chunk cache
        # Parent texts never change after ingestion.
        # Cache them immediately so the first queries
        # after ingestion get instant parent lookups
        # instead of hitting Qdrant.
        # -------------------------------------------------- #
        logger.info("[IngestionService] Pre-warming parent chunk cache...")
        cached_count = 0
        for parent in parents:
            try:
                self.cache.set_parent(
                    parent_id=parent.id,
                    payload={
                        "parent_id":   parent.id,
                        "text":        parent.text,
                        "chapter":     parent.chapter,
                        "chapter_num": parent.chapter_num,
                        "page_start":  parent.page_start,
                        "page_end":    parent.page_end,
                        "source":      parent.source,
                    },
                )
                cached_count += 1
            except Exception as e:
                logger.warning(
                    f"[IngestionService] Failed to cache parent "
                    f"{parent.id[:16]}: {e}"
                )

        logger.info(
            f"[IngestionService] Parent cache warm: "
            f"{cached_count}/{len(parents)} chapters cached"
        )

        result = {
            "status":          "success",
            "source":          pdf_path,
            "parents_stored":  len(parents),
            "children_stored": len(children),
            "dense_dim":       len(dense_embeddings[0]),
            "sparse_avg_nnz":  round(
                sum(len(s) for s in sparse_embeddings) / len(sparse_embeddings)
            ),
            "cache_warmed":    cached_count,
        }

        logger.info(f"[IngestionService] Done: {result}")
        return result