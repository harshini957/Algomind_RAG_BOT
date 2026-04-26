from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
    Filter,
    FieldCondition,
    MatchValue,
    Prefetch,
    FusionQuery,
    Fusion,
)
from app.config import settings
from app.core.chunker import ParentChunk, ChildChunk
import logging
import uuid

logger = logging.getLogger(__name__)

DENSE_VECTOR_NAME  = "dense"
SPARSE_VECTOR_NAME = "sparse"


class VectorStore:
    """
    Qdrant wrapper with hybrid search support.

    child_chunks collection has TWO vector spaces:
      - "dense"  : 768-dim cosine  — semantic search
      - "sparse" : SPLADE sparse   — keyword/BM25 search

    At query time both are searched simultaneously and
    fused with RRF in a single Qdrant API call.

    parent_chunks collection is payload-only.
    Retrieved by parent_id filter after child matches.
    """

    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        self._ensure_collections()

    def _ensure_collections(self):
        existing = [c.name for c in self.client.get_collections().collections]

        # --- parent_chunks: payload-only ---
        if settings.parent_collection not in existing:
            self.client.create_collection(
                collection_name=settings.parent_collection,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE)
            )
            logger.info(f"[VectorStore] Created: {settings.parent_collection}")
        else:
            logger.info(f"[VectorStore] Exists: {settings.parent_collection}")

        # --- child_chunks: dense + sparse hybrid ---
        if settings.child_collection not in existing:
            self.client.create_collection(
                collection_name=settings.child_collection,
                vectors_config={
                    DENSE_VECTOR_NAME: VectorParams(
                        size=settings.embedding_dim,
                        distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                }
            )
            logger.info(f"[VectorStore] Created hybrid: {settings.child_collection}")
        else:
            logger.info(f"[VectorStore] Exists: {settings.child_collection}")

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #

    def store_parents(self, parents: list[ParentChunk]) -> None:
        points = []
        for p in parents:
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=[0.0],
                payload={
                    "parent_id":   p.id,
                    "text":        p.text,
                    "chapter":     p.chapter,
                    "chapter_num": p.chapter_num,
                    "page_start":  p.page_start,
                    "page_end":    p.page_end,
                    "source":      p.source,
                }
            ))
        self.client.upsert(
            collection_name=settings.parent_collection,
            points=points
        )
        logger.info(f"[VectorStore] Stored {len(points)} parents")

    def store_children(
        self,
        children: list[ChildChunk],
        dense_embeddings: list[list[float]],
        sparse_embeddings: list[dict[int, float]],
    ) -> None:
        assert len(children) == len(dense_embeddings) == len(sparse_embeddings), \
            "children, dense_embeddings, sparse_embeddings must be same length"

        points = []
        for child, dense_vec, sparse_dict in zip(
            children, dense_embeddings, sparse_embeddings
        ):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    DENSE_VECTOR_NAME: dense_vec,
                    SPARSE_VECTOR_NAME: SparseVector(
                        indices=list(sparse_dict.keys()),
                        values=list(sparse_dict.values()),
                    )
                },
                payload={
                    "child_id":    child.id,
                    "parent_id":   child.parent_id,
                    "text":        child.text,
                    "chapter":     child.chapter,
                    "chapter_num": child.chapter_num,
                    "section":     child.section,
                    "section_num": child.section_num,
                    "page":        child.page,
                    "chunk_index": child.chunk_index,
                    "source":      child.source,
                }
            ))

        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i: i + batch_size]
            self.client.upsert(
                collection_name=settings.child_collection,
                points=batch
            )
            logger.info(
                f"[VectorStore] Stored batch "
                f"{i // batch_size + 1} ({len(batch)} points)"
            )
        logger.info(f"[VectorStore] Total children stored: {len(points)}")

    # ------------------------------------------------------------------ #
    # Retrieval — hybrid search with RRF fusion
    # ------------------------------------------------------------------ #

    def search_children(
        self,
        query_dense_vector: list[float],
        query_sparse_vector: dict[int, float],
        top_k: int = 20,
    ) -> list[dict]:
        """
        Hybrid search using RRF fusion.
        - dense prefetch  : semantic cosine search
        - sparse prefetch : BM25 keyword search
        - RRF fusion      : merges both ranked lists
        uses `using` parameter to specify which vector space
        """
        results = self.client.query_points(
            collection_name=settings.child_collection,
            prefetch=[
                Prefetch(
                    query=query_dense_vector,
                    using=DENSE_VECTOR_NAME,
                    limit=20,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=list(query_sparse_vector.keys()),
                        values=list(query_sparse_vector.values()),
                    ),
                    using=SPARSE_VECTOR_NAME,
                    limit=20,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return [
            {**hit.payload, "score": hit.score}
            for hit in results.points
        ]

    def get_parents_by_ids(self, parent_ids: list[str]) -> list[dict]:
        found = []
        for pid in parent_ids:
            results, _ = self.client.scroll(
                collection_name=settings.parent_collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="parent_id",
                            match=MatchValue(value=pid)
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            if results:
                found.append(results[0].payload)
        return found

    def get_collection_stats(self) -> dict:
        parent_info = self.client.get_collection(settings.parent_collection)
        child_info  = self.client.get_collection(settings.child_collection)
        return {
            "parent_chunks": parent_info.points_count,
            "child_chunks":  child_info.points_count,
        }