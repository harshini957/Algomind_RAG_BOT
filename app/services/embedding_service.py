# from sentence_transformers import SentenceTransformer
# from app.config import settings
# import logging

# logger = logging.getLogger(__name__)


# class EmbeddingService:
#     """
#     Wraps nomic-embed-text-v1 via sentence-transformers.

#     Nomic requires task-specific prefixes:
#       - Chunks/documents → "search_document: <text>"
#       - Queries          → "search_query: <text>"

#     This is the ONLY place in the codebase that touches
#     the embedding model. Both ingestion and retrieval use
#     this same instance — guaranteeing the vector spaces match.
#     """

#     def __init__(self):
#         logger.info(f"[EmbeddingService] Loading model: {settings.embedding_model}")
#         self.model = SentenceTransformer(
#             settings.embedding_model,
#             trust_remote_code=True      # required for nomic-embed-text-v1
#         )
#         self.dim = settings.embedding_dim
#         logger.info(f"[EmbeddingService] Model loaded. Dimension: {self.dim}")

#     def embed_document(self, text: str) -> list[float]:
#         """Embed a single chunk/document."""
#         prefixed = f"search_document: {text}"
#         vector = self.model.encode(prefixed, normalize_embeddings=True)
#         return vector.tolist()

#     def embed_documents(self, texts: list[str]) -> list[list[float]]:
#         """Embed a batch of chunks — used during ingestion."""
#         prefixed = [f"search_document: {t}" for t in texts]
#         vectors = self.model.encode(
#             prefixed,
#             normalize_embeddings=True,
#             batch_size=32,
#             show_progress_bar=True
#         )
#         return vectors.tolist()

#     def embed_query(self, query: str) -> list[float]:
#         """Embed a user query — used during retrieval."""
#         prefixed = f"search_query: {query}"
#         vector = self.model.encode(prefixed, normalize_embeddings=True)
#         return vector.tolist()

import logging
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding
from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Manages three models:

    1. Dense encoder  — nomic-embed-text-v1 (768-dim)
       Used for semantic similarity search.
       Requires task prefix: search_document / search_query.

    2. Sparse encoder — prithivida/Splade_PP_en_v1
       Used for BM25-style keyword search.
       Produces sparse vectors (dict of token_id → weight).
       Excellent at exact technical terms: BFS, Dijkstra, O(n log n).

    3. Cross-encoder reranker — ms-marco-MiniLM-L-6-v2
       Used AFTER retrieval to rerank top-20 candidates.
       Reads query + document together — much more accurate than bi-encoder.
       Only runs on top-20 candidates, not all 1703 chunks.
    """

    def __init__(self):
        # --- dense model ---
        logger.info(f"[EmbeddingService] Loading dense model: {settings.embedding_model}")
        self.dense_model = SentenceTransformer(
            settings.embedding_model,
            trust_remote_code=True
        )
        self.dim = settings.embedding_dim
        logger.info(f"[EmbeddingService] Dense model loaded. Dim: {self.dim}")

        # --- sparse model ---
        logger.info("[EmbeddingService] Loading sparse model: Splade_PP_en_v1")
        self.sparse_model = SparseTextEmbedding(
            model_name="prithivida/Splade_PP_en_v1"
        )
        logger.info("[EmbeddingService] Sparse model loaded")

        # --- cross-encoder reranker ---
        logger.info("[EmbeddingService] Loading reranker: ms-marco-MiniLM-L-6-v2")
        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        logger.info("[EmbeddingService] Reranker loaded")

    # ------------------------------------------------------------------ #
    # Dense methods
    # ------------------------------------------------------------------ #

    def embed_document(self, text: str) -> list[float]:
        """Embed a single chunk for indexing."""
        vector = self.dense_model.encode(
            f"search_document: {text}",
            normalize_embeddings=True
        )
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of chunks — used during ingestion."""
        prefixed = [f"search_document: {t}" for t in texts]
        vectors = self.dense_model.encode(
            prefixed,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=True
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a user query — used during retrieval."""
        vector = self.dense_model.encode(
            f"search_query: {query}",
            normalize_embeddings=True
        )
        return vector.tolist()

    # ------------------------------------------------------------------ #
    # Sparse methods
    # ------------------------------------------------------------------ #

    def sparse_embed_documents(
        self, texts: list[str]
    ) -> list[dict[int, float]]:
        """
        Generate sparse vectors for a batch of chunks.
        Returns list of {token_id: weight} dicts.
        Used during ingestion alongside dense vectors.
        """
        results = []
        for embedding in self.sparse_model.embed(texts, batch_size=32):
            # FastEmbed returns SparseEmbedding with .indices and .values
            sparse_dict = {
                int(idx): float(val)
                for idx, val in zip(embedding.indices, embedding.values)
            }
            results.append(sparse_dict)
        return results

    def sparse_embed_query(self, query: str) -> dict[int, float]:
        """
        Generate sparse vector for a single query.
        Used during retrieval.
        """
        embeddings = list(self.sparse_model.embed([query]))
        embedding = embeddings[0]
        return {
            int(idx): float(val)
            for idx, val in zip(embedding.indices, embedding.values)
        }

    # ------------------------------------------------------------------ #
    # Reranker
    # ------------------------------------------------------------------ #

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_n: int = 5
    ) -> list[dict]:
        """
        Rerank candidates using cross-encoder.

        candidates: list of child chunk dicts from hybrid search
        top_n: how many to return after reranking

        The cross-encoder reads query + chunk text together and
        produces a relevance score much more accurate than cosine sim.
        We run it only on the top-20 hybrid candidates — not all chunks.
        """
        if not candidates:
            return []

        # build (query, text) pairs for cross-encoder
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)

        # attach reranker score to each candidate
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        # sort by reranker score descending
        reranked = sorted(
            candidates,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked[:top_n]