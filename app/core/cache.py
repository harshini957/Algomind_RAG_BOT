import json
import hashlib
import numpy as np
import redis
from typing import Optional
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class CacheService:
    """
    Redis-backed cache for the RAG pipeline.

    Four cache layers:
      1. exact_query    — md5(question) → full result dict
      2. semantic_query — embedding similarity → full result dict
      3. rerank         — hash(query+chunk_ids) → reranked order
      4. parent_chunk   — parent_id → chapter text
    """

    def __init__(self):
        self.client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True,
        )
        self.semantic_index: list[dict] = []  # in-memory index of cached queries
        logger.info("[Cache] Redis connected")

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    # ── Layer 1: Exact query cache ──────────────────────────────

    def get_exact(self, question: str) -> Optional[dict]:
        key = f"exact:{self._hash(question.lower().strip())}"
        val = self.client.get(key)
        if val:
            logger.info(f"[Cache] Exact hit: {question[:40]}")
            return json.loads(val)
        return None

    def set_exact(self, question: str, result: dict, ttl: int = 86400):
        key = f"exact:{self._hash(question.lower().strip())}"
        # don't cache trace_id — it's per-request
        cacheable = {k: v for k, v in result.items() if k != "trace_id"}
        self.client.setex(key, ttl, json.dumps(cacheable))

    # ── Layer 2: Semantic query cache ───────────────────────────

    def get_semantic(
        self,
        query_vector: list[float],
        threshold: float = 0.95
    ) -> Optional[dict]:
        if not self.semantic_index:
            self._load_semantic_index()

        q = np.array(query_vector)
        best_score, best_result = 0.0, None

        for entry in self.semantic_index:
            cached_vec = np.array(entry["vector"])
            score = float(np.dot(q, cached_vec) / (np.linalg.norm(q) * np.linalg.norm(cached_vec) + 1e-9))
            if score > best_score:
                best_score = score
                best_result = entry["result"]

        if best_score >= threshold:
            logger.info(f"[Cache] Semantic hit: similarity={best_score:.4f}")
            return best_result
        return None

    def set_semantic(self, question: str, query_vector: list[float], result: dict, ttl: int = 86400):
        key = f"semantic:{self._hash(question.lower().strip())}"
        cacheable = {k: v for k, v in result.items() if k != "trace_id"}
        payload = json.dumps({
            "question": question,
            "vector":   query_vector,
            "result":   cacheable,
        })
        self.client.setex(key, ttl, payload)
        # update in-memory index
        self.semantic_index.append({
            "vector": query_vector,
            "result": cacheable,
        })

    def _load_semantic_index(self):
        keys = self.client.keys("semantic:*")
        for key in keys:
            val = self.client.get(key)
            if val:
                try:
                    entry = json.loads(val)
                    self.semantic_index.append({
                        "vector": entry["vector"],
                        "result": entry["result"],
                    })
                except Exception:
                    pass
        logger.info(f"[Cache] Loaded {len(self.semantic_index)} semantic entries")

    # ── Layer 3: Rerank cache ────────────────────────────────────

    def get_rerank(self, question: str, chunk_ids: list[str]) -> Optional[list]:
        key = f"rerank:{self._hash(question + '|' + ','.join(sorted(chunk_ids)))}"
        val = self.client.get(key)
        if val:
            logger.info("[Cache] Rerank hit")
            return json.loads(val)
        return None

    def set_rerank(self, question: str, chunk_ids: list[str], reranked: list, ttl: int = 21600):
        key = f"rerank:{self._hash(question + '|' + ','.join(sorted(chunk_ids)))}"
        self.client.setex(key, ttl, json.dumps(reranked))

    # ── Layer 4: Parent chunk cache ──────────────────────────────

    def get_parent(self, parent_id: str) -> Optional[dict]:
        key = f"parent:{parent_id}"
        val = self.client.get(key)
        if val:
            return json.loads(val)
        return None

    def set_parent(self, parent_id: str, payload: dict, ttl: int = 604800):
        key = f"parent:{parent_id}"
        self.client.setex(key, ttl, json.dumps(payload))

    # ── Invalidation ─────────────────────────────────────────────

    def invalidate_all(self):
        """Call this when re-ingesting a PDF."""
        patterns = ["exact:*", "semantic:*", "rerank:*", "parent:*"]
        total = 0
        for pattern in patterns:
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
                total += len(keys)
        self.semantic_index = []
        logger.info(f"[Cache] Invalidated {total} keys")