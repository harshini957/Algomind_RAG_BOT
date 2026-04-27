import json
import hashlib
import os
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

    All methods are wrapped in try/except with self.enabled check.
    If Redis is unavailable the pipeline continues without caching —
    performance degrades but nothing breaks.
    """

    def __init__(self):
        redis_url = os.environ.get("REDIS_URL", "")
        try:
            if redis_url:
                self.client = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    ssl_cert_reqs=None,
                )
            else:
                self.client = redis.Redis(
                    host=settings.redis_host,
                    port=settings.redis_port,
                    decode_responses=True,
                )
            self.client.ping()
            self.enabled = True
            logger.info("[Cache] Redis connected")
        except Exception as e:
            self.enabled = False
            logger.warning(f"[Cache] Redis unavailable — caching disabled: {e}")

        self.semantic_index: list[dict] = []

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    # ── Layer 1: Exact query cache ──────────────────────────────

    def get_exact(self, question: str) -> Optional[dict]:
        if not self.enabled:
            return None
        try:
            key = f"exact:{self._hash(question.lower().strip())}"
            val = self.client.get(key)
            if val:
                logger.info(f"[Cache] Exact hit: {question[:40]}")
                return json.loads(val)
        except Exception:
            pass
        return None

    def set_exact(self, question: str, result: dict, ttl: int = 86400):
        if not self.enabled:
            return
        try:
            key = f"exact:{self._hash(question.lower().strip())}"
            cacheable = {k: v for k, v in result.items() if k != "trace_id"}
            self.client.setex(key, ttl, json.dumps(cacheable))
        except Exception:
            pass

    # ── Layer 2: Semantic query cache ───────────────────────────

    def get_semantic(
        self,
        query_vector: list[float],
        threshold: float = 0.95
    ) -> Optional[dict]:
        if not self.enabled:
            return None
        try:
            if not self.semantic_index:
                self._load_semantic_index()

            q = np.array(query_vector)
            best_score, best_result = 0.0, None

            for entry in self.semantic_index:
                cached_vec = np.array(entry["vector"])
                score = float(
                    np.dot(q, cached_vec) /
                    (np.linalg.norm(q) * np.linalg.norm(cached_vec) + 1e-9)
                )
                if score > best_score:
                    best_score = score
                    best_result = entry["result"]

            if best_score >= threshold:
                logger.info(
                    f"[Cache] Semantic hit: similarity={best_score:.4f}"
                )
                return best_result
        except Exception:
            pass
        return None

    def set_semantic(
        self,
        question: str,
        query_vector: list[float],
        result: dict,
        ttl: int = 86400
    ):
        if not self.enabled:
            return
        try:
            key = f"semantic:{self._hash(question.lower().strip())}"
            cacheable = {k: v for k, v in result.items() if k != "trace_id"}
            payload = json.dumps({
                "question": question,
                "vector":   query_vector,
                "result":   cacheable,
            })
            self.client.setex(key, ttl, payload)
            self.semantic_index.append({
                "vector": query_vector,
                "result": cacheable,
            })
        except Exception:
            pass

    def _load_semantic_index(self):
        if not self.enabled:
            return
        try:
            keys = self.client.keys("semantic:*")
            for key in keys:
                val = self.client.get(key)
                if val:
                    entry = json.loads(val)
                    self.semantic_index.append({
                        "vector": entry["vector"],
                        "result": entry["result"],
                    })
            logger.info(
                f"[Cache] Loaded {len(self.semantic_index)} semantic entries"
            )
        except Exception:
            pass

    # ── Layer 3: Rerank cache ────────────────────────────────────

    def get_rerank(
        self,
        question: str,
        chunk_ids: list[str]
    ) -> Optional[list]:
        if not self.enabled:
            return None
        try:
            key = f"rerank:{self._hash(question + '|' + ','.join(sorted(chunk_ids)))}"
            val = self.client.get(key)
            if val:
                logger.info("[Cache] Rerank hit")
                return json.loads(val)
        except Exception:
            pass
        return None

    def set_rerank(
        self,
        question: str,
        chunk_ids: list[str],
        reranked: list,
        ttl: int = 21600
    ):
        if not self.enabled:
            return
        try:
            key = f"rerank:{self._hash(question + '|' + ','.join(sorted(chunk_ids)))}"
            self.client.setex(key, ttl, json.dumps(reranked))
        except Exception:
            pass

    # ── Layer 4: Parent chunk cache ──────────────────────────────

    def get_parent(self, parent_id: str) -> Optional[dict]:
        if not self.enabled:
            return None
        try:
            key = f"parent:{parent_id}"
            val = self.client.get(key)
            if val:
                return json.loads(val)
        except Exception:
            pass
        return None

    def set_parent(self, parent_id: str, payload: dict, ttl: int = 604800):
        if not self.enabled:
            return
        try:
            key = f"parent:{parent_id}"
            self.client.setex(key, ttl, json.dumps(payload))
        except Exception:
            pass

    # ── Invalidation ─────────────────────────────────────────────

    def invalidate_all(self):
        if not self.enabled:
            return
        try:
            patterns = ["exact:*", "semantic:*", "rerank:*", "parent:*"]
            total = 0
            for pattern in patterns:
                keys = self.client.keys(pattern)
                if keys:
                    self.client.delete(*keys)
                    total += len(keys)
            self.semantic_index = []
            logger.info(f"[Cache] Invalidated {total} keys")
        except Exception:
            pass