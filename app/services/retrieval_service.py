import logging
import time
from groq import Groq
from app.config import settings
from app.core.vector_store import VectorStore
from app.services.embedding_service import EmbeddingService
from app.core.langfuse_client import get_langfuse_client
from app.core.cache import CacheService

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        cache_service: CacheService,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.cache = cache_service
        self.groq = Groq(api_key=settings.groq_api_key)
        self.langfuse = get_langfuse_client()

    def query(self, question: str, top_k: int = 5) -> dict:
        total_start = time.perf_counter()

        # -------------------------------------------------- #
        # Layer 1 — Exact query cache
        # Fastest possible check — hash(question) lookup
        # Skips the entire pipeline on hit
        # -------------------------------------------------- #
        cached = self.cache.get_exact(question)
        if cached:
            cached["trace_id"] = "exact_cache_hit"
            cached["total_latency_ms"] = round(
                (time.perf_counter() - total_start) * 1000
            )
            logger.info(f"[Retrieval] Exact cache hit for: {question[:50]}")
            return cached

        with self.langfuse.start_as_current_observation(
            name="rag_query",
            as_type="chain",
            input={"question": question},
        ):
            self.langfuse.set_current_trace_io(
                input={"question": question},
            )

            # -------------------------------------------------- #
            # Step 1 — embed query (dense + sparse)
            # -------------------------------------------------- #
            with self.langfuse.start_as_current_observation(
                name="embed_query",
                as_type="embedding",
                input={"question": question},
            ):
                t0 = time.perf_counter()
                dense_vector  = self.embedding_service.embed_query(question)
                sparse_vector = self.embedding_service.sparse_embed_query(question)
                ms = round((time.perf_counter() - t0) * 1000)
                self.langfuse.update_current_span(output={
                    "latency_ms": ms,
                    "dense_dim":  len(dense_vector),
                    "sparse_nnz": len(sparse_vector),
                })
                logger.info(
                    f"[Retrieval] Embed: {ms}ms | "
                    f"dense={len(dense_vector)}d "
                    f"sparse={len(sparse_vector)} terms"
                )

            # -------------------------------------------------- #
            # Layer 2 — Semantic query cache
            # Checks if a semantically similar question was
            # already answered (cosine_sim > 0.95)
            # Skips retrieval + rerank + LLM on hit
            # -------------------------------------------------- #
            cached = self.cache.get_semantic(dense_vector, threshold=0.95)
            if cached:
                cached["trace_id"] = "semantic_cache_hit"
                cached["total_latency_ms"] = round(
                    (time.perf_counter() - total_start) * 1000
                )
                logger.info(f"[Retrieval] Semantic cache hit for: {question[:50]}")
                self.langfuse.set_current_trace_io(
                    output={"answer": cached["answer"], "cache": "semantic_hit"}
                )
                self.langfuse.flush()
                return cached

            # -------------------------------------------------- #
            # Step 2 — hybrid search (dense + sparse + RRF)
            # retrieves top-20 candidates for reranker
            # -------------------------------------------------- #
            with self.langfuse.start_as_current_observation(
                name="hybrid_retrieve",
                as_type="retriever",
                input={"top_k": 20},
            ):
                t0 = time.perf_counter()
                candidates = self.vector_store.search_children(
                    query_dense_vector=dense_vector,
                    query_sparse_vector=sparse_vector,
                    top_k=20,
                )
                ms = round((time.perf_counter() - t0) * 1000)
                self.langfuse.update_current_span(output={
                    "latency_ms":     ms,
                    "num_candidates": len(candidates),
                    "top_rrf_score":  candidates[0]["score"] if candidates else 0,
                    "sections":       list({c["section"] for c in candidates[:5]}),
                })
                logger.info(
                    f"[Retrieval] Hybrid retrieve: {ms}ms | "
                    f"{len(candidates)} candidates | "
                    + (f"top rrf: {candidates[0]['score']:.4f}"
                       if candidates else "no results")
                )

            if not candidates:
                self.langfuse.set_current_trace_io(
                    output={"answer": "No relevant content found."}
                )
                self.langfuse.flush()
                return {
                    "answer":           "I could not find relevant content.",
                    "sources":          [],
                    "trace_id":         self.langfuse.get_current_trace_id() or "",
                    "context_used":     0,
                    "total_latency_ms": round(
                        (time.perf_counter() - total_start) * 1000
                    ),
                }

            # -------------------------------------------------- #
            # Step 3 — rerank top-20 → top-5
            # Layer 3 — Rerank cache checked first
            # Cache key: hash(question + sorted chunk_ids)
            # Saves ~1800ms on repeated/similar queries
            # -------------------------------------------------- #
            chunk_ids = [
                c.get("child_id", c.get("parent_id", ""))
                for c in candidates
            ]

            with self.langfuse.start_as_current_observation(
                name="rerank",
                as_type="chain",
            ):
                t0 = time.perf_counter()

                cached_rerank = self.cache.get_rerank(question, chunk_ids)
                if cached_rerank:
                    reranked = cached_rerank
                    logger.info("[Retrieval] Rerank cache hit")
                else:
                    reranked = self.embedding_service.rerank(
                        query=question,
                        candidates=candidates,
                        top_n=top_k,
                    )
                    self.cache.set_rerank(question, chunk_ids, reranked)

                ms = round((time.perf_counter() - t0) * 1000)
                self.langfuse.update_current_span(output={
                    "latency_ms":       ms,
                    "top_rerank_score": reranked[0]["rerank_score"] if reranked else 0,
                    "top_sections":     [r["section"] for r in reranked[:3]],
                    "cache_hit":        cached_rerank is not None,
                })
                logger.info(
                    f"[Retrieval] Rerank: {ms}ms | "
                    + (f"top score: {reranked[0]['rerank_score']:.4f}"
                       if reranked else "empty")
                )

            # -------------------------------------------------- #
            # Step 4 — expand parents
            # Layer 4 — Parent chunk cache
            # Parent texts never change between ingestions
            # Cache permanently (TTL=7 days)
            # -------------------------------------------------- #
            with self.langfuse.start_as_current_observation(
                name="expand_parents",
                as_type="retriever",
            ):
                t0 = time.perf_counter()
                parent_ids  = list({r["parent_id"] for r in reranked})
                parent_docs = []

                for pid in parent_ids:
                    # check parent cache first
                    cached_parent = self.cache.get_parent(pid)
                    if cached_parent:
                        parent_docs.append(cached_parent)
                        logger.info(f"[Retrieval] Parent cache hit: {pid[:16]}")
                    else:
                        # fetch from Qdrant and cache it
                        fetched = self.vector_store.get_parents_by_ids([pid])
                        if fetched:
                            self.cache.set_parent(pid, fetched[0])
                            parent_docs.append(fetched[0])

                ms = round((time.perf_counter() - t0) * 1000)
                self.langfuse.update_current_span(output={
                    "latency_ms":  ms,
                    "num_parents": len(parent_docs),
                    "chapters":    [p["chapter"] for p in parent_docs],
                })
                logger.info(
                    f"[Retrieval] Expand: {ms}ms | {len(parent_docs)} parents"
                )

            # -------------------------------------------------- #
            # Build context from reranked child chunks
            # -------------------------------------------------- #
            context = "\n\n---\n\n".join(
                f"[{r['section']} | {r['chapter']} | p.{r['page']}]\n{r['text']}"
                for r in reranked
            )

            prompt_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert computer science tutor specialising "
                        "in algorithms and data structures. "
                        "Answer the question using ONLY the provided context. "
                        "Be precise, reference algorithm names and complexity "
                        "where relevant. "
                        "If the answer is not in the context, say: "
                        "'I don't have enough context to answer this.'"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ]

            # -------------------------------------------------- #
            # Step 5 — LLM generation via Groq
            # -------------------------------------------------- #
            with self.langfuse.start_as_current_observation(
                name="llm_generation",
                as_type="generation",
                model=settings.groq_model,
                input=prompt_messages,
            ):
                t0 = time.perf_counter()
                response = self.groq.chat.completions.create(
                    model=settings.groq_model,
                    messages=prompt_messages,
                    temperature=0.1,
                    max_tokens=1024,
                )
                answer = response.choices[0].message.content
                usage  = response.usage
                ms = round((time.perf_counter() - t0) * 1000)
                self.langfuse.update_current_generation(
                    output=answer,
                    usage_details={
                        "input":  usage.prompt_tokens,
                        "output": usage.completion_tokens,
                        "total":  usage.total_tokens,
                    },
                    metadata={"latency_ms": ms},
                )
                logger.info(
                    f"[Retrieval] LLM: {ms}ms | "
                    f"tokens {usage.prompt_tokens}p + {usage.completion_tokens}c"
                )

            # -------------------------------------------------- #
            # Build sources + finalise trace
            # -------------------------------------------------- #
            sources = []
            seen = set()
            for r in reranked:
                if r["section"] not in seen:
                    seen.add(r["section"])
                    sources.append({
                        "section":      r["section"],
                        "chapter":      r["chapter"],
                        "page":         r["page"],
                        "rrf_score":    round(r["score"], 4),
                        "rerank_score": round(r["rerank_score"], 4),
                    })

            total_ms = round((time.perf_counter() - total_start) * 1000)
            trace_id = self.langfuse.get_current_trace_id() or ""

            result = {
                "answer":           answer,
                "sources":          sources,
                "trace_id":         trace_id,
                "context_used":     len(parent_docs),
                "total_latency_ms": total_ms,
            }

            # -------------------------------------------------- #
            # Store in both query caches for future hits
            # Exact cache  — same question string
            # Semantic cache — similar question vectors
            # -------------------------------------------------- #
            self.cache.set_exact(question, result)
            self.cache.set_semantic(question, dense_vector, result)

            self.langfuse.set_current_trace_io(
                output={
                    "answer":           answer,
                    "total_tokens":     usage.total_tokens,
                    "total_latency_ms": total_ms,
                }
            )
            self.langfuse.flush()

            return result