import logging
logging.basicConfig(level=logging.INFO)

from app.services.embedding_service import EmbeddingService
from app.core.vector_store import VectorStore
from app.core.cache import CacheService
from app.services.retrieval_service import RetrievalService

emb_service   = EmbeddingService()
vs            = VectorStore()
cache_service = CacheService()
svc           = RetrievalService(
    embedding_service=emb_service,
    vector_store=vs,
    cache_service=cache_service,
)

questions = [
    "What is the nearest neighbor heuristic and why does it fail?",
    "Explain BFS and its time complexity",
    "What is dynamic programming and how does memoization work?",
    # run first question again to verify cache hit
    "What is the nearest neighbor heuristic and why does it fail?",
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    print('='*60)

    result = svc.query(q, top_k=5)

    print(f"\nANSWER:\n{result['answer']}")
    print(f"\nSOURCES:")
    for s in result["sources"]:
        print(f"  [rrf={s['rrf_score']} | rerank={s['rerank_score']}]")
        print(f"  {s['section']} — {s['chapter']} p.{s['page']}")
    print(f"\nMETRICS:")
    print(f"  trace_id      : {result['trace_id']}")
    print(f"  context_used  : {result['context_used']} parent chunks")
    print(f"  total_latency : {result['total_latency_ms']}ms")