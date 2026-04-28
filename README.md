# AlgoMind — Production RAG System for The Algorithm Design Manual

> A production-grade Retrieval-Augmented Generation system built from scratch over the Algorithm Design Manual by Steven S. Skiena. Combines hybrid dense + sparse search, cross-encoder reranking, parent-child chunking, real-time observability, and multi-layer caching into a fully deployed FastAPI service on Google Cloud Run.

**Live API:** https://cs-rag-944198740984.us-central1.run.app  
**Swagger UI:** https://cs-rag-944198740984.us-central1.run.app/docs

---

## Problem Statement

Computer science students and professionals frequently need to look up algorithm concepts, complexity analysis, and implementation strategies from dense reference textbooks. Traditional keyword search fails on semantic queries ("why does the greedy approach fail here?") and semantic-only vector search fails on exact technical terms ("Dijkstra", "BFS", "O(n log n)").

**AlgoMind solves this by:**

- Parsing and indexing the full Algorithm Design Manual (739 pages, 19 chapters) into a structured parent-child chunk hierarchy that preserves section context
- Running hybrid dense + sparse search so both semantic similarity and exact keyword matching work simultaneously
- Reranking retrieved candidates with a cross-encoder for precision that bi-encoder embeddings alone cannot achieve
- Generating grounded answers through Groq's LLM with source citations back to specific chapters and pages
- Tracing every query end-to-end through Langfuse for latency visibility and quality monitoring

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT / UI                              │
│              AlgoMind Chat Interface (HTML/JS)                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTPS
┌─────────────────────────▼───────────────────────────────────────┐
│              Google Cloud Run (us-central1)                     │
│                     FastAPI Application                         │
│                                                                 │
│   POST /api/v1/ingest    POST /api/v1/query                     │
│   POST /api/v1/evaluate  GET  /api/v1/health                    │
└──────┬──────────────────────────┬──────────────────────────────┘
       │                          │
┌──────▼──────────┐    ┌──────────▼──────────────────────────────┐
│ INGESTION       │    │ RETRIEVAL SERVICE                        │
│ SERVICE         │    │                                          │
│                 │    │  1. Exact cache check  (Redis)           │
│ DoclingChunker  │    │  2. Embed query        (Nomic + SPLADE)  │
│  ├─ Parents     │    │  3. Semantic cache     (Redis cosine)    │
│  └─ Children    │    │  4. Hybrid search      (Qdrant RRF)      │
│                 │    │  5. Rerank cache       (Redis)           │
│ EmbeddingService│    │  6. Cross-encoder      (ms-marco)        │
│  ├─ Dense       │    │  7. Parent expansion   (Redis/Qdrant)    │
│  └─ Sparse      │    │  8. LLM generation     (Groq)            │
│                 │    │  9. Cache store        (Redis)           │
│ VectorStore     │    │  10. Langfuse trace    (observability)   │
│  ├─ parent_chunks│   └─────────────────────────────────────────┘
│  └─ child_chunks│
└──────┬──────────┘
       │
┌──────▼────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE LAYER                        │
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐  │
│  │ Qdrant Cloud│  │ Redis Cloud │  │      Langfuse        │  │
│  │ (us-east-2) │  │ (us-east-1) │  │   cloud.langfuse.com │  │
│  │             │  │             │  │                      │  │
│  │ parent_     │  │ exact:*     │  │ trace per query      │  │
│  │ chunks      │  │ semantic:*  │  │ spans: embed,        │  │
│  │ (payload)   │  │ rerank:*    │  │ retrieve, rerank,    │  │
│  │             │  │ parent:*    │  │ expand, generate     │  │
│  │ child_      │  │             │  │ token counts         │  │
│  │ chunks      │  │ TTL: 24h    │  │ latency per step     │  │
│  │ (768d+sparse│  │             │  │                      │  │
│  └─────────────┘  └─────────────┘  └──────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **API** | FastAPI + Uvicorn | Async HTTP server, OpenAPI docs |
| **Deployment** | Google Cloud Run | Serverless containerised deployment |
| **Container Registry** | Google Container Registry | Docker image storage |
| **PDF Parsing** | pypdf | Text extraction from textbook |
| **Chunking** | Custom DoclingChunker | Structure-aware parent-child splits |
| **Dense Embeddings** | nomic-embed-text-v1 | 768-dim semantic vectors |
| **Sparse Embeddings** | SPLADE (FastEmbed) | BM25-style keyword vectors |
| **Reranker** | ms-marco-MiniLM-L-6-v2 | Cross-encoder precision scoring |
| **Vector DB** | Qdrant Cloud | Hybrid dense+sparse collections |
| **LLM** | Groq (llama-3.3-70b) | Answer generation |
| **Observability** | Langfuse v4 | Per-query tracing + token counts |
| **Caching** | Redis Cloud | 4-layer query + chunk caching |
| **Config** | Pydantic Settings | Typed env var management |

---

## Project Structure

```
cs_rag/
├── app/
│   ├── main.py                    # FastAPI app factory + CORS
│   ├── config.py                  # Pydantic settings from env vars
│   ├── dependencies.py            # FastAPI DI — service singletons
│   │
│   ├── api/
│   │   ├── routes/
│   │   │   ├── ingest.py          # POST /api/v1/ingest
│   │   │   ├── query.py           # POST /api/v1/query
│   │   │   ├── evaluate.py        # POST /api/v1/evaluate
│   │   │   └── health.py          # GET  /api/v1/health
│   │   └── schemas/
│   │       ├── ingest.py          # IngestResponse
│   │       ├── query.py           # QueryRequest, QueryResponse
│   │       ├── evaluate.py        # EvalRequest, EvalResponse
│   │       └── health.py          # HealthResponse
│   │
│   ├── services/
│   │   ├── embedding_service.py   # Dense + sparse + reranker
│   │   ├── ingestion_service.py   # PDF → chunks → Qdrant
│   │   ├── retrieval_service.py   # Query → answer + Langfuse
│   │   └── evaluation_service.py  # RAGAS metrics
│   │
│   └── core/
│       ├── chunker.py             # ParentChildChunker (pypdf)
│       ├── vector_store.py        # Qdrant hybrid wrapper
│       ├── cache.py               # Redis 4-layer cache
│       └── langfuse_client.py     # Singleton Langfuse client
│
├── ui/
│   └── algomind.html              # Chat UI
├── .env.example                   # Environment variable template
├── requirements.txt
├── Dockerfile
└── docker-compose.yml             # Local development setup
```

---

## Parent-Child Chunking Strategy

Standard fixed-size chunking destroys the hierarchical structure of a textbook. AlgoMind uses structure-aware parent-child chunking:

```
Chapter 5: Graph Traversal           ← ParentChunk (stored, NOT embedded)
│   Full chapter text ~11,000 words
│   Retrieved by ID after child match
│
├── 5.1 Flavors of Graphs            ← ChildChunks (embedded + indexed)
│       window 0: 200 tokens
│       window 1: 200 tokens (20 overlap)
│
├── 5.6 Breadth-First Search
│       window 0: 200 tokens
│       window 1: 200 tokens
│
└── 5.8 Depth-First Search
        window 0: 200 tokens
```

**Result:** 19 parent chunks (chapters), 1,703 child chunks (sections), ~200 tokens each.

---

## Hybrid Search + RRF Fusion

```
Query: "What is the time complexity of BFS?"
           │
           ├─── Dense encoder (Nomic 768d) ──► semantic matches
           │    "graph traversal", "level-by-level", "queue"
           │
           └─── Sparse encoder (SPLADE) ──────► keyword matches
                "BFS", "Breadth-First", "O(n+m)"
                           │
                    RRF Fusion (k=60)
                    score = Σ 1/(k + rank_i)
                           │
                    Top-20 candidates
                           │
               Cross-encoder reranker
               ms-marco-MiniLM-L-6-v2
                           │
                    Top-5 for LLM context
```

---

## Caching Architecture

Four independent cache layers in Redis:

| Layer | Key | TTL | Savings |
|---|---|---|---|
| **Exact query** | `exact:md5(question)` | 24h | Full pipeline ~3200ms → 1ms |
| **Semantic query** | `semantic:md5(question)` | 24h | Paraphrase queries cached |
| **Rerank scores** | `rerank:hash(query+chunk_ids)` | 6h | Cross-encoder ~1800ms → 3ms |
| **Parent chunks** | `parent:{uuid}` | 7 days | Qdrant scroll ~80ms → 1ms |

Cache is fully invalidated on every re-ingestion. If Redis is unavailable the pipeline continues without caching — performance degrades but nothing breaks.

---

## Observability

Every query creates a Langfuse trace with 5 child spans:

```
rag_query  (chain)
├── embed_query      → latency_ms, dense_dim, sparse_nnz
├── hybrid_retrieve  → latency_ms, num_candidates, top_rrf_score
├── rerank           → latency_ms, top_rerank_score, top_sections
├── expand_parents   → latency_ms, num_parents, chapters
└── llm_generation   → latency_ms, prompt_tokens, completion_tokens
```

---

## API Reference

### `GET /api/v1/health`

```bash
curl https://cs-rag-944198740984.us-central1.run.app/api/v1/health
```

```json
{
  "status": "ok",
  "version": "1.0.0",
  "qdrant": "ok",
  "collections": {
    "parent_chunks": 19,
    "child_chunks": 1703
  }
}
```

### `POST /api/v1/ingest`

```bash
curl -X POST https://cs-rag-944198740984.us-central1.run.app/api/v1/ingest \
  -F "file=@algorithm_design_manual.pdf"
```

```json
{
  "status": "success",
  "parents_stored": 19,
  "children_stored": 1703,
  "dense_dim": 768,
  "sparse_avg_nnz": 142,
  "cache_warmed": 19
}
```

### `POST /api/v1/query`

```bash
curl -X POST https://cs-rag-944198740984.us-central1.run.app/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is BFS and its time complexity?", "top_k": 5}'
```

```json
{
  "answer": "BFS explores a graph level by level using a queue...",
  "sources": [
    {
      "section": "5.6 Breadth-First Search",
      "chapter": "5 Graph Traversal",
      "page": 162,
      "rrf_score": 0.6429,
      "rerank_score": 2.4151
    }
  ],
  "trace_id": "3eb5db1c2915a7c2dbeac29329cc5621",
  "context_used": 3,
  "total_latency_ms": 3194
}
```

---

## Deployment — Google Cloud Run

### Infrastructure

| Component | Service | Details |
|---|---|---|
| **API** | Google Cloud Run | us-central1, 8GB RAM, 4 vCPU |
| **Vector DB** | Qdrant Cloud | Free tier, us-east-2 |
| **Cache** | Redis Cloud | Free tier, us-east-1 |
| **Observability** | Langfuse Cloud | cloud.langfuse.com |
| **LLM** | Groq API | llama-3.3-70b-versatile |

### Deploy from scratch

```bash
# authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# enable APIs
gcloud services enable run.googleapis.com containerregistry.googleapis.com

# build and push
docker build -t gcr.io/YOUR_PROJECT_ID/cs-rag:latest .
docker push gcr.io/YOUR_PROJECT_ID/cs-rag:latest

# deploy
gcloud run deploy cs-rag \
  --image gcr.io/YOUR_PROJECT_ID/cs-rag:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 8Gi \
  --cpu 4 \
  --timeout 3600 \
  --port 8000 \
  --set-env-vars "GROQ_API_KEY=..." \
  --set-env-vars "QDRANT_HOST=..." \
  --set-env-vars "QDRANT_API_KEY=..." \
  --set-env-vars "REDIS_URL=..." \
  --set-env-vars "LANGFUSE_PUBLIC_KEY=..." \
  --set-env-vars "LANGFUSE_SECRET_KEY=..."
```

### Update environment variables

```bash
gcloud run services update cs-rag \
  --region us-central1 \
  --set-env-vars "KEY=value"
```

### Cost analysis

```
Cloud Run  = 3s × 4 vCPU × $0.0000240/vCPU-s
           + 3s × 8GB  × $0.0000025/GB-s
           = $0.000348 per query

Groq Dev   = 1500 tokens × $0.05/1M tokens
           = $0.000075 per query

Total      ≈ $0.000425 per query
           = $0.43 per 1,000 queries
```

Cloud Run free tier covers 2 million requests/month — effectively $0 for portfolio usage.

---

## Local Development

### Prerequisites

- Python 3.11+
- Docker Desktop

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/cs-rag.git
cd cs-rag
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env
# fill in your keys in .env
```

### Start with Docker Compose

```bash
docker compose up
```

This starts three services: FastAPI on port 8000, Qdrant on port 6333, Redis on port 6379.

### Ingest the textbook

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@data/books/algorithm_design_manual.pdf"
```

---

## Environment Variables

```dotenv
GROQ_API_KEY=gsk_...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1
EMBEDDING_DIM=768
GROQ_MODEL=llama-3.3-70b-versatile
QDRANT_HOST=your-cluster.us-east-2-0.aws.cloud.qdrant.io
QDRANT_PORT=6333
QDRANT_API_KEY=your_qdrant_api_key
REDIS_URL=redis://default:password@host:port
PARENT_COLLECTION=parent_chunks
CHILD_COLLECTION=child_chunks
PARENT_CHUNK_TOKENS=1200
CHILD_CHUNK_TOKENS=200
CHILD_OVERLAP_TOKENS=20
```

---

## System Design Considerations

**Throughput:** 1 concurrent request per Cloud Run instance, ~20 requests/minute. Cloud Run auto-scales to 10 instances = 200 requests/minute. Cold start ~60 seconds (fix: `--min-instances=1`).

**Bottleneck:** Cross-encoder reranking at ~1800ms = 56% of total latency. 20 sequential CPU inference passes. Fix: Redis rerank cache (already built, 3ms on hit), batch inference (~400ms), GPU instance (~18ms).

**Cost per query:** ~$0.000425 = $0.43 per 1,000 queries. Cache hit = $0.000001. 50% cache hit rate halves cost.

**Failure modes:** Qdrant down → Redis cache fallback + graceful 503. Groq down → return sources without generation + exponential backoff. Redis down → caching disabled, pipeline continues (already handled with `self.enabled=False`).

**Scaling to 1M users:** Decompose into GPU microservices (embedding, reranking, LLM), async ingestion via Pub/Sub, shard Qdrant, Redis Cluster. Per-query cost drops to ~$0.0002 at scale due to GPU amortisation.

---

## Known Limitations

**Groq free tier** — 100,000 tokens/day. Upgrade to Dev tier for production workloads.

**CPU-only inference** — embedding and reranking run on CPU. Ingestion takes ~10 minutes on Cloud Run. GPU instance reduces this to under 1 minute.

**Cold start latency** — first request on a new Cloud Run instance takes ~60 seconds to load model weights. Set `--min-instances=1` for production demos.

**RAGAS evaluation** — Groq's API does not support `n>1` completions which RAGAS requires internally. Evaluation requires a paid tier or OpenAI key.

---

## What I Learned Building This

- Parent-child chunking significantly outperforms fixed-size chunking for structured documents — preserves semantic context at retrieval while maintaining precision
- Hybrid search is essential for technical content — exact algorithm names require keyword matching that dense-only search consistently misses
- The reranker is the biggest quality lever — cross-encoder reranking on top-20 candidates surfaces the correct section even when RRF ranking is imprecise
- Langfuse span-level tracing reveals that reranking dominates latency, making it the most valuable cache target
- RAGAS evaluation in production requires a paid LLM tier — the judge LLM makes many sequential calls that hit free tier rate limits

---

## Author

**Harshini** — AI/ML Engineer  
[GitHub](https://github.com/harshini957) · [LinkedIn](https://www.linkedin.com/in/harshini-n-d-a6040b22a/)

Built as a production learning project to understand every component of a RAG system from raw PDF to deployed API on Google Cloud Run.
