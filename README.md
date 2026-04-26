# AlgoMind — Production RAG System for The Algorithm Design Manual

> A production-grade Retrieval-Augmented Generation system built from scratch over the Algorithm Design Manual by Steven S. Skiena. Combines hybrid dense + sparse search, cross-encoder reranking, parent-child chunking, real-time observability, and multi-layer caching into a deployable FastAPI service.

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
                          │ HTTP
┌─────────────────────────▼───────────────────────────────────────┐
│                     API LAYER (FastAPI)                         │
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
│  │   Qdrant    │  │    Redis    │  │      Langfuse        │  │
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
├── 5.6 Breadth-First Search         ← ChildChunks
│       window 0: 200 tokens
│       window 1: 200 tokens
│
└── 5.8 Depth-First Search           ← ChildChunks
        window 0: 200 tokens
```

**Why this works:** Small child chunks are precise — they match specific queries accurately. When a child is retrieved, its parent's full chapter text is fetched and sent to the LLM as context. This gives precise retrieval with rich generation context.

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
                    score = Σ 1/(k + rank)
                           │
                    Top-20 candidates
                           │
               Cross-encoder reranker
               ms-marco-MiniLM-L-6-v2
               reads query + chunk together
                           │
                    Top-5 for LLM context
```

**Why RRF over score normalisation:** Dense and sparse scores have incompatible scales. RRF operates on ranks not scores, so no normalisation is needed. A chunk ranked #1 by both systems gets a very high fused score regardless of raw score magnitudes.

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **API** | FastAPI + Uvicorn | Async HTTP server, OpenAPI docs |
| **PDF Parsing** | pypdf | Text extraction from textbook |
| **Chunking** | Custom DoclingChunker | Structure-aware parent-child splits |
| **Dense Embeddings** | nomic-embed-text-v1 | 768-dim semantic vectors |
| **Sparse Embeddings** | SPLADE (FastEmbed) | BM25-style keyword vectors |
| **Reranker** | ms-marco-MiniLM-L-6-v2 | Cross-encoder precision scoring |
| **Vector DB** | Qdrant | Hybrid dense+sparse collections |
| **LLM** | Groq (llama-3.3-70b) | Answer generation |
| **Observability** | Langfuse v4 | Per-query tracing + token counts |
| **Caching** | Redis | 4-layer query + chunk caching |
| **Config** | Pydantic Settings | Typed env var management |
| **UI** | Vanilla HTML/CSS/JS | Interactive chat interface |

---

## Project Structure

```
cs_rag/
├── app/
│   ├── main.py                    # FastAPI app factory + CORS
│   ├── config.py                  # Pydantic settings from .env
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
├── data/books/                    # PDF storage (gitignored)
├── qdrant_storage/                # Qdrant persistence (gitignored)
├── ui/
│   └── algomind.html              # Chat UI
├── .env                           # Secrets (gitignored)
├── requirements.txt
├── Dockerfile
└── render.yaml
```

---

## RAG Pipeline — Step by Step

### Ingestion (one-time, offline)

```
1. pypdf extracts raw text from all 739 pages
2. DoclingChunker detects chapter headings via regex
   pattern: standalone digit + Title Case title
3. Each chapter becomes a ParentChunk (~1,200 tokens)
4. Each section becomes ChildChunks (200 tokens, 20 overlap)
   children store parent_id, section, chapter, page metadata
5. EmbeddingService generates 768-dim dense vectors
   using nomic-embed-text-v1 with "search_document:" prefix
6. EmbeddingService generates SPLADE sparse vectors
   using FastEmbed prithivida/Splade_PP_en_v1
7. VectorStore stores parents (payload only) in Qdrant
8. VectorStore stores children (dense + sparse) in Qdrant
9. CacheService pre-warms parent chunk cache in Redis
```

### Query (real-time, per request)

```
1.  Redis exact cache check      → cache hit: return in 1ms
2.  Embed query (dense + sparse) → 768d + sparse terms
3.  Redis semantic cache check   → cosine_sim > 0.95: return cached
4.  Qdrant hybrid search         → dense prefetch + sparse prefetch
5.  RRF fusion                   → top-20 fused candidates
6.  Redis rerank cache check     → cache hit: skip cross-encoder
7.  Cross-encoder reranking      → top-20 → top-5
8.  Redis parent cache lookup    → fetch chapter texts
9.  Prompt assembly              → system + context + question
10. Groq LLM generation          → grounded answer
11. Redis cache store            → exact + semantic
12. Langfuse flush               → trace visible in dashboard
```

---

## Caching Architecture

Four independent cache layers in Redis:

| Layer | Key | TTL | Savings |
|---|---|---|---|
| **Exact query** | `exact:md5(question)` | 24h | Full pipeline ~3200ms → 1ms |
| **Semantic query** | `semantic:md5(question)` | 24h | Full pipeline on paraphrases |
| **Rerank scores** | `rerank:hash(query+chunk_ids)` | 6h | Cross-encoder ~1800ms → 3ms |
| **Parent chunks** | `parent:{uuid}` | 7 days | Qdrant scroll ~80ms → 1ms |

Cache is fully invalidated on every re-ingestion to prevent stale answers.

---

## Observability

Every query creates a Langfuse trace with 5 child spans:

```
rag_query  (chain)
├── embed_query      → latency_ms, dense_dim, sparse_nnz
├── hybrid_retrieve  → latency_ms, num_candidates, top_rrf_score, sections
├── rerank           → latency_ms, top_rerank_score, top_sections
├── expand_parents   → latency_ms, num_parents, chapters
└── llm_generation   → latency_ms, prompt_tokens, completion_tokens, model
```

**What you can monitor in Langfuse:**
- End-to-end latency per query
- Token usage and cost per query
- Retrieval quality (RRF + rerank scores)
- Which sections are retrieved most frequently
- Failed traces and error rates

---

## API Reference

### `POST /api/v1/ingest`

Upload a PDF and run the full ingestion pipeline.

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@algorithm_design_manual.pdf"
```

**Response:**
```json
{
  "status": "success",
  "source": "data/uploads/algorithm_design_manual.pdf",
  "parents_stored": 19,
  "children_stored": 1703,
  "dense_dim": 768,
  "sparse_avg_nnz": 142,
  "cache_warmed": 19
}
```

---

### `POST /api/v1/query`

Ask a question about the indexed content.

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is BFS and its time complexity?", "top_k": 5}'
```

**Response:**
```json
{
  "answer": "BFS (Breadth-First Search) explores a graph level by level using a queue...",
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

### `GET /api/v1/health`

```bash
curl http://localhost:8000/api/v1/health
```

**Response:**
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

---

## Local Setup

### Prerequisites

- Python 3.11+
- Docker Desktop
- 4GB RAM minimum (8GB recommended)

### 1 — Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/cs-rag.git
cd cs-rag
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2 — Start infrastructure

```bash
# Qdrant
docker run -d --name qdrant_local \
  -p 6333:6333 \
  -v ${PWD}/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Redis
docker run -d --name redis_local \
  -p 6379:6379 \
  redis:7-alpine
```

### 3 — Configure environment

```bash
cp .env.example .env
# fill in your keys in .env
```

```dotenv
GROQ_API_KEY=your_groq_key
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxx
LANGFUSE_BASE_URL=https://cloud.langfuse.com
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1
EMBEDDING_DIM=768
GROQ_MODEL=llama-3.3-70b-versatile
QDRANT_HOST=localhost
QDRANT_PORT=6333
PARENT_COLLECTION=parent_chunks
CHILD_COLLECTION=child_chunks
PARENT_CHUNK_TOKENS=1200
CHILD_CHUNK_TOKENS=200
CHILD_OVERLAP_TOKENS=20
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 4 — Start the API

```bash
python -m uvicorn app.main:app --reload --port 8000
```

### 5 — Ingest the textbook

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@data/books/algorithm_design_manual.pdf"
```

⏱ Takes 8–12 minutes on CPU (embedding 1703 chunks).

### 6 — Start the UI

```bash
python -m http.server 3000
# open http://localhost:3000/ui/algomind.html
```

---

## Deployment on Render

### Prerequisites
- GitHub account with this repo pushed
- Render account (render.com)
- Groq API key
- Langfuse account + project keys

### Steps

1. Go to **render.com** → **New** → **Blueprint**
2. Connect your GitHub repo
3. Render detects `render.yaml` and provisions 3 services:
   - `cs-rag-api` — FastAPI app (Starter $7/mo)
   - `cs-rag-qdrant` — Qdrant vector DB (Starter $7/mo)
   - `cs-rag-redis` — Redis cache (Starter $7/mo)
4. In `cs-rag-api` → **Environment** → add secrets:
   - `GROQ_API_KEY`
   - `LANGFUSE_PUBLIC_KEY`
   - `LANGFUSE_SECRET_KEY`
5. First deploy takes 10–12 minutes (model download)
6. Once live, ingest your PDF via the `/ingest` endpoint
7. Deploy `ui/algomind.html` as a **Static Site** (free)

**Live URLs:**
```
https://cs-rag-api.onrender.com/docs   ← API + Swagger
https://algomind.onrender.com          ← Chat UI
```

---

## Evaluation Metrics (RAGAS)

The system is designed to be evaluated with three RAGAS metrics:

| Metric | Measures | Target |
|---|---|---|
| **Faithfulness** | Did the answer stay grounded in retrieved context? | > 0.80 |
| **Answer Relevancy** | Did the answer address the question asked? | > 0.85 |
| **Context Precision** | Were retrieved chunks actually useful? | > 0.70 |

```bash
curl -X POST http://localhost:8000/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [
      {
        "question": "What is BFS time complexity?",
        "ground_truth": "BFS runs in O(n+m) where n is vertices and m is edges."
      }
    ]
  }'
```

---

## Known Limitations

**Groq free tier** — 100,000 tokens/day limit. Upgrade to Dev tier for production workloads or use `llama-3.1-8b-instant` for development.

**CPU-only inference** — embedding and reranking run on CPU. On Render starter, ingestion takes ~10 minutes. Adding a GPU instance reduces this to under 1 minute.

**PDF quality dependency** — chunking uses regex heading detection tuned for the Algorithm Design Manual's typography. Different PDFs will need the `CHAPTER_RE` and `SECTION_RE` patterns in `chunker.py` adjusted.

**RAGAS evaluation** — Groq's API does not support `n>1` completions which RAGAS requires internally. Evaluation requires an OpenAI key or Groq Dev tier.

---

## What I Learned Building This

- **Parent-child chunking** is significantly better than fixed-size chunking for structured documents — preserves semantic context at retrieval while maintaining precision
- **Hybrid search is not optional for technical content** — exact algorithm names (BFS, Dijkstra, O(n log n)) require keyword matching that dense-only search misses
- **The reranker is the biggest quality lever** — cross-encoder reranking on top-20 candidates consistently surfaces the correct section even when RRF ranking is imprecise
- **Langfuse span-level tracing** reveals that reranking (~1800ms) dominates latency, making it the most valuable cache target
- **RAGAS evaluation in production** requires a paid LLM tier — the judge LLM makes many sequential calls that hit free tier rate limits

---

## Author

**Harshini** — Final year student and AI/ML Engineer  
[GitHub](https://github.com/harshini957) · [LinkedIn](https://linkedin.com/in/harshini)

Built as a production learning project to understand every component of a RAG system from raw PDF to deployed API.