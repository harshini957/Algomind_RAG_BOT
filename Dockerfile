FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

RUN mkdir -p data/uploads && chown -R appuser:appuser /app

USER appuser

ENV PYTHONPATH="/app"
ENV PORT=8000

# pre-cache nomic dense model
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True); \
print('nomic cached')" || echo "skipped"

# pre-cache cross-encoder reranker
RUN python -c "\
from sentence_transformers import CrossEncoder; \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('reranker cached')" || echo "skipped"

# pre-cache SPLADE sparse model
RUN python -c "\
from fastembed import SparseTextEmbedding; \
model = SparseTextEmbedding(model_name='prithivida/Splade_PP_en_v1'); \
print('SPLADE cached')" || echo "skipped"

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/v1/health || exit 1

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 65"]