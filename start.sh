#!/bin/bash
exec /app/venv/bin/uvicorn app.main:app \
    --host 0.0.0.0 \
    --port ${PORT:-8000} \
    --workers 1 \
    --timeout-keep-alive 65