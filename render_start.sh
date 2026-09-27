#!/bin/sh

python -m src.rag.rag_pipeline ingest-all

exec uvicorn src.api.main:app --host 0.0.0.0 --port "$PORT"