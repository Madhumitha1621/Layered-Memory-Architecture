#!/usr/bin/env bash
set -euo pipefail

echo "Starting RAG Chat (v6.0.0)..."
echo ""
echo "Tip: first startup downloads the embedding model (~440MB) and caches it."
echo "     Use 'python backend/cli.py ingest' to index your knowledge base files."
echo ""
cd "$(dirname "$0")"
.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 --app-dir backend
