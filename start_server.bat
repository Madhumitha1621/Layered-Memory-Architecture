@echo off
echo Starting RAG Chat (v6.0.0)...
echo.
echo Tip: first startup downloads the embedding model (~440MB) and caches it.
echo      Use "python backend\cli.py ingest" to index your knowledge base files.
echo.
cd /d "%~dp0"
.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --app-dir backend
