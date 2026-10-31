# ── Stage 1: build the React frontend ──────────────────────────────
# frontend/dist is gitignored, so it can't be copied from the build context
# of a fresh clone; build it here instead.
FROM node:20-bookworm-slim AS frontend

WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python app ────────────────────────────────────────────
FROM python:3.12.2-slim-bookworm

WORKDIR /app

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Use a fixed Debian mirror to avoid Hash Sum mismatch
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && printf 'Acquire::http::Pipeline-Depth "0";\nAcquire::http::No-Cache "true";\nAcquire::BrokenProxy "true";\nAcquire::ForceIPv4 "true";\n' > /etc/apt/apt.conf.d/99fixbadproxy \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*



COPY requirements.txt ./requirements.txt

# Install CPU-only PyTorch first (smaller image, no CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY backend/ ./backend/
COPY knowledge/ ./knowledge/
COPY --from=frontend /frontend/dist/ ./frontend/dist/

EXPOSE 8000

WORKDIR /app/backend

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
