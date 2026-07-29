# Crypto Market Intelligence — query API / ingest image (pgvector deployment).
#
# The ONNX MiniLM embedding model is baked into the image at build time, so there
# is no ~79 MB download on the first request (no cold-start penalty on Railway).
# Torch is NOT installed — onnxruntime ships with chromadb via the default `onnx`
# embedding backend.
#
# Same image serves both Railway services (different commands):
#   - Query API (default CMD): initial populate (once) then uvicorn on $PORT.
#   - Cron ingest (override):  crypto-intel ingest --sources news,exchange,regulator --lookback-hours 3
#
# Required runtime env (set in Railway): STORE_BACKEND=pgvector, DATABASE_URL=...

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/app \
    EMBED_BACKEND=onnx \
    ANONYMIZED_TELEMETRY=False

# Runtime libs: libgomp1 for onnxruntime; ca-certificates for HTTPS
# (RSS feeds, CoinGecko, Supabase). No compilers needed — psycopg[binary] and
# onnxruntime install from wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime user; HOME holds the baked model cache (~/.cache/chroma).
RUN useradd --create-home --home-dir /home/app --shell /usr/sbin/nologin app

WORKDIR /app
COPY pyproject.toml README.md ./
COPY crypto_intel ./crypto_intel
COPY deploy ./deploy

RUN pip install --upgrade pip \
    && pip install -e ".[web,pg]"

# Bake the ONNX MiniLM model into the image (downloads + extracts into
# $HOME/.cache/chroma). After this, runtime needs no model download.
RUN python -c "from crypto_intel.config import Settings; from crypto_intel.embeddings import get_embedder; get_embedder(Settings(embed_backend='onnx')).encode(['warm up the onnx model'])"

# Writable scratch for the dedup JSONL + price cache (repo-relative data/store).
RUN mkdir -p /app/data/store \
    && chown -R app:app /app /home/app

USER app
EXPOSE 8000

# Query API: one-time initial populate (--if-empty, no-op once populated) then
# uvicorn on Railway's $PORT. The outer `sh -c` expands ${PORT:-8000} before
# handing the command to deploy/start.sh.
CMD ["sh", "-c", "sh deploy/start.sh uvicorn crypto_intel.web.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
