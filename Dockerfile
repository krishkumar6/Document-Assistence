# Document Assistant: FastAPI app with local embeddings + reranker.
# Works on Railway, Render or any Docker host. Needs ~0.6 GB RAM at peak.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    # Models are baked into the image (below), outside the data volume.
    MODEL_CACHE_DIR=/opt/models

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Download both models at build time so the container starts fast and never
# needs the internet for them: Chroma's MiniLM embedder (~80 MB, into ~/.cache/chroma)
# and the fastembed reranker (~80 MB, into /opt/models).
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as E; E()(['warm up'])" \
 && python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder as T; list(T('Xenova/ms-marco-MiniLM-L-6-v2', cache_dir='/opt/models').rerank('q', ['d']))"

COPY app ./app

# Uploaded PDFs, the search index and logs live under DATA_DIR. Mount a persistent
# volume there (Railway: mount path /data, which also sets RAILWAY_VOLUME_MOUNT_PATH).
# The process runs as root because platform volumes are typically mounted root-owned,
# and a non-root user could not write uploads to them.
ENV DATA_DIR=/data
RUN mkdir -p /data

EXPOSE 8000
# PORT is provided by the platform when it sets one; 8000 otherwise.
# --proxy-headers: trust the platform proxy's X-Forwarded-For so rate limiting sees real client IPs.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
