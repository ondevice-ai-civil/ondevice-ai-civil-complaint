# Dockerfile for GovOn Backend
# Single Dockerfile for unified local/HuggingFace Spaces/production management
# Per-environment configuration is overridden via docker-compose.yml or HF Space environment variables

FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/GovOn-Org/GovOn"
LABEL org.opencontainers.image.description="GovOn AX Platform — Agentic Transformation for National Infrastructure"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    SERVING_PROFILE=container \
    PORT=7860 \
    HOST=0.0.0.0 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.7.12 /uv /bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev git gcc ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu126"

# 1) Install torch CUDA 12.6 wheel
RUN uv pip install --system --no-cache "torch>=2.8.0"
# 2) autoawq: requires torch at build time → no-build-isolation
RUN uv pip install --system --no-cache --no-build-isolation "autoawq>=0.2.8"
# 3) Install remaining packages (torch/autoawq already satisfied)
RUN grep -vE "^(torch([>=<! ]|$)|autoawq([>=<! ]|$)|#|[[:space:]]*$)" requirements.txt \
    | uv pip install --system --no-cache -r /dev/stdin
# 4) hf_transfer: parallel model download (10x speed boost)
RUN uv pip install --system --no-cache hf_transfer

# uid 1000: HF Spaces compatible + security
RUN useradd -m -u 1000 user \
    && mkdir -p models/faiss_index models/bm25_index data/processed .cache logs config

COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY agents* ./agents/

RUN chown -R user:user /app
USER user

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD CUDA_VISIBLE_DEVICES="" python3.10 -c "import urllib.request,json; r=urllib.request.urlopen('http://localhost:${PORT}/health'); d=json.loads(r.read()); exit(0 if r.status==200 and d.get('status')=='healthy' else 1)"

EXPOSE ${PORT}

# Start vLLM OpenAI server + FastAPI sequentially via entrypoint script
COPY scripts/entrypoint.sh ./scripts/entrypoint.sh
CMD ["bash", "scripts/entrypoint.sh"]
