# Dockerfile for GovOn Backend
# 단일 Dockerfile로 로컬/HuggingFace Spaces/프로덕션 통합 관리
# 환경별 설정은 docker-compose.yml 또는 HF Space 환경변수로 오버라이드

FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/GovOn-Org/GovOn"
LABEL org.opencontainers.image.description="GovOn AI Civil Complaint Analysis System"

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

# 1) torch CUDA 12.6 wheel 설치
RUN uv pip install --system --no-cache "torch>=2.8.0"
# 2) autoawq: 빌드 시 torch 필요 → no-build-isolation
RUN uv pip install --system --no-cache --no-build-isolation "autoawq>=0.2.8"
# 3) 나머지 패키지 (torch/autoawq 이미 충족)
RUN grep -vE "^(torch([>=<! ]|$)|autoawq([>=<! ]|$)|#|[[:space:]]*$)" requirements.txt \
    | uv pip install --system --no-cache -r /dev/stdin
# 4) hf_transfer: 병렬 모델 다운로드 (10x 속도 향상)
RUN uv pip install --system --no-cache hf_transfer

# uid 1000: HF Spaces 호환 + 보안
RUN useradd -m -u 1000 user \
    && mkdir -p models/faiss_index models/bm25_index data/processed .cache logs config

COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY agents* ./agents/

RUN chown -R user:user /app
USER user

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD CUDA_VISIBLE_DEVICES="" python3.10 -c "import urllib.request; r = urllib.request.urlopen('http://localhost:${PORT}/health'); exit(0 if r.status == 200 else 1)"

EXPOSE ${PORT}

# vLLM OpenAI 서버 + FastAPI를 entrypoint script로 순차 기동
COPY scripts/entrypoint.sh ./scripts/entrypoint.sh
CMD ["bash", "scripts/entrypoint.sh"]
