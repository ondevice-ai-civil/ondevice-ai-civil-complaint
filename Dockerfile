# Dockerfile for GovOn Backend

# Use NVIDIA CUDA base image with Python 3.10
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/GovOn-Org/GovOn"
LABEL org.opencontainers.image.description="GovOn AI Civil Complaint Analysis System"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    SERVING_PROFILE="container" \
    MODEL_PATH="LGAI-EXAONE/EXAONE-4.0-32B-AWQ" \
    DATA_PATH="/app/data/processed/v2_train.jsonl" \
    INDEX_PATH="/app/models/faiss_index/complaints.index"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy project files
COPY requirements.txt .

# 1) torch: requirements.txt를 단일 소스로 읽어 CUDA 12.1 wheel 설치
# 2) autoawq: --no-build-isolation으로 시스템 torch를 빌드 백엔드에 노출
# 3) remaining deps: extra-index-url로 CUDA wheels (vllm, bitsandbytes) 해석
RUN set -eux; \
    TORCH_SPEC="$(grep -E '^[[:space:]]*torch([[:space:]]*[<>=!~].*)?$' requirements.txt | head -n 1 | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"; \
    test -n "$TORCH_SPEC"; \
    uv pip install --system --no-cache \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        "$TORCH_SPEC" && \
    uv pip install --system --no-cache --no-build-isolation "autoawq>=0.2.8" && \
    uv pip install --system --no-cache \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY agents/ ./agents/

# Create directories for models and data
RUN mkdir -p models/faiss_index data/processed

# Expose port
EXPOSE 8000

# Non-root user for security
RUN groupadd -r govon && useradd -r -g govon -d /app govon
RUN chown -R govon:govon /app
USER govon

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python3.10 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Command to run the application
CMD ["python3.10", "-m", "src.inference.api_server"]
