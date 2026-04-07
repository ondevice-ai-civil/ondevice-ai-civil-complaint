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

# autoawq's build backend imports torch during wheel build (build isolation hides
# system packages), so torch must be pre-installed and autoawq built without isolation.
RUN uv pip install --system --no-cache "torch>=2.8.0" && \
    uv pip install --system --no-cache --no-build-isolation "autoawq>=0.2.8" && \
    uv pip install --system --no-cache -r requirements.txt

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

# Command to run the application
CMD ["python3.10", "-m", "src.inference.api_server"]
