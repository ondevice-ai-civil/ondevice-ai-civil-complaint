# Dockerfile for GovOn Backend

# Use NVIDIA CUDA base image with Python 3.10
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/GovOn-Org/GovOn"
LABEL org.opencontainers.image.description="GovOn AI Civil Complaint Analysis System"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    SERVING_PROFILE="container" \
    MODEL_PATH="umyunsang/GovOn-EXAONE-LoRA-v2" \
    DATA_PATH="/app/data/processed/v2_train.jsonl" \
    INDEX_PATH="/app/models/faiss_index/complaints.index"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt .

# Install runtime dependencies once. The source tree is copied below and does
# not require installing the project package or dev extras inside the image.
RUN python3.10 -m pip install --no-cache-dir --upgrade pip \
    && python3.10 -m pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY agents/ ./agents/

# Create directories for models and data
RUN mkdir -p models/faiss_index data/processed

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["python3.10", "-m", "src.inference.api_server"]
