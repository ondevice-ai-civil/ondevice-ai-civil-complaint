# Dockerfile for GovOn Backend

# Use NVIDIA CUDA base image with Python 3.10
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
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

# Upgrade pip
RUN python3.10 -m pip install --upgrade pip

# Copy project files
COPY pyproject.toml .
COPY requirements.txt .

# Install dependencies
# Note: autoawq and vllm require specific CUDA versions
RUN python3.10 -m pip install .

# Copy source code
COPY src/ ./src/
COPY agents/ ./agents/

# Create directories for models and data
RUN mkdir -p models/faiss_index data/processed

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["python3.10", "-m", "src.inference.api_server"]
