FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Set up working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-dev \
    python3.11-venv \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create user and set permissions
RUN useradd -m -u 1000 user && \
    chown -R user:user /app

# Switch to user
USER user

# Create and activate virtual environment
RUN python3.11 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Copy necessary files maintaining structure
COPY --chown=user:user api/ /app/api/
COPY --chown=user:user scripts/ /app/scripts/
COPY --chown=user:user configs/ /app/configs/
COPY --chown=user:user setup.py /app/

# Install dependencies in venv
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r api/requirements.txt && \
    pip install --no-cache-dir runpod && \
    pip install --no-cache-dir -e .

# Copy the service file
COPY --chown=user:user serverless/text-to-image/run_text-to-image.py .

CMD ["python", "run_text-to-image.py"]