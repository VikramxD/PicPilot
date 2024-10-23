FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/user/.local/bin:$PATH" \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-venv \
    python3.11-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd -m -u 1000 user

# Create venv
RUN python3.11 -m venv /home/user/venv && chown -R user:user /home/user/venv
ENV PATH="/home/user/venv/bin:$PATH"

# Switch to user
USER user

# Copy necessary files
COPY --chown=user:user api/requirements.txt .
COPY --chown=user:user setup.py .
COPY --chown=user:user scripts/ scripts/
COPY --chown=user:user api/ api/
COPY --chown=user:user serverless/text-to-image/run_text-to-image.py .

# Install Python dependencies
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir runpod && \
    pip install --no-cache-dir -e .

# Command to run the handler
CMD ["python", "run_text-to-image.py"]