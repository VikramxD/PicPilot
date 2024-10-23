# Image-to-Video Dockerfile (serverless/image-to-video/Dockerfile)
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/user/.local/bin:$PATH" \
    PYTHONPATH=/app

WORKDIR /app

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

RUN useradd -m -u 1000 user
RUN python3.11 -m venv /home/user/venv && chown -R user:user /home/user/venv
ENV PATH="/home/user/venv/bin:$PATH"

USER user

# Copy all required files and directories
COPY --chown=user:user api/ /app/api/
COPY --chown=user:user scripts/ /app/scripts/
COPY --chown=user:user configs/ /app/configs/
COPY --chown=user:user setup.py /app/
COPY --chown=user:user pyproject.toml /app/

# Install Python dependencies
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r api/requirements.txt && \
    pip install --no-cache-dir runpod && \
    pip install --no-cache-dir -e .

# Copy the handler
COPY --chown=user:user serverless/image-to-video/run_image-to-video.py .

CMD ["python", "run_image-to-video.py"]