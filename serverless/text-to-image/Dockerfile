FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

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

RUN useradd -m -u 1000 user && \
    chown -R user:user /app

USER user

RUN python3.11 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

COPY --chown=user:user . .

RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r api/requirements.txt && \
    pip install --no-cache-dir runpod && \
    pip install --no-cache-dir -e .

CMD ["python", "serverless/text-to-image/run_text-to-image.py"]