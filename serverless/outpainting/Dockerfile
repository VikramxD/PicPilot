FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/user/.local/bin:$PATH" \
    PYTHONPATH=/app

# Create user first
RUN useradd -m -u 1000 user

# Set up working directory and permissions
WORKDIR /app
RUN chown user:user /app

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

# Create venv with correct permissions
RUN python3.11 -m venv /home/user/venv && \
    chown -R user:user /home/user/venv
ENV PATH="/home/user/venv/bin:$PATH"

# Switch to user
USER user

# Copy all required files and directories with correct permissions
COPY --chown=user:user api/ /app/api/
COPY --chown=user:user scripts/ /app/scripts/
COPY --chown=user:user configs/ /app/configs/
COPY --chown=user:user setup.py /app/
COPY --chown=user:user pyproject.toml /app/

# Set HOME for pip cache
ENV HOME=/home/user

# Install Python dependencies as user
RUN pip install --user --no-cache-dir -U pip setuptools wheel && \
    pip install --user --no-cache-dir -r api/requirements.txt && \
    pip install --user --no-cache-dir runpod && \
    pip install --user --no-cache-dir -e .

# Copy the handler (service-specific)
COPY --chown=user:user serverless/outpainting/run_outpainting.py .

CMD ["python", "run_outpainting.py"]