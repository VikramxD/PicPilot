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
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Create user and set permissions
RUN useradd -m -u 1000 user && \
    chown -R user:user /app
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

# Copy the handler (change this for each service)
COPY --chown=user:user serverless/text-to-image/run_outpainting.py .

# Use full path to python
CMD ["/usr/bin/python3.11", "run_outpainting.py"]