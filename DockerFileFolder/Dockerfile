FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/user/.local/bin:$PATH" \
    PYTHONPATH=/app

# Set working directory in the container
WORKDIR /app

# Install system dependencies, Python, and build tools
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

# Create a non-root user to run the application
RUN useradd -m -u 1000 user

# Create a virtual environment and give ownership to the non-root user
RUN python3.11 -m venv /home/user/venv && chown -R user:user /home/user/venv
ENV PATH="/home/user/venv/bin:$PATH"

# Switch to the non-root user
USER user

# Copy the entire application code
COPY --chown=user:user . .

# Install Python dependencies
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r api/requirements.txt

# Install the application in editable mode using setup.py
RUN pip install --no-cache-dir -e .

# Set working directory for the application
WORKDIR /app/api

# Command to run the application
CMD ["python3", "picpilot.py"]