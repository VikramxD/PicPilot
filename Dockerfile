FROM python:3.11

# Set working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy everything from the current directory to /app in the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r api/requirements.txt

# Create a non-root user to run the application
RUN useradd -m -u 1000 user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=$HOME/app/scripts

# Set ownership to non-root user
RUN chown -R user:user /app

# Switch to the non-root user
USER user

# Install the application in editable mode
RUN pip install --no-cache-dir -e /app

# Set working directory for the application
WORKDIR /app/api

# Command to run the application
CMD ["python3", "picpilot.py"]
