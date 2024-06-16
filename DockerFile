# Use the official Python base image
FROM python:3.11-slim

# Set the initial working directory
WORKDIR /api

# Copy the requirements.txt file from the api directory
COPY api/requirements.txt ./

# Install dependencies specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Create a non-root user and set up the environment
RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the final working directory
WORKDIR $HOME/app

# Copy the entire project into the container, setting the appropriate ownership
COPY --chown=user . $HOME/app

# Set the working directory to /home/user/app/api
WORKDIR $HOME/app/api

# Command to run the API
CMD ["python", "endpoints.py"]
