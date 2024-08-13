#!/bin/bash

# Install necessary packages
sudo apt-get install -y python3-venv python3-pip
sudo apt install -y libgl1-mesa-glx

# Remove the old virtual environment if it exists and create a new one
if [ -d ".venv" ]; then
    rm -rf .venv
fi

python3 -m venv .venv

# Activate the virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Failed to create the virtual environment. Please check for errors."
    exit 1
fi

# Install the uv package within the virtual environment
pip install uv

# Change directory to api and install the required packages
cd api || { echo "API directory not found"; exit 1; }


# Ensure the uv command is available
if ! command -v uv &> /dev/null; then
    echo "uv command not found. Ensure it is installed and accessible."
    exit 1
fi

uv pip install -r requirements.txt
