#!/usr/bin/env bash
set -e

echo "Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing Diffusion & PEFT dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "Configuring Hugging Face Accelerate default environment..."
accelerate config default

echo "Launching Jupyter Notebook..."
jupyter notebook
