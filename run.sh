#!/bin/bash
# Super simple script to run the engine
# Usage: ./run.sh

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Install missing dependencies if needed
python3 -c "import boto3, matplotlib, pydantic_settings" 2>/dev/null || {
    echo "Installing missing dependencies (boto3, matplotlib, pydantic-settings)..."
    pip install -q boto3 matplotlib pydantic-settings
}

# Run the engine
python3 src/cloud/training/pipelines/daily_retrain.py

