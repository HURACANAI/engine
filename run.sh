#!/bin/bash
# Super simple script to run the engine
# Usage: ./run.sh

set -e  # Exit on error

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Change to project root
cd "$PROJECT_ROOT"

# Set PYTHONPATH to project root (so Python can find 'src' package)
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Install missing dependencies if needed
python3 -c "import boto3, matplotlib, pydantic_settings, dropbox" 2>/dev/null || {
    echo "Installing missing dependencies (boto3, matplotlib, pydantic-settings, dropbox)..."
    pip install -q boto3 matplotlib pydantic-settings dropbox
}

# Run the engine as a module (required for relative imports)
# PYTHONPATH must point to directory containing 'src', not 'src' itself
python3 -m src.cloud.training.pipelines.daily_retrain

