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

# Install ALL dependencies from pyproject.toml to avoid missing module errors
# Using correct pip package names (some differ from import names)
echo "📦 Installing all required dependencies..."
pip install -q --root-user-action=ignore \
    polars pyarrow pandas duckdb \
    lightgbm xgboost \
    "ray[default]" \
    apscheduler sqlalchemy alembic \
    boto3 s3fs \
    pydantic pydantic-settings \
    structlog \
    prometheus-client \
    great-expectations \
    tenacity \
    ccxt \
    requests \
    python-telegram-bot \
    numpy scipy matplotlib \
    psycopg2-binary psutil \
    dropbox \
    || {
    echo "⚠️  Some dependencies may have failed to install, continuing anyway..."
}

# Run the engine as a module (required for relative imports)
# PYTHONPATH must point to directory containing 'src', not 'src' itself
python3 -m src.cloud.training.pipelines.daily_retrain

