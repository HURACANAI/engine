#!/bin/bash
# Simple script to run the engine on RunPod
# Usage: ./scripts/run_engine.sh

set -e  # Exit on error

echo "🚀 Starting Engine on RunPod..."
echo ""

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Change to project root
cd "$PROJECT_ROOT"

echo "📁 Project root: $PROJECT_ROOT"
echo "🐍 Python path: $PYTHONPATH"
echo ""

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
    || echo "⚠️  Some dependencies may have failed to install, continuing anyway..."

# Check if PostgreSQL is running
echo "🔍 Checking PostgreSQL..."
if ! pg_isready -h localhost -p 5432 >/dev/null 2>&1; then
    echo "⚠️  PostgreSQL not running, starting it..."
    /etc/init.d/postgresql start || service postgresql start || true
    sleep 2
fi

# Check database connection
echo "🔍 Testing database connection..."
python3 -c "
import psycopg2
try:
    conn = psycopg2.connect('postgresql://haq:huracan123@localhost:5432/huracan')
    print('✅ Database connection OK')
    conn.close()
except Exception as e:
    print(f'❌ Database connection failed: {e}')
    exit(1)
" || {
    echo "❌ Database connection failed. Please check your database setup."
    exit 1
}

echo ""
echo "🎯 Running Engine..."
echo ""

# Run the engine as a module (required for relative imports)
python3 -m src.cloud.training.pipelines.daily_retrain

