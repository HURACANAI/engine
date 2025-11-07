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
# Only install packages that are not already installed
echo "📦 Checking and installing required dependencies..."
PACKAGES="polars pyarrow pandas duckdb lightgbm xgboost 'ray[default]' apscheduler sqlalchemy alembic boto3 s3fs pydantic pydantic-settings structlog prometheus-client great-expectations tenacity ccxt requests python-telegram-bot numpy scipy matplotlib psycopg2-binary psutil dropbox"

MISSING_PACKAGES=""
for pkg in $PACKAGES; do
    # Remove quotes for checking
    pkg_name=$(echo "$pkg" | tr -d "'")
    # Check if package is installed
    if ! python3 -c "import ${pkg_name//\[default\]/}" 2>/dev/null; then
        MISSING_PACKAGES="$MISSING_PACKAGES $pkg"
    fi
done

if [ -n "$MISSING_PACKAGES" ]; then
    echo "📥 Installing missing packages:$MISSING_PACKAGES"
    pip install --root-user-action=ignore $MISSING_PACKAGES 2>&1 | tee /tmp/pip_install.log || {
        echo "⚠️  Some dependencies may have failed to install, continuing anyway..."
        echo "📋 Check /tmp/pip_install.log for details"
    }
    echo "✅ Dependencies installation complete"
else
    echo "✅ All dependencies already installed"
fi

# Run the engine as a module (required for relative imports)
# PYTHONPATH must point to directory containing 'src', not 'src' itself
python3 -m src.cloud.training.pipelines.daily_retrain

