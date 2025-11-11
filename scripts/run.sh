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

# Map pip package names to import names (some differ)
declare -A PACKAGE_MAP=(
    ["pydantic-settings"]="pydantic_settings"
    ["python-telegram-bot"]="telegram"
    ["psycopg2-binary"]="psycopg2"
    ["prometheus-client"]="prometheus_client"
    ["great-expectations"]="great_expectations"
)

# Check each package
MISSING_PACKAGES=""
for pkg in polars pyarrow pandas duckdb lightgbm xgboost "ray[default]" apscheduler sqlalchemy alembic boto3 s3fs pydantic pydantic-settings structlog prometheus-client great-expectations tenacity ccxt requests python-telegram-bot numpy scipy matplotlib psycopg2-binary psutil dropbox opentelemetry-api opentelemetry-sdk; do
    # Get import name (use mapped name if exists, otherwise use package name)
    if [[ -n "${PACKAGE_MAP[$pkg]}" ]]; then
        import_name="${PACKAGE_MAP[$pkg]}"
    else
        import_name="${pkg//\[default\]/}"
        import_name="${import_name//-/_}"
    fi
    
    # Check if package is installed
    if ! python3 -c "import $import_name" 2>/dev/null; then
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

