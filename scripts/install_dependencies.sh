#!/bin/bash
# Install dependencies for local candle download script
# Usage: ./scripts/install_dependencies.sh

set -e

echo "📦 Installing dependencies for candle download script..."
echo ""

# Core dependencies needed for the download script
REQUIRED_PACKAGES=(
    "pydantic>=2.4.0"
    "pydantic-settings>=2.0.3"
    "polars>=0.19.0"
    "pyarrow>=15.0.0"
    "ccxt>=4.1.0"
    "structlog>=23.1.0"
    "dropbox>=12.0.0"
    "numpy>=1.26.0"
)

echo "Installing packages..."
pip install "${REQUIRED_PACKAGES[@]}"

echo ""
echo "✅ Dependencies installed successfully!"
echo ""
echo "You can now run:"
echo "  python scripts/download_and_upload_candles.py --symbols BTC/USDT ETH/USDT --days 150"

