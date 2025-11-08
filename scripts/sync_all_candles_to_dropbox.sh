#!/bin/bash
# Simple script to download all candles and upload to Dropbox
# Usage: ./scripts/sync_all_candles_to_dropbox.sh [days]

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Set environment
export HURACAN_ENV="${HURACAN_ENV:-local}"

# Get days from argument or default to 150
DAYS="${1:-150}"

echo "🚀 Downloading all candles and uploading to Dropbox"
echo "   Days: $DAYS"
echo "   Environment: $HURACAN_ENV"
echo ""

# Run the download script
python scripts/download_and_upload_candles.py \
    --all-symbols \
    --days "$DAYS" \
    --timeframe 1m

echo ""
echo "✅ Done! All candles downloaded and uploaded to Dropbox"


