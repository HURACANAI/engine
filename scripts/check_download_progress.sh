#!/bin/bash
# Quick script to check download progress

LOG_FILE="/tmp/download_250_1d.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    echo "   Download may not be running."
    exit 1
fi

# Count uploads
UPLOADED=$(grep -c "📤 Uploaded" "$LOG_FILE" 2>/dev/null || echo "0")
TOTAL=249

# Get current coin being processed
CURRENT=$(tail -20 "$LOG_FILE" | grep -o "\[[0-9]*/249\]" | tail -1 || echo "[?/249]")

# Check if process is still running
if pgrep -f "simple_download_candles.py" > /dev/null; then
    STATUS="🟢 RUNNING"
else
    STATUS="🔴 STOPPED"
fi

echo "=========================================="
echo "📊 Download Progress"
echo "=========================================="
echo "Status: $STATUS"
echo "Progress: $CURRENT"
echo "Uploaded: $UPLOADED / $TOTAL coins"
echo "Remaining: $((TOTAL - UPLOADED)) coins"
echo "=========================================="
echo ""
echo "Recent activity:"
tail -10 "$LOG_FILE" | grep -E "(\[.*\]|✅|📤|⚠️)" | tail -5

