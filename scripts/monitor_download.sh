#!/bin/bash
# Monitor download progress

LOG_FILE="/tmp/robust_download.log"
PROGRESS_FILE="data/download_progress.json"

echo "=========================================="
echo "📊 DOWNLOAD MONITOR"
echo "=========================================="

# Check if process is running
if pgrep -f "robust_download_top250.py" > /dev/null; then
    echo "Status: 🟢 RUNNING"
else
    echo "Status: 🔴 STOPPED"
fi

echo ""

# Check progress file
if [ -f "$PROGRESS_FILE" ]; then
    echo "Progress File: $PROGRESS_FILE"
    COMPLETED=$(python3 -c "import json; p=json.load(open('$PROGRESS_FILE')); print(len(p.get('completed', [])))" 2>/dev/null || echo "0")
    FAILED=$(python3 -c "import json; p=json.load(open('$PROGRESS_FILE')); print(len(p.get('failed', [])))" 2>/dev/null || echo "0")
    echo "   Completed: $COMPLETED coins"
    echo "   Failed: $FAILED coins"
    echo ""
fi

# Check log file
if [ -f "$LOG_FILE" ]; then
    echo "Recent Activity:"
    tail -10 "$LOG_FILE" | grep -E "(\[.*\]|✅|❌|📥)" | tail -5
    echo ""
    
    # Count downloads
    DOWNLOADED=$(grep -c "✅" "$LOG_FILE" 2>/dev/null || echo "0")
    FAILED_COUNT=$(grep -c "❌" "$LOG_FILE" 2>/dev/null || echo "0")
    echo "   Downloaded: $DOWNLOADED"
    echo "   Failed: $FAILED_COUNT"
else
    echo "Log file not found: $LOG_FILE"
fi

echo "=========================================="
echo ""
echo "To watch live: tail -f $LOG_FILE"
echo "To check progress: cat $PROGRESS_FILE"

