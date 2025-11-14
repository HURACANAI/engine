#!/bin/bash
# Monitor SOL training progress

LOG_FILE="/tmp/sol_training_test.log"

echo "Monitoring SOL training progress..."
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    clear
    echo "=== SOL Training Progress ==="
    echo "Time: $(date)"
    echo ""
    
    # Check if process is running
    if pgrep -f "train_sol.py" > /dev/null; then
        echo "✅ Training process is RUNNING"
    else
        echo "❌ Training process is NOT running"
    fi
    
    echo ""
    echo "=== Recent Log Events ==="
    tail -20 "$LOG_FILE" 2>/dev/null | grep -E "(download|feature|labeling|training|error|Error|ERROR|failed|Failed|Published|Status|Reason|trades_oos|sharpe|training_complete)" | tail -10
    
    echo ""
    echo "=== Key Metrics (if available) ==="
    tail -200 "$LOG_FILE" 2>/dev/null | grep -E "(total_splits|training_split|all_splits_training_complete|final_production_model|Published|Status)" | tail -5
    
    echo ""
    echo "=== Errors (if any) ==="
    tail -500 "$LOG_FILE" 2>/dev/null | grep -E "(error|Error|ERROR|failed|Failed|TypeError|Exception)" | tail -5
    
    echo ""
    echo "=== Log File Size ==="
    wc -l "$LOG_FILE" 2>/dev/null
    
    sleep 10
done

