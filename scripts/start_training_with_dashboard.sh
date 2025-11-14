#!/bin/bash
# Start training with live web dashboard
# Usage: ./scripts/start_training_with_dashboard.sh --symbol BTC/USDT

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Huracan Training with Live Dashboard                  ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if symbol is provided
SYMBOL="${1:-BTC/USDT}"
if [[ "$1" == "--symbol" ]]; then
    SYMBOL="$2"
fi

echo -e "${GREEN}Symbol:${NC} $SYMBOL"
echo -e "${GREEN}Dashboard:${NC} http://localhost:5055/"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start web dashboard in background
echo -e "${YELLOW}[1/2]${NC} Starting web dashboard..."
python3 scripts/web_dashboard_server.py > /tmp/dashboard.log 2>&1 &
DASHBOARD_PID=$!
echo -e "${GREEN}✅ Dashboard started (PID: $DASHBOARD_PID)${NC}"
echo "   Logs: /tmp/dashboard.log"
echo "   URL: http://localhost:5055/"
echo ""

# Wait a moment for dashboard to start
sleep 3

# Start training
echo -e "${YELLOW}[2/2]${NC} Starting training..."
echo ""
python3 scripts/train_single_coin_with_monitoring.py --symbol "$SYMBOL"

TRAINING_EXIT_CODE=$?

# Cleanup
echo ""
echo -e "${YELLOW}Stopping dashboard...${NC}"
kill $DASHBOARD_PID 2>/dev/null || true

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Training completed successfully!${NC}"
    exit 0
else
    echo -e "${RED}❌ Training failed with exit code $TRAINING_EXIT_CODE${NC}"
    exit $TRAINING_EXIT_CODE
fi

