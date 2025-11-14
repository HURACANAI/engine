#!/bin/bash
# Simple unified startup script for Huracan Engine
# Works consistently across different laptops
# Usage: ./start.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Huracan Engine - Unified Startup    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Check Python version
echo -e "${YELLOW}[1/6]${NC} Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${RED}❌ Python 3.8+ required. Found: Python $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python $PYTHON_VERSION found${NC}"
echo ""

# Step 2: Setup virtual environment
echo -e "${YELLOW}[2/6]${NC} Setting up virtual environment..."
VENV_DIR="$PROJECT_ROOT/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${GREEN}✅ Virtual environment already exists${NC}"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✅ Virtual environment activated${NC}"
echo ""

# Step 3: Upgrade pip
echo -e "${YELLOW}[3/6]${NC} Upgrading pip..."
pip install --quiet --upgrade pip setuptools wheel
echo -e "${GREEN}✅ pip upgraded${NC}"
echo ""

# Step 4: Install dependencies
echo -e "${YELLOW}[4/6]${NC} Installing dependencies..."
if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    pip install --quiet -r "$PROJECT_ROOT/requirements.txt" || {
        echo -e "${YELLOW}⚠️  Some dependencies may have failed, continuing anyway...${NC}"
    }
    echo -e "${GREEN}✅ Dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠️  requirements.txt not found, skipping dependency installation${NC}"
fi
echo ""

# Step 5: Set environment variables
echo -e "${YELLOW}[5/6]${NC} Setting up environment..."

# Set PYTHONPATH to project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Check for required environment variables and set defaults if needed
if [ -z "$DROPBOX_ACCESS_TOKEN" ]; then
    echo -e "${YELLOW}⚠️  DROPBOX_ACCESS_TOKEN not set (optional for some operations)${NC}"
fi

if [ -z "$DATABASE_URL" ]; then
    echo -e "${YELLOW}⚠️  DATABASE_URL not set (using default if configured)${NC}"
fi

if [ -z "$TELEGRAM_TOKEN" ]; then
    echo -e "${YELLOW}⚠️  TELEGRAM_TOKEN not set (optional for notifications)${NC}"
fi

echo -e "${GREEN}✅ Environment configured${NC}"
echo ""

# Step 6: Start Dashboard
echo -e "${YELLOW}[6/7]${NC} Starting comprehensive dashboard..."
DASHBOARD_SCRIPT="$PROJECT_ROOT/scripts/comprehensive_dashboard_server.py"
if [ -f "$DASHBOARD_SCRIPT" ]; then
    python3 "$DASHBOARD_SCRIPT" > /tmp/dashboard.log 2>&1 &
    DASHBOARD_PID=$!
    echo -e "${GREEN}✅ Dashboard started (PID: $DASHBOARD_PID)${NC}"
    echo -e "${BLUE}   📊 Dashboard URL: http://localhost:5055/${NC}"
    echo -e "${BLUE}   📝 Logs: /tmp/dashboard.log${NC}"
    sleep 2  # Give dashboard time to start
else
    echo -e "${YELLOW}⚠️  Dashboard script not found, skipping...${NC}"
    DASHBOARD_PID=""
fi
echo ""

# Step 7: Run the application
echo -e "${YELLOW}[7/7]${NC} Starting Huracan Engine..."
echo ""
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo -e "${BLUE}Starting Engine...${NC}"
echo -e "${BLUE}Dashboard: http://localhost:5055/${NC}"
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo ""

# Check which entry point to use
if [ -f "$PROJECT_ROOT/run_daily.py" ]; then
    python3 "$PROJECT_ROOT/run_daily.py"
elif [ -f "$PROJECT_ROOT/engine/run.py" ]; then
    python3 "$PROJECT_ROOT/engine/run.py"
else
    # Fallback to module execution
    python3 -m src.cloud.training.pipelines.daily_retrain
fi

EXIT_CODE=$?

# Cleanup: Stop dashboard if it was started
if [ -n "$DASHBOARD_PID" ]; then
    echo ""
    echo -e "${YELLOW}Stopping dashboard (PID: $DASHBOARD_PID)...${NC}"
    kill $DASHBOARD_PID 2>/dev/null || true
    echo -e "${GREEN}✅ Dashboard stopped${NC}"
fi

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║   ✅ Engine finished successfully      ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
else
    echo -e "${RED}╔════════════════════════════════════════╗${NC}"
    echo -e "${RED}║   ❌ Engine exited with error code $EXIT_CODE   ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════╝${NC}"
fi

exit $EXIT_CODE

