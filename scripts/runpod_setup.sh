#!/bin/bash

# RunPod Setup Script for Huracan Engine
# This script sets up the engine on a RunPod instance

set -e  # Exit on error

echo "🚀 Setting up Huracan Engine on RunPod..."
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Update system
echo -e "${YELLOW}1️⃣  Updating system...${NC}"
apt-get update -qq
apt-get upgrade -y -qq

# 2. Install PostgreSQL
echo -e "${YELLOW}2️⃣  Installing PostgreSQL...${NC}"
apt-get install -y postgresql postgresql-contrib -qq

# 3. Start PostgreSQL
echo -e "${YELLOW}3️⃣  Starting PostgreSQL...${NC}"
systemctl start postgresql
systemctl enable postgresql

# 4. Create database and user
echo -e "${YELLOW}4️⃣  Creating database...${NC}"
sudo -u postgres psql -c "CREATE DATABASE huracan;" 2>/dev/null || echo "Database already exists"
sudo -u postgres psql -c "CREATE USER haq WITH PASSWORD 'huracan123';" 2>/dev/null || echo "User already exists"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE huracan TO haq;"
sudo -u postgres psql -d huracan -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 5. Install Python dependencies
echo -e "${YELLOW}5️⃣  Installing Python dependencies...${NC}"
pip install --upgrade pip -q
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
pip install xgboost lightgbm scikit-learn polars numpy pandas structlog psycopg2-binary ray requests -q

# 6. Check if we're in the engine directory
if [ ! -f "config/base.yaml" ]; then
    echo -e "${YELLOW}⚠️  Not in engine directory. Please cd to engine directory first.${NC}"
    exit 1
fi

# 7. Setup database schema
echo -e "${YELLOW}6️⃣  Setting up database schema...${NC}"
if [ -f "scripts/setup_database.sh" ]; then
    chmod +x scripts/setup_database.sh
    ./scripts/setup_database.sh
else
    echo -e "${YELLOW}⚠️  Database setup script not found. Run manually: ./scripts/setup_database.sh${NC}"
fi

# 8. Update config for RunPod (if GPU available)
echo -e "${YELLOW}7️⃣  Updating config for RunPod...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ GPU detected! Updating config to use CUDA...${NC}"
    sed -i 's/device: "cpu"/device: "cuda"/' config/base.yaml 2>/dev/null || echo "Config already updated"
    
    # Get CPU cores
    CPU_CORES=$(nproc)
    WORKERS=$((CPU_CORES / 2))  # Use half the cores for workers
    sed -i "s/num_workers: 6/num_workers: $WORKERS/" config/base.yaml 2>/dev/null || echo "Workers already configured"
else
    echo -e "${YELLOW}⚠️  No GPU detected. Using CPU.${NC}"
fi

# 9. Update PostgreSQL DSN
echo -e "${YELLOW}8️⃣  Updating PostgreSQL DSN...${NC}"
sed -i 's|dsn: "postgresql://haq@localhost:5432/huracan"|dsn: "postgresql://haq:huracan123@localhost:5432/huracan"|' config/base.yaml 2>/dev/null || echo "DSN already configured"

# 10. Create logs directory
echo -e "${YELLOW}9️⃣  Creating logs directory...${NC}"
mkdir -p logs

# 11. Test GPU (if available)
if command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}🔟 Testing GPU...${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
fi

echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "📋 Next steps:"
echo "1. Update config/base.yaml with your PostgreSQL password"
echo "2. Update config/base.yaml with your Telegram chat_id (if not already done)"
echo "3. Test with: python -m src.cloud.training.pipelines.daily_retrain"
echo ""
echo "📊 Resource usage:"
echo "- CPU cores: $(nproc)"
echo "- RAM: $(free -h | grep Mem | awk '{print $2}')"
if command -v nvidia-smi &> /dev/null; then
    echo "- GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "- VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
fi
echo ""

