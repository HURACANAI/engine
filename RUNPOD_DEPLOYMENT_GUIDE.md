# 🚀 RunPod Deployment Guide

## 📊 Hardware Requirements

### **Recommended Configuration (20 Coins)**

#### **Option 1: CPU-Only (Budget-Friendly)**
- **CPU**: 8-16 cores (Intel/AMD)
- **RAM**: 32GB
- **Storage**: 100GB SSD
- **Network**: Good connection (for exchange API)
- **Cost**: ~$0.30-0.50/hour

**Why:**
- 23 alpha engines running in parallel
- 6 Ray workers for parallel processing
- ML models (XGBoost, LightGBM, Random Forest)
- PostgreSQL database with pgvector
- Data processing for 20 coins

#### **Option 2: GPU-Enabled (Recommended for RL)**
- **CPU**: 8-16 cores
- **RAM**: 32GB
- **GPU**: **1x RTX 3090 (24GB VRAM)** or RTX 4090 (24GB VRAM) ⭐ **1 GPU is optimal**
- **Storage**: 100GB SSD
- **Network**: Good connection
- **Cost**: ~$0.50-1.00/hour

**Why:**
- RL agent uses PyTorch (single GPU per coin)
- Neural networks (Transformer, Mixture of Experts)
- Ray handles parallelization (20 coins in parallel)
- **1 GPU is sufficient** - 2-3 GPUs add cost without significant speedup
- **GPU utilization: 60-80%** (good utilization with 1 GPU)

#### **Option 3: High-Performance (All Coins)**
- **CPU**: 16-32 cores
- **RAM**: 64GB
- **GPU**: 1x RTX 4090 (24GB VRAM) or A100 (40GB VRAM)
- **Storage**: 200GB SSD
- **Network**: Excellent connection
- **Cost**: ~$1.00-2.00/hour

**Why:**
- Scaling to all Binance coins
- More parallel processing
- Larger database
- Faster training

---

## 🎯 Recommended: Option 2 (GPU-Enabled)

**Best balance of performance and cost for 20 coins.**

---

## 📋 RunPod Setup Steps

### Step 1: Create RunPod Pod

1. **Go to RunPod**: https://www.runpod.io/
2. **Create Pod**:
   - **Template**: `RunPod PyTorch 2.1.0`
   - **GPU**: RTX 3090 (24GB) or RTX 4090 (24GB)
   - **CPU**: 8-16 cores
   - **RAM**: 32GB
   - **Storage**: 100GB SSD
   - **Network Volume**: Optional (for persistent data)

### Step 2: Connect to Pod

```bash
# SSH into your RunPod instance
ssh root@<your-pod-ip>
```

### Step 3: Install Dependencies

```bash
# Update system
apt-get update && apt-get upgrade -y

# Install PostgreSQL
apt-get install -y postgresql postgresql-contrib

# Install Python dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install xgboost lightgbm scikit-learn polars numpy pandas structlog psycopg2-binary ray requests
```

### Step 4: Setup PostgreSQL

```bash
# Start PostgreSQL
systemctl start postgresql
systemctl enable postgresql

# Create database
sudo -u postgres psql -c "CREATE DATABASE huracan;"
sudo -u postgres psql -c "CREATE USER haq WITH PASSWORD 'your_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE huracan TO haq;"

# Install pgvector extension
sudo -u postgres psql -d huracan -c "CREATE EXTENSION vector;"
```

### Step 5: Clone Your Repository

```bash
# Clone your repository
git clone <your-repo-url>
cd engine

# Or upload your code via RunPod's file manager
```

### Step 6: Configure Settings

Edit `config/base.yaml`:

```yaml
postgres:
  dsn: "postgresql://haq:your_password@localhost:5432/huracan"

training:
  rl_agent:
    device: "cuda"  # Use GPU instead of CPU

  optimization:
    parallel_processing:
      num_workers: 8  # Increase for more cores
      use_ray: true

ray:
  address: null  # Local Ray cluster
  namespace: "huracan-engine"
  runtime_env: {}
```

### Step 7: Run Database Setup

```bash
# Run database setup script
./scripts/setup_database.sh
```

### Step 8: Test Run

```bash
# Test with single coin first
python -m src.cloud.training.pipelines.daily_retrain
```

---

## 🔧 Configuration for RunPod

### Update `config/base.yaml` for RunPod:

```yaml
training:
  rl_agent:
    device: "cuda"  # Use GPU for RL agent
    batch_size: 128  # Increase for GPU
    n_epochs: 10

  optimization:
    parallel_processing:
      num_workers: 8  # Adjust based on CPU cores
      use_ray: true

ray:
  address: null  # Local Ray cluster
  namespace: "huracan-engine"
  runtime_env:
    working_dir: "/workspace/engine"
```

---

## 📊 Resource Usage Estimates

### **For 20 Coins (HTF - Daily Candles)**

**Per Coin:**
- **Data Loading**: ~100MB RAM, 1-2 minutes
- **Feature Engineering**: ~200MB RAM, 2-3 minutes
- **Model Training**: ~500MB RAM, 5-10 minutes
- **RL Training**: ~1GB RAM, 10-20 minutes (with GPU)
- **Validation**: ~200MB RAM, 2-3 minutes

**Total (20 Coins):**
- **Peak RAM**: ~8-12GB (with parallel processing)
- **Total Time**: 1-2 hours (with parallel processing)
- **Database**: ~5-10GB (grows over time)
- **Storage**: ~20-30GB (data cache + models)

### **For All Binance Coins (~500 coins)**

**Total:**
- **Peak RAM**: ~32-48GB
- **Total Time**: 4-8 hours (with parallel processing)
- **Database**: ~50-100GB
- **Storage**: ~100-200GB

---

## 🎯 Recommended RunPod Templates

### **Template 1: RTX 3090 (24GB VRAM)**
- **GPU**: 1x RTX 3090
- **CPU**: 8-16 cores
- **RAM**: 32GB
- **Storage**: 100GB SSD
- **Cost**: ~$0.50-0.70/hour
- **Best for**: 20 coins with GPU acceleration

### **Template 2: RTX 4090 (24GB VRAM)**
- **GPU**: 1x RTX 4090
- **CPU**: 8-16 cores
- **RAM**: 32GB
- **Storage**: 100GB SSD
- **Cost**: ~$0.70-1.00/hour
- **Best for**: All coins with GPU acceleration

### **Template 3: CPU-Only (Budget)**
- **CPU**: 16-32 cores
- **RAM**: 32-64GB
- **Storage**: 100GB SSD
- **Cost**: ~$0.30-0.50/hour
- **Best for**: 20 coins without GPU (slower RL training)

---

## 🚀 Quick Start Script for RunPod

Create `runpod_setup.sh`:

```bash
#!/bin/bash

# RunPod Setup Script

echo "🚀 Setting up Huracan Engine on RunPod..."

# 1. Install PostgreSQL
apt-get update
apt-get install -y postgresql postgresql-contrib

# 2. Start PostgreSQL
systemctl start postgresql
systemctl enable postgresql

# 3. Create database
sudo -u postgres psql -c "CREATE DATABASE huracan;"
sudo -u postgres psql -c "CREATE USER haq WITH PASSWORD 'your_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE huracan TO haq;"
sudo -u postgres psql -d huracan -c "CREATE EXTENSION vector;"

# 4. Install Python dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install xgboost lightgbm scikit-learn polars numpy pandas structlog psycopg2-binary ray requests

# 5. Setup database
cd /workspace/engine
./scripts/setup_database.sh

# 6. Update config for RunPod
sed -i 's/device: "cpu"/device: "cuda"/' config/base.yaml
sed -i 's/num_workers: 6/num_workers: 8/' config/base.yaml

echo "✅ Setup complete! Run: python -m src.cloud.training.pipelines.daily_retrain"
```

---

## 📊 Monitoring on RunPod

### Check Resource Usage

```bash
# CPU usage
htop

# GPU usage (if using GPU)
nvidia-smi

# Memory usage
free -h

# Disk usage
df -h
```

### Watch Logs

```bash
# Watch training logs
tail -f logs/engine_monitoring_*.log

# Watch system logs
journalctl -u postgresql -f
```

---

## 🔧 Optimization Tips

### 1. **Use GPU for RL Agent**
```yaml
training:
  rl_agent:
    device: "cuda"  # Much faster than CPU
```

### 2. **Increase Ray Workers**
```yaml
optimization:
  parallel_processing:
    num_workers: 8  # Match your CPU cores
```

### 3. **Use Network Volume for Data**
- Store data cache on network volume
- Faster I/O, persistent across pod restarts

### 4. **Enable Caching**
```yaml
optimization:
  caching:
    enabled: true
    max_size: 2000  # Increase for more cache
```

---

## 💰 Cost Estimates

### **20 Coins (Daily Training)**
- **RTX 3090**: ~$0.50/hour × 2 hours = **$1.00/day**
- **RTX 4090**: ~$0.70/hour × 2 hours = **$1.40/day**
- **CPU-Only**: ~$0.30/hour × 3 hours = **$0.90/day**

### **All Coins (Daily Training)**
- **RTX 3090**: ~$0.50/hour × 6 hours = **$3.00/day**
- **RTX 4090**: ~$0.70/hour × 5 hours = **$3.50/day**
- **CPU-Only**: ~$0.30/hour × 8 hours = **$2.40/day**

---

## ✅ Recommended Setup

**For 20 Coins:**
- **Template**: RTX 3090 (24GB VRAM)
- **CPU**: 8-16 cores
- **RAM**: 32GB
- **Storage**: 100GB SSD
- **Cost**: ~$1.00/day

**For All Coins:**
- **Template**: RTX 4090 (24GB VRAM)
- **CPU**: 16-32 cores
- **RAM**: 64GB
- **Storage**: 200GB SSD
- **Cost**: ~$3.50/day

---

## 🎯 Next Steps

1. **Create RunPod Pod** (RTX 3090 recommended)
2. **Run setup script** (`runpod_setup.sh`)
3. **Update config** (GPU, workers, database)
4. **Test with 1 coin** first
5. **Scale to 20 coins** after validation
6. **Monitor Telegram** for updates

**You're ready to deploy!** 🚀

