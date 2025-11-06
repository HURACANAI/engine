# 🚀 RunPod Quick Start Guide

## 📊 Recommended Hardware

### **For 20 Coins (HTF Trading)**
- **GPU**: RTX 3090 (24GB VRAM) or RTX 4090 (24GB VRAM)
- **CPU**: 8-16 cores
- **RAM**: 32GB
- **Storage**: 100GB SSD
- **Cost**: ~$0.50-1.00/hour (~$1-2/day)

### **For All Coins**
- **GPU**: RTX 4090 (24GB VRAM) or A100 (40GB VRAM)
- **CPU**: 16-32 cores
- **RAM**: 64GB
- **Storage**: 200GB SSD
- **Cost**: ~$1.00-2.00/hour (~$3-5/day)

---

## 🚀 Quick Setup (5 Steps)

### Step 1: Create RunPod Pod

1. Go to https://www.runpod.io/
2. **Create Pod**:
   - **Template**: `RunPod PyTorch 2.1.0`
   - **GPU**: RTX 3090 (24GB) or RTX 4090 (24GB)
   - **CPU**: 8-16 cores
   - **RAM**: 32GB
   - **Storage**: 100GB SSD

### Step 2: Connect to Pod

```bash
# SSH into your RunPod instance
ssh root@<your-pod-ip>
```

### Step 3: Upload Your Code

**Option A: Git Clone**
```bash
git clone <your-repo-url>
cd engine
```

**Option B: Upload via RunPod File Manager**
- Upload your code via RunPod's web interface
- Extract to `/workspace/engine`

### Step 4: Run Setup Script

```bash
cd /workspace/engine
chmod +x scripts/runpod_setup.sh
./scripts/runpod_setup.sh
```

**This will:**
- ✅ Install PostgreSQL
- ✅ Create database
- ✅ Install Python dependencies
- ✅ Setup database schema
- ✅ Configure for GPU (if available)
- ✅ Update config files

### Step 5: Update Config

Edit `config/base.yaml` or use `config/runpod.yaml`:

```yaml
postgres:
  dsn: "postgresql://haq:your_password@localhost:5432/huracan"

training:
  rl_agent:
    device: "cuda"  # Use GPU

  optimization:
    parallel_processing:
      num_workers: 8  # Adjust based on CPU cores
```

### Step 6: Run the Engine

```bash
# Test with single coin first
python -m src.cloud.training.pipelines.daily_retrain
```

---

## 📊 Resource Usage

### **20 Coins (HTF - Daily Candles)**
- **Peak RAM**: 8-12GB
- **Total Time**: 1-2 hours
- **Database**: 5-10GB
- **Storage**: 20-30GB

### **All Coins (~500 coins)**
- **Peak RAM**: 32-48GB
- **Total Time**: 4-8 hours
- **Database**: 50-100GB
- **Storage**: 100-200GB

---

## 🔧 Configuration Tips

### **Use GPU for RL Agent**
```yaml
training:
  rl_agent:
    device: "cuda"  # Much faster than CPU
    batch_size: 128  # Increase for GPU
```

### **Optimize Ray Workers**
```yaml
optimization:
  parallel_processing:
    num_workers: 8  # Use: nproc / 2
```

### **Enable Caching**
```yaml
optimization:
  caching:
    enabled: true
    max_size: 2000  # Increase for more cache
```

---

## 📱 Monitor on Telegram

You'll receive notifications for:
- ✅ System startup/shutdown
- ✅ Trade executions
- ✅ Learning updates
- ✅ Validation failures (CRITICAL!)
- ✅ Health checks
- ✅ Errors

---

## 💰 Cost Estimates

### **20 Coins (Daily)**
- **RTX 3090**: ~$1.00/day
- **RTX 4090**: ~$1.40/day
- **CPU-Only**: ~$0.90/day

### **All Coins (Daily)**
- **RTX 3090**: ~$3.00/day
- **RTX 4090**: ~$3.50/day
- **CPU-Only**: ~$2.40/day

---

## ✅ You're Ready!

1. ✅ Create RunPod Pod (RTX 3090 recommended)
2. ✅ Run setup script (`./scripts/runpod_setup.sh`)
3. ✅ Update config (database password, GPU)
4. ✅ Test with 1 coin first
5. ✅ Scale to 20 coins after validation
6. ✅ Monitor Telegram for updates

**Everything is ready for RunPod deployment!** 🚀

