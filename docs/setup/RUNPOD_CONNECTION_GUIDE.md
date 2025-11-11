# 🔌 RunPod Connection Guide - Connect to Your Engine

## 🚀 Step-by-Step: Connect RunPod to Engine

### **Step 1: Create RunPod Pod**

1. **Go to RunPod**: https://www.runpod.io/
2. **Click "Pods"** → **"Create Pod"**
3. **Select Configuration**:
   - **Template**: `RunPod PyTorch 2.1.0` (or latest PyTorch)
   - **GPU**: RTX 3090 (24GB) or RTX 4090 (24GB) - **1 GPU is optimal**
   - **CPU**: 8-16 cores
   - **RAM**: 32GB
   - **Storage**: 100GB SSD
   - **Pricing**: **Interruptible/Spot** (50-70% cheaper!)
4. **Click "Deploy"** and wait for pod to start (~2-3 minutes)

---

### **Step 2: Connect to Your Pod**

#### **Option A: SSH (Recommended)**

1. **Get SSH Connection Info**:
   - In RunPod dashboard, click on your pod
   - Copy the **SSH command** (looks like: `ssh root@<ip-address>`)
   - Or note the **IP address** and **port**

2. **Connect via Terminal**:
   ```bash
   # Use the SSH command from RunPod dashboard
   ssh root@<your-pod-ip>
   
   # Or if you have a key:
   ssh -i ~/.ssh/your-key.pem root@<your-pod-ip>
   ```

3. **Verify Connection**:
   ```bash
   # Check GPU
   nvidia-smi
   
   # Check CPU
   nproc
   
   # Check RAM
   free -h
   ```

#### **Option B: RunPod Web Terminal**

1. **In RunPod Dashboard**:
   - Click on your pod
   - Click **"Connect"** → **"Web Terminal"**
   - Opens browser-based terminal

---

### **Step 3: Upload Your Engine Code**

#### **Option A: Git Clone (Recommended)**

```bash
# Install git if needed
apt-get update
apt-get install -y git

# Clone your repository
cd /workspace
git clone <your-repo-url>
cd engine

# Or if using SSH:
git clone git@github.com:your-username/your-repo.git
cd engine
```

#### **Option B: Upload via RunPod File Manager**

1. **In RunPod Dashboard**:
   - Click on your pod
   - Click **"Connect"** → **"File Manager"**
   - Navigate to `/workspace`
   - Upload your code (zip file or individual files)
   - Extract if needed:
     ```bash
     cd /workspace
     unzip your-code.zip
     cd engine
     ```

#### **Option C: SCP (From Your Local Machine)**

```bash
# From your local machine
scp -r /path/to/engine root@<your-pod-ip>:/workspace/engine
```

---

### **Step 4: Run Setup Script**

```bash
# Navigate to engine directory
cd /workspace/engine

# Make setup script executable
chmod +x scripts/runpod_setup.sh

# Run setup script
./scripts/runpod_setup.sh
```

**This will:**
- ✅ Install PostgreSQL
- ✅ Create database
- ✅ Install Python dependencies (PyTorch, XGBoost, etc.)
- ✅ Setup database schema
- ✅ Configure for GPU (if available)
- ✅ Update config files

**Expected output:**
```
🚀 Setting up Huracan Engine on RunPod...
1️⃣  Updating system...
2️⃣  Installing PostgreSQL...
3️⃣  Starting PostgreSQL...
4️⃣  Creating database...
5️⃣  Installing Python dependencies...
6️⃣  Setting up database schema...
7️⃣  Updating config for RunPod...
✅ Setup complete!
```

---

### **Step 5: Configure Settings**

#### **5.1 Update PostgreSQL Password**

```bash
# Edit config file
nano config/base.yaml

# Or use vi:
vi config/base.yaml
```

**Update PostgreSQL DSN:**
```yaml
postgres:
  dsn: "postgresql://haq:huracan123@localhost:5432/huracan"
  # Change 'huracan123' to your secure password
```

**Or use the RunPod config:**
```bash
# Copy RunPod config
cp config/runpod.yaml config/base.yaml
```

#### **5.2 Verify GPU Configuration**

```bash
# Check if GPU is detected
nvidia-smi

# Should show your GPU (RTX 3090/4090)
```

**Update config if needed:**
```yaml
training:
  rl_agent:
    device: "cuda"  # Use GPU (auto-detected)
    batch_size: 128  # Increased for GPU
```

#### **5.3 Verify Telegram Configuration**

```yaml
notifications:
  telegram_enabled: true
  telegram_webhook_url: "https://api.telegram.org/bot8229109041:AAFIcLRx3V50khoaEIG7WXeI1ITzy4s6hf0/sendMessage"
  telegram_chat_id: "7914196017"  # Your Telegram user ID
```

---

### **Step 6: Test Connection**

#### **6.1 Test Database Connection**

```bash
# Test PostgreSQL connection
psql -U haq -d huracan -c "SELECT version();"

# Or test via Python
python3 -c "
import psycopg2
conn = psycopg2.connect('postgresql://haq:huracan123@localhost:5432/huracan')
print('✅ Database connected!')
conn.close()
"
```

#### **6.2 Test GPU**

```bash
# Test PyTorch GPU
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"
```

**Expected output:**
```
CUDA available: True
GPU count: 1
GPU name: NVIDIA GeForce RTX 3090
```

#### **6.3 Test Engine Import**

```bash
# Test if engine imports correctly
python3 -c "
from src.cloud.training.pipelines.daily_retrain import run_daily_retrain
print('✅ Engine imports successfully!')
"
```

---

### **Step 7: Run the Engine**

#### **7.1 First Run (Test with 1 Coin)**

```bash
# Test with single coin first
# Edit config to use 1 coin for testing
nano config/base.yaml

# Change:
universe:
  target_size: 1  # Test with 1 coin first

# Run engine
python3 -m src.cloud.training.pipelines.daily_retrain
```

#### **7.2 Full Run (20 Coins)**

```bash
# Update config back to 20 coins
nano config/base.yaml

# Change:
universe:
  target_size: 20  # Full run

# Run engine
python3 -m src.cloud.training.pipelines.daily_retrain
```

#### **7.3 Run in Background (Optional)**

```bash
# Run in background with logging
nohup python3 -m src.cloud.training.pipelines.daily_retrain > training.log 2>&1 &

# Check status
tail -f training.log

# Or check process
ps aux | grep daily_retrain
```

---

### **Step 8: Monitor Progress**

#### **8.1 Check Logs**

```bash
# Watch training logs
tail -f logs/engine_monitoring_*.log

# Or check latest log
ls -lt logs/ | head -1
tail -f logs/$(ls -t logs/ | head -1)
```

#### **8.2 Check Telegram**

- You should receive notifications on Telegram:
  - ✅ System startup
  - ✅ Training progress
  - ✅ Trade executions
  - ✅ Validation results
  - ✅ Errors (if any)

#### **8.3 Check Database**

```bash
# Check training results
psql -U haq -d huracan -c "
SELECT symbol, created_at, sharpe, hit_rate 
FROM model_registry 
ORDER BY created_at DESC 
LIMIT 10;
"
```

#### **8.4 Check Resource Usage**

```bash
# GPU usage
watch -n 1 nvidia-smi

# CPU usage
htop

# Memory usage
free -h

# Disk usage
df -h
```

---

### **Step 9: Schedule Daily Runs (Optional)**

#### **Option A: Cron Job**

```bash
# Edit crontab
crontab -e

# Add daily run at 02:00 UTC
0 2 * * * cd /workspace/engine && /usr/bin/python3 -m src.cloud.training.pipelines.daily_retrain >> /workspace/engine/logs/cron.log 2>&1
```

#### **Option B: Systemd Service**

```bash
# Create service file
sudo nano /etc/systemd/system/huracan-engine.service
```

**Service file content:**
```ini
[Unit]
Description=Huracan Engine Daily Training
After=network.target postgresql.service

[Service]
Type=oneshot
User=root
WorkingDirectory=/workspace/engine
ExecStart=/usr/bin/python3 -m src.cloud.training.pipelines.daily_retrain
StandardOutput=append:/workspace/engine/logs/service.log
StandardError=append:/workspace/engine/logs/service.log

[Install]
WantedBy=multi-user.target
```

**Enable and schedule:**
```bash
# Enable service
sudo systemctl enable huracan-engine.service

# Create timer for daily run
sudo nano /etc/systemd/system/huracan-engine.timer
```

**Timer file content:**
```ini
[Unit]
Description=Daily Huracan Engine Training Timer
Requires=huracan-engine.service

[Timer]
OnCalendar=*-*-* 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

**Enable timer:**
```bash
sudo systemctl enable huracan-engine.timer
sudo systemctl start huracan-engine.timer
```

---

## 🔧 Troubleshooting

### **Issue: Can't Connect via SSH**

**Solution:**
- Check RunPod dashboard for correct IP/port
- Verify firewall settings
- Try RunPod Web Terminal instead

### **Issue: Database Connection Failed**

**Solution:**
```bash
# Check PostgreSQL is running
systemctl status postgresql

# Start PostgreSQL if needed
systemctl start postgresql

# Check connection
psql -U haq -d huracan -c "SELECT 1;"
```

### **Issue: GPU Not Detected**

**Solution:**
```bash
# Check GPU
nvidia-smi

# If not detected, check drivers
nvidia-smi --query-gpu=driver_version --format=csv

# Reinstall drivers if needed (RunPod should have them pre-installed)
```

### **Issue: Import Errors**

**Solution:**
```bash
# Install missing dependencies
pip install -r requirements.txt

# Or install individually
pip install torch xgboost lightgbm scikit-learn polars numpy pandas structlog psycopg2-binary ray requests
```

### **Issue: Out of Memory**

**Solution:**
- Reduce `batch_size` in config
- Reduce `num_workers` in config
- Use smaller dataset for testing

---

## ✅ Quick Checklist

Before running the engine, verify:

- [ ] RunPod pod is running
- [ ] SSH connection works
- [ ] Code is uploaded/cloned
- [ ] Setup script completed successfully
- [ ] PostgreSQL is running
- [ ] Database connection works
- [ ] GPU is detected (`nvidia-smi`)
- [ ] PyTorch can use GPU
- [ ] Config files are updated
- [ ] Telegram chat_id is set
- [ ] Test run with 1 coin works

---

## 🎯 Quick Start Commands

```bash
# 1. Connect to RunPod
ssh root@<your-pod-ip>

# 2. Clone code
cd /workspace
git clone <your-repo-url>
cd engine

# 3. Run setup
chmod +x scripts/runpod_setup.sh
./scripts/runpod_setup.sh

# 4. Update config
nano config/base.yaml  # Update PostgreSQL password, Telegram chat_id

# 5. Test
python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "from src.cloud.training.pipelines.daily_retrain import run_daily_retrain; print('OK')"

# 6. Run engine
python3 -m src.cloud.training.pipelines.daily_retrain
```

---

## 📱 Monitor on Telegram

You'll receive notifications for:
- ✅ System startup
- ✅ Training progress (per coin)
- ✅ Trade executions
- ✅ Validation results
- ✅ Errors (if any)
- ✅ Completion summary

**Check your Telegram - you should see notifications!** 📲

---

## 🎉 You're Connected!

Your RunPod is now connected to the engine. The engine will:
- Train on 20 coins daily
- Save models to database
- Send notifications to Telegram
- Log everything for monitoring

**Happy training!** 🚀

