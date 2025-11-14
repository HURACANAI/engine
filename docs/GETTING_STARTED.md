# 🚀 Getting Started with Huracan Engine

**Welcome!** This guide will get you up and running in minutes.

---

## ⚡ Quick Start (One Command)

### Run Everything Automatically

**On Mac/Linux:**
```bash
./scripts/start.sh
```

**On Windows or any platform:**
```bash
python scripts/start.py
```

That's it! The script automatically:
- ✅ Checks Python version (requires 3.8+)
- ✅ Creates virtual environment
- ✅ Installs all dependencies
- ✅ Sets up environment
- ✅ Runs the engine

---

## 📋 Prerequisites

- **Python 3.8+** installed
- **Internet connection** (for dependencies)

### Optional (for specific features):
```bash
export DROPBOX_ACCESS_TOKEN="your_token"      # For Dropbox sync
export DATABASE_URL="postgresql://..."        # For database features
export TELEGRAM_TOKEN="your_token"            # For Telegram notifications
export TELEGRAM_CHAT_ID="your_chat_id"        # For Telegram notifications
```

---

## 🎯 Common Tasks

### 1. Download Market Data

Download top 250 coins for training:

```bash
export DROPBOX_ACCESS_TOKEN="your_token"

python scripts/simple_download_candles.py \
  --top 250 \
  --days 1095 \
  --timeframe 1h
```

### 2. Train Models

Train on top 3 coins (BTC, ETH, SOL):

```bash
python scripts/test_end_to_end_training.py \
  --symbols BTC/USDT ETH/USDT SOL/USDT \
  --days 1095 \
  --timeframe 1h
```

### 3. Monitor Training

Start the web dashboard:

```bash
# Terminal 1: Start training
python scripts/train_sol_full.py

# Terminal 2: Start dashboard
python scripts/web_dashboard_server.py

# Open browser
open http://localhost:5055/
```

### 4. Run Daily Training

```bash
python scripts/run_daily.py
```

---

## 📚 Next Steps

- **[Architecture Guide](architecture/ARCHITECTURE.md)** - Understand the system structure
- **[Dashboard Guide](DASHBOARD_GUIDE.md)** - Monitor training and results
- **[Dropbox Guide](DROPBOX_GUIDE.md)** - Configure Dropbox sync
- **[API Reference](API_REFERENCE.md)** - Use the engine programmatically

---

## 🆘 Troubleshooting

### Python Version Issues
```bash
python3 --version  # Should be 3.8+
```

### Virtual Environment Issues
```bash
# Remove and recreate
rm -rf venv
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
# or
venv\Scripts\activate  # On Windows
```

### Dependency Issues
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Import Errors
```bash
# Make sure PYTHONPATH is set
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

---

## 📖 Documentation Structure

```
docs/
├── GETTING_STARTED.md      # This file - start here!
├── architecture/           # Architecture standards
├── guides/                 # Feature guides
├── setup/                  # Detailed setup guides
└── reports/                # Implementation reports
```

---

**Ready to go?** Run `./scripts/start.sh` or `python scripts/start.py` to begin!

