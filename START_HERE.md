# 🚀 Simple Start Guide

## One Command to Run Everything

This project now has a **unified startup script** that works the same way on any laptop.

### Quick Start

**On Mac/Linux:**
```bash
./start.sh
```

**On Windows:**
```bash
python start.py
```

**Or use Python on any platform:**
```bash
python start.py
```

### What It Does

The startup script automatically:

1. ✅ **Checks Python version** (requires 3.8+)
2. ✅ **Creates virtual environment** (if it doesn't exist)
3. ✅ **Upgrades pip** to latest version
4. ✅ **Installs all dependencies** from `requirements.txt`
5. ✅ **Sets up environment variables** (PYTHONPATH, etc.)
6. ✅ **Runs the engine** using the correct entry point

### Requirements

- **Python 3.8 or higher** installed on your system
- **Internet connection** (for downloading dependencies on first run)

### Optional Environment Variables

These are optional and only needed for specific features:

```bash
export DROPBOX_ACCESS_TOKEN="your_token"      # For Dropbox sync
export DATABASE_URL="postgresql://..."        # For database features
export TELEGRAM_TOKEN="your_token"            # For Telegram notifications
export TELEGRAM_CHAT_ID="your_chat_id"        # For Telegram notifications
```

### First Run

On the first run, the script will:
- Create a virtual environment in `venv/`
- Install all dependencies (this may take a few minutes)
- Then run the engine

Subsequent runs will be faster since dependencies are already installed.

### Troubleshooting

**If you get permission errors:**
```bash
chmod +x start.sh
```

**If Python is not found:**
- Make sure Python 3.8+ is installed
- On Mac: `brew install python3`
- On Ubuntu: `sudo apt install python3 python3-venv`
- On Windows: Download from python.org

**If dependencies fail to install:**
- Check your internet connection
- Try running: `pip install --upgrade pip`
- Check `requirements.txt` exists in the project root

### What Gets Created

- `venv/` - Virtual environment (created automatically)
- All dependencies installed in the virtual environment
- No changes to your system Python

### Running on Different Laptops

The script works the same way on:
- ✅ Mac (macOS)
- ✅ Linux (Ubuntu, Debian, etc.)
- ✅ Windows (using `start.py`)

Just run the same command and it will work!

---

## Alternative: Manual Setup

If you prefer to set up manually:

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

# 5. Run the engine
python run_daily.py
```

But using `start.sh` or `start.py` is much simpler! 🎉

