# 📦 Dropbox Guide

Complete guide to Dropbox integration, structure, and setup.

---

## 🚀 Quick Setup

### 1. Get Access Token

**Option 1: From Dropbox App Console (Easiest)**
1. Visit: https://www.dropbox.com/developers/apps
2. Find your app (or create one)
3. Go to Settings → OAuth 2
4. Click "Generate" under "Generated access token"
5. Copy the entire token (starts with `sl.`, 1000+ characters)

**Option 2: Use Script**
```bash
python scripts/generate_dropbox_token.py
```

### 2. Set Environment Variable
```bash
export DROPBOX_ACCESS_TOKEN="your_token_here"
```

### 3. Verify Setup
```bash
python scripts/test_dropbox_token.py
```

---

## 📁 Dropbox Structure

```
/{app_folder}/                    # Default: "Runpodhuracan"
├── data/                         # Shared data (persists across days)
│   ├── candles/                 # Historical candle data
│   │   ├── BTCUSDT/
│   │   │   └── BTCUSDT_1h_*.parquet
│   │   └── ...
│   └── features/                # Feature data
│
├── models/                       # Trained models
│   ├── champions/              # Champion models
│   │   ├── latest/            # Latest champion per symbol
│   │   │   ├── BTCUSDT.bin
│   │   │   └── ...
│   │   └── archive/           # Historical champions
│   └── training/               # Training artifacts
│       └── YYYY-MM-DD/        # Dated training runs
│           └── {SYMBOL}/
│               ├── model.bin
│               └── metrics.json
│
├── hamilton/                     # Hamilton exports (live trading)
│   ├── roster.json             # Ranked coins
│   ├── champion.json           # Champion pointer
│   ├── configs/                # Per-symbol configs
│   └── active/                 # Active model pointers
│
├── exports/                      # Comprehensive exports
│   ├── trades/                 # Trade history
│   ├── metrics/                # Performance metrics
│   └── reports/                # Reports
│
└── logs/                         # Logs (dated)
    └── YYYY-MM-DD/
```

---

## 🔑 Key Directories

### `/data/candles/` - Historical Data
- **Purpose**: Shared candle data for all modules
- **Structure**: Organized by symbol
- **Usage**: Training, backtesting, analysis

### `/models/champions/latest/` - Latest Champions
- **Purpose**: Latest champion model per symbol
- **Usage**: Hamilton loads for live trading
- **Format**: `{SYMBOL}.bin`

### `/models/training/` - Training Artifacts
- **Purpose**: Complete training run artifacts
- **Structure**: `{YYYY-MM-DD}/{SYMBOL}/`
- **Contains**: Models, metrics, features, data

### `/hamilton/` - Hamilton Exports
- **Purpose**: Files for live trading
- **Files**: `roster.json`, `champion.json`, configs, active pointers

---

## 📤 What Gets Exported

### Training Run Exports
- `model.bin` - Trained model (128-307 KB)
- `metrics.json` - Performance metrics (~300 bytes)
- `config.json` - Training configuration
- `sha256.txt` - Model hash

### Data Exports
- Candle data (`.parquet` files)
- Feature data
- Market data

### Hamilton Exports
- Roster (ranked coins)
- Champion pointer
- Per-symbol configs
- Active model IDs

---

## 🔧 Configuration

### App Credentials
- **App Key**: `yxnputg7g9kijch`
- **App Secret**: `8llmdzmxj5hw6i8`
- **App Folder**: `Runpodhuracan`

### Required Permissions
- `files.content.write` - Create folders, upload files
- `files.content.read` - Read files
- `files.metadata.read` - Read metadata, list folders

---

## 🛠️ Common Tasks

### Upload Models
```bash
# Models are automatically uploaded after training
# Manual upload:
python scripts/upload_local_candles_to_dropbox.py
```

### Download Data
```bash
# Data is automatically downloaded when needed
# Manual download:
python scripts/simple_download_candles.py --top 250
```

### Verify Upload
```bash
# Check Dropbox folder structure
python scripts/test_dropbox_simple.py
```

---

## 🐛 Troubleshooting

### Token Invalid
- **Fix**: Generate new token from App Console
- **Check**: Token starts with `sl.` and is 1000+ characters

### Permission Denied
- **Fix**: Enable `files.content.write` and `files.content.read` in App Console
- **Check**: Token has correct permissions

### Upload Failed
- **Check**: Internet connection
- **Check**: Token is valid
- **Check**: App folder exists

### Files Not Found
- **Check**: Correct app folder name
- **Check**: File paths are correct
- **Check**: Files were actually uploaded

---

## 📚 Related Files

- Scripts: `scripts/generate_dropbox_token.py`, `scripts/test_dropbox_token.py`
- Code: `src/cloud/training/integrations/dropbox_sync.py`
- Config: `config/base.yaml` (dropbox section)

---

**Dropbox integration is automatic - just set your token!** 🚀

