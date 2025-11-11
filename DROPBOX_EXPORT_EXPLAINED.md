# Dropbox Export Explanation

## 📁 What Was Exported to Dropbox

The test run exported **2 files per coin** (3 coins total = 6 files) to your Dropbox.

### Dropbox Folder Structure

```
/Runpodhuracan/huracan/models/baselines/20251111/
├── BTCUSDT/
│   ├── model.bin          (128 KB) - Trained XGBoost model
│   └── metrics.json       (332 bytes) - Model performance metrics
├── ETHUSDT/
│   ├── model.bin          (241 KB) - Trained XGBoost model
│   └── metrics.json       (296 bytes) - Model performance metrics
└── SOLUSDT/
    ├── model.bin          (307 KB) - Trained XGBoost model
    └── metrics.json       (299 bytes) - Model performance metrics
```

---

## 📄 File Contents Explained

### 1. **model.bin** (Binary Model File)
- **What it is**: The trained XGBoost model, serialized using Python's `pickle` format
- **Size**: 128-307 KB (varies by coin complexity)
- **Contains**:
  - Trained XGBoost model weights and structure
  - Feature names and types
  - Model hyperparameters
  - All information needed to make predictions

**How to use it:**
```python
import pickle

# Load the model
with open('model.bin', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(features)
```

---

### 2. **metrics.json** (Performance Metrics)
- **What it is**: JSON file containing model performance statistics
- **Size**: ~300 bytes
- **Contains**:
  - `symbol`: Trading symbol (e.g., "BTCUSDT")
  - `sample_size`: Number of training samples used
  - `sharpe`: Sharpe ratio (risk-adjusted returns)
  - `hit_rate`: Percentage of correct predictions (0.0-1.0)
  - `net_pnl_pct`: Net profit/loss percentage
  - `gross_pnl_pct`: Gross profit/loss percentage
  - `max_drawdown_pct`: Maximum drawdown percentage
  - `avg_trade_bps`: Average trade profit in basis points
  - `status`: Model status ("ok" or "failed")

**Example metrics.json:**
```json
{
  "symbol": "ETHUSDT",
  "sample_size": 172647,
  "sharpe": 0.0162,
  "hit_rate": 0.4990,
  "net_pnl_pct": 0.0,
  "gross_pnl_pct": 0.0,
  "max_drawdown_pct": 0.0,
  "avg_trade_bps": 0.0,
  "status": "ok"
}
```

---

## 🔍 What You're Looking At in Dropbox

### Path Structure
The files are organized by:
1. **Base folder**: `huracan` (project name)
2. **Type**: `models/baselines` (baseline models)
3. **Date**: `20251111` (November 11, 2025 - training date)
4. **Symbol**: `BTCUSDT`, `ETHUSDT`, `SOLUSDT` (coin symbols)

### Full Dropbox Paths

```
/Runpodhuracan/huracan/models/baselines/20251111/BTCUSDT/model.bin
/Runpodhuracan/huracan/models/baselines/20251111/BTCUSDT/metrics.json
/Runpodhuracan/huracan/models/baselines/20251111/ETHUSDT/model.bin
/Runpodhuracan/huracan/models/baselines/20251111/ETHUSDT/metrics.json
/Runpodhuracan/huracan/models/baselines/20251111/SOLUSDT/model.bin
/Runpodhuracan/huracan/models/baselines/20251111/SOLUSDT/metrics.json
```

---

## 📊 What the Metrics Mean

### Performance Metrics Explained

| Metric | Meaning | Good Value | Our Results |
|--------|---------|------------|-------------|
| **Sharpe Ratio** | Risk-adjusted returns | > 1.0 | ~0.01-0.02 (low, expected for test) |
| **Hit Rate** | % of correct predictions | > 0.55 | ~50% (near random, expected for test) |
| **R² Score** | Model fit quality | > 0.0 | Negative (model worse than baseline) |
| **Sample Size** | Training data points | > 10,000 | ~172K (good) |

**Note**: These are test models with minimal features. Production models would have:
- Better feature engineering
- More sophisticated training
- Cost-aware evaluation
- Regime classification
- Walk-forward validation

---

## 🔧 What Was NOT Exported (But Could Be)

The test script created these files locally but didn't upload them:

### Local Files (Not Uploaded)
- `config.json` - Model configuration (features, hyperparameters, training date)
- `sha256.txt` - Model file hash for integrity verification

### Why They Weren't Uploaded
- The script focused on the essential files: **model** (for predictions) and **metrics** (for evaluation)
- `config.json` is useful for debugging but not required for deployment
- `sha256.txt` is for integrity checks but not needed for basic usage

---

## 🚀 How to Use These Files

### 1. **Load and Use a Model**
```python
import pickle
import json

# Load model
with open('model.bin', 'rb') as f:
    model = pickle.load(f)

# Load metrics
with open('metrics.json', 'r') as f:
    metrics = json.load(f)

# Make predictions
predictions = model.predict(features)
```

### 2. **Check Model Performance**
```python
import json

# Load metrics
with open('metrics.json', 'r') as f:
    metrics = json.load(f)

print(f"Symbol: {metrics['symbol']}")
print(f"Hit Rate: {metrics['hit_rate']:.2%}")
print(f"Sharpe: {metrics['sharpe']:.4f}")
```

### 3. **Compare Models**
```python
# Load metrics for all coins
btc_metrics = json.load(open('BTCUSDT/metrics.json'))
eth_metrics = json.load(open('ETHUSDT/metrics.json'))
sol_metrics = json.load(open('SOLUSDT/metrics.json'))

# Compare performance
print(f"BTC Hit Rate: {btc_metrics['hit_rate']:.2%}")
print(f"ETH Hit Rate: {eth_metrics['hit_rate']:.2%}")
print(f"SOL Hit Rate: {sol_metrics['hit_rate']:.2%}")
```

---

## 📋 Summary

### What You Have in Dropbox:
✅ **6 files total** (2 per coin × 3 coins)
- 3 model files (`.bin`) - Trained XGBoost models
- 3 metrics files (`.json`) - Performance statistics

### What Each File Does:
1. **model.bin** - The actual trained model (use for predictions)
2. **metrics.json** - Model performance stats (use for evaluation)

### Where They Are:
- **Location**: `/Runpodhuracan/huracan/models/baselines/20251111/`
- **Organization**: One folder per coin (BTCUSDT, ETHUSDT, SOLUSDT)
- **Date**: November 11, 2025 (20251111)

### Next Steps:
1. **Download models** from Dropbox to use in production
2. **Review metrics** to understand model performance
3. **Compare models** across different coins
4. **Integrate** into your trading system (Hamilton)

---

## 🎯 Production Pipeline

In a production setup, you would also export:
- `config.json` - Model configuration
- `sha256.txt` - File integrity hash
- `feature_recipe.json` - Feature engineering recipe
- `champion/latest.json` - Champion model pointer
- `manifest.json` - Run manifest with all artifacts

But for this test, we focused on the **essential files**: the model and its metrics.

---

## 📝 Notes

- **Model Format**: XGBoost models are pickled Python objects
- **File Size**: Models are relatively small (128-307 KB) because they're tree-based models
- **Metrics Format**: JSON for easy parsing and human readability
- **Organization**: Files are organized by date and symbol for easy tracking

---

## 🔗 Related Files

- Test Script: `scripts/test_end_to_end_training.py`
- Contract Writer: `src/shared/contracts/writer.py`
- Path Helpers: `src/shared/contracts/paths.py`
- Metrics Schema: `src/shared/contracts/per_coin.py`

