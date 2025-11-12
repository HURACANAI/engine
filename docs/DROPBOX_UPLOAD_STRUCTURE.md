# Comprehensive Dropbox Upload Structure

## Overview

The training pipeline now automatically uploads **all training data** to Dropbox in a highly organized, categorized structure that makes it easy to extract and analyze data later.

## Upload Structure

All data is organized by **date** and **coin symbol** under the main Huracan folder:

```
/Huracan/
├── models/
│   └── training/
│       └── {YYYY-MM-DD}/              # Training date
│           └── {SYMBOL}/              # e.g., SOL-USDT, BTC-USDT
│               ├── model/
│               │   ├── model.bin      # Trained RL model
│               │   └── model_metadata.json
│               ├── metrics/
│               │   └── training_metrics.json
│               ├── features/
│               │   └── features.json  # Feature metadata
│               ├── data/
│               │   └── {symbol}_candles.parquet
│               └── results/
│                   └── trades.csv     # All trades executed
│
└── data/
    └── candles/
        └── {SYMBOL}/                  # Shared location for candle data
            └── {symbol}_{timestamp}_candles.parquet
```

## What Gets Uploaded

### 1. **Model Files** (`/models/training/{date}/{symbol}/model/`)
- **model.bin**: The trained PyTorch RL agent
- **model_metadata.json**: Metadata about the model (symbol, date, upload timestamp)

### 2. **Training Metrics** (`/models/training/{date}/{symbol}/metrics/`)
- **training_metrics.json**: Complete training performance metrics including:
  - Total trades
  - Win rate
  - Sharpe ratio
  - Total profit (GBP)
  - Mean return (bps)
  - Max drawdown
  - Learning metrics (patterns learned, insights generated)
  - Agent update metrics (policy loss, value loss, entropy)

### 3. **Feature Metadata** (`/models/training/{date}/{symbol}/features/`)
- **features.json**: Feature engineering details:
  - Feature column names
  - Total number of features
  - Data dimensions (rows × columns)
  - Which advanced features are enabled:
    - Higher-order features (interactions, polynomials, ratios, time lags)
    - Granger causality
    - Regime prediction

### 4. **Historical Candle Data** (`/models/training/{date}/{symbol}/data/`)
- **{symbol}_candles.parquet**: Complete OHLCV data used for training
  - Polars DataFrame in Parquet format (compressed, efficient)
  - Includes all candles from lookback period (e.g., 90 days)
  - Also uploaded to shared location: `/data/candles/{SYMBOL}/`

### 5. **Trade History** (`/models/training/{date}/{symbol}/results/`)
- **trades.csv**: All simulated trades from shadow trading:
  - Entry/exit timestamps
  - Entry/exit prices
  - Direction (LONG/SHORT)
  - Position size (GBP)
  - Profit/loss (bps and GBP)
  - Exit reason (TAKE_PROFIT, STOP_LOSS, TIMEOUT, MODEL_SIGNAL)
  - Hold duration (minutes)
  - Win/loss classification
  - Model confidence
  - Market regime

## Benefits of This Structure

### 1. **Easy Data Retrieval**
- Find all data for a specific coin by navigating to its folder
- Compare training runs across different dates
- Download entire training session in one go

### 2. **Time-Based Organization**
- Archive historical training runs by date
- Track model evolution over time
- Compare performance across different time periods

### 3. **Comprehensive Backup**
- Every piece of training data is safely stored
- Can reconstruct entire training session from Dropbox
- Nothing is lost even if local storage fails

### 4. **Analysis-Friendly**
- All data in standard formats (JSON, CSV, Parquet)
- Easy to load in Jupyter notebooks or analysis scripts
- Can build dashboards from uploaded metrics

### 5. **Production-Ready**
- Model files can be downloaded directly for deployment
- Metadata makes it easy to track which model is which
- Trade history enables backtesting validation

## How It Works

The upload happens automatically at the end of each training run:

1. **Training Completes**: Shadow trading finishes, RL agent is trained
2. **Local Save**: All artifacts saved to `/tmp/` temporarily:
   - Model → `rl_agent_{symbol}_{timestamp}.pt`
   - Metrics → `{symbol}_{timestamp}_metrics.json`
   - Features → `{symbol}_{timestamp}_features.json`
   - Trades → `{symbol}_{timestamp}_trades.csv`
   - Candles → `{symbol}_{timestamp}_candles.parquet`

3. **Dropbox Upload**: `export_coin_results()` uploads everything to organized structure
4. **Cleanup**: Temporary files removed after successful upload (model kept locally)

## Example Paths

### SOL/USDT Training on 2025-11-12:
```
/Huracan/models/training/2025-11-12/SOL-USDT/
├── model/
│   ├── model.bin
│   └── model_metadata.json
├── metrics/
│   └── training_metrics.json
├── features/
│   └── features.json
├── data/
│   └── SOL_USDT_20251112_164920_candles.parquet
└── results/
    └── trades.csv
```

### BTC/USDT Training on 2025-11-13:
```
/Huracan/models/training/2025-11-13/BTC-USDT/
├── model/
│   ├── model.bin
│   └── model_metadata.json
├── metrics/
│   └── training_metrics.json
├── features/
│   └── features.json
├── data/
│   └── BTC_USDT_20251113_091545_candles.parquet
└── results/
    └── trades.csv
```

## Retrieving Data

### Download All Data for a Coin
Use the Dropbox API or web interface to download:
```
/Huracan/models/training/2025-11-12/SOL-USDT/
```

### Get Latest Model
Navigate to the most recent date folder for the coin.

### Analyze Training Metrics
Download the `training_metrics.json` file and load it:
```python
import json

with open('training_metrics.json', 'r') as f:
    metrics = json.load(f)

print(f"Win Rate: {metrics['win_rate']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Total Profit: £{metrics['total_profit_gbp']:.2f}")
```

### Load Trade History
```python
import pandas as pd

trades = pd.read_csv('trades.csv')
winners = trades[trades['is_winner'] == True]
print(f"Winning trades: {len(winners)}/{len(trades)}")
```

### Load Candle Data
```python
import polars as pl

candles = pl.read_parquet('SOL_USDT_20251112_164920_candles.parquet')
print(f"Candles: {candles.height} rows × {len(candles.columns)} columns")
```

## Configuration

Dropbox upload requires configuration in your `config.yaml`:

```yaml
dropbox:
  access_token: "YOUR_DROPBOX_ACCESS_TOKEN"
  app_folder: "Huracan"  # Optional, defaults to "Huracan"
```

If Dropbox is not configured, training will still succeed but files will only be saved locally.

## Technical Implementation

The comprehensive upload is implemented in:
- **Pipeline**: [enhanced_rl_pipeline.py:269-418](../src/cloud/training/pipelines/enhanced_rl_pipeline.py)
- **Dropbox Client**: [dropbox_sync.py:625-895](../src/cloud/training/integrations/dropbox_sync.py)

Key method: `DropboxSync.export_coin_results()`
- Takes all artifact paths as parameters
- Uploads each to organized location
- Creates metadata files automatically
- Also uploads candle data to shared location
- Returns success status for each upload

## Monitoring

Watch the training logs for upload status:

```
[2025-11-12T16:58:00Z] [info] model_saved_locally path=/tmp/rl_agent_SOL_USDT_20251112_164920.pt
[2025-11-12T16:58:00Z] [info] metrics_saved_locally path=/tmp/SOL_USDT_20251112_164920_metrics.json
[2025-11-12T16:58:00Z] [info] features_metadata_saved_locally path=/tmp/SOL_USDT_20251112_164920_features.json
[2025-11-12T16:58:00Z] [info] trades_saved_locally path=/tmp/SOL_USDT_20251112_164920_trades.csv count=157
[2025-11-12T16:58:00Z] [info] candle_data_saved_locally path=/tmp/SOL_USDT_20251112_164920_candles.parquet rows=2160
[2025-11-12T16:58:00Z] [info] starting_comprehensive_dropbox_upload symbol=SOL/USDT
[2025-11-12T16:58:05Z] [info] comprehensive_dropbox_upload_complete symbol=SOL/USDT results={'model': True, 'metrics': True, 'features': True, 'candle_data': True, 'trades': True}
[2025-11-12T16:58:05Z] [info] temp_files_cleaned_up
```

## Troubleshooting

### Upload Fails
- Check Dropbox access token is valid
- Verify internet connectivity
- Check Dropbox storage quota
- Training still succeeds, model saved locally

### Missing Files
- Check that training generated trades (empty trades = no model saved)
- Verify all file paths were created successfully
- Check logs for save/upload errors

### Large File Sizes
- Candle data: ~1-5 MB per 90-day training period (compressed Parquet)
- Model: ~5-10 MB
- Metrics/Features: <100 KB
- Trades CSV: <1 MB for typical training run
- **Total per training run: ~10-20 MB**

---

**Last Updated**: 2025-11-12
**Version**: 1.0
