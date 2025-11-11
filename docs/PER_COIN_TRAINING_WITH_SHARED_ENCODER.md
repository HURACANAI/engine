# Per-Coin Training with Shared Encoder

## Overview

This document describes the per-coin training pipeline with shared encoder for cross-coin learning.

## Architecture

### Per-Coin Training
- One tailored model per coin
- Job queue loops symbols and runs train → validate → export
- Saves to Dropbox: `models/{SYMBOL}/baseline_DATE/`
- Files: `model.bin`, `metrics.json`, `costs.json`, `features.json`
- Lightweight champion pointer: `champions/{SYMBOL}.json`

### Shared Encoder
- Trained on all coins to capture common patterns
- Option A: Shared encoder (PCA or autoencoder)
- Option B: Meta learner (maps regime tags to weights)
- Avoids one global model for all coins

### Cross-Coin Learning
- Shared feature encoder trained on union of all features
- Each coin model uses: `[shared_features + coin_specific_features]`
- Per-coin heads (XGBoost or light neural) on top of encoder outputs
- Feature importance tracked in `meta/feature_bank.json`

## Training Pipeline

### 1. Job Queue
- Loops through symbols
- Runs train → validate → export for each symbol
- Parallel processing with configurable workers
- Tracks job status (pending, running, completed, failed, skipped)

### 2. Data Gates
- Checks volume, gaps, spreads, data coverage
- Returns skip_reasons for failed symbols
- Configurable thresholds

### 3. Slippage Calibration
- Fits slippage_bps_per_sigma per symbol from last 30 days
- Stores fit date for tracking
- Uses actual trade fills if available

### 4. Shared Encoder Training
- Trained on all coin features after individual training
- Captures common microstructure patterns
- Frozen for stability (updated weekly)
- Saved to `meta/shared_encoder.pkl`

### 5. Model Training
- Uses shared encoder features + coin-specific features
- XGBoost or light neural head per coin
- Walk-forward validation
- After-cost metrics scoring

### 6. Cost and Liquidity Gates
- Computes after-cost metrics using venue fees, spread, slippage
- Tags symbols as trade_ok only if:
  - net_pnl_pct > 0
  - sample_size > 100
  - sharpe > 0.5
  - hit_rate > 0.45
  - max_drawdown_pct < 20.0

### 7. Artifact Export
- Saves to Dropbox: `models/{SYMBOL}/baseline_DATE/`
- Files: `model.bin`, `metrics.json`, `costs.json`, `features.json`
- Updates champion: `champions/{SYMBOL}.json`
- Updates feature bank: `meta/feature_bank.json`

### 8. Roster Export
- Exports `champions/roster.json` for Hamilton
- Ranked by liquidity, cost, and recent net edge
- Fields: symbol, model_path, rank, spread_bps, fee_bps, avg_slip_bps, last_7d_net_bps, trade_ok
- Hamilton uses this to decide which symbols to trade

## File Structure

```
Dropbox/Huracan/
├── models/
│   ├── BTCUSDT/
│   │   └── baseline_20250101/
│   │       ├── model.bin
│   │       ├── metrics.json
│   │       ├── costs.json
│   │       └── features.json
│   └── ETHUSDT/
│       └── baseline_20250101/
│           └── ...
├── champions/
│   ├── BTCUSDT.json
│   ├── ETHUSDT.json
│   └── roster.json
└── meta/
    ├── shared_encoder.pkl
    └── feature_bank.json
```

## Champion Structure

### Per-Symbol Champion (`champions/{SYMBOL}.json`)
```json
{
  "symbol": "BTCUSDT",
  "date": "20250101",
  "model_path": "models/BTCUSDT/baseline_20250101/model.bin",
  "metrics": {
    "sharpe": 1.5,
    "hit_rate": 0.55,
    "net_pnl_pct": 2.0,
    "max_drawdown_pct": 10.0,
    "sample_size": 1000
  },
  "cost_model": {
    "taker_fee_bps": 4.0,
    "maker_fee_bps": 2.0,
    "median_spread_bps": 5.0,
    "slippage_bps_per_sigma": 2.0
  },
  "feature_recipe_hash": "abc123",
  "updated_at": "2025-01-01T02:00:00Z"
}
```

### Roster (`champions/roster.json`)
```json
{
  "date": "20250101",
  "total_symbols": 400,
  "trade_ok_count": 150,
  "symbols": [
    {
      "symbol": "BTCUSDT",
      "model_path": "models/BTCUSDT/baseline_20250101/model.bin",
      "rank": 1,
      "spread_bps": 5.0,
      "fee_bps": 4.0,
      "avg_slip_bps": 2.0,
      "total_cost_bps": 11.0,
      "last_7d_net_bps": 25.0,
      "trade_ok": true,
      "metrics": {
        "sharpe": 1.5,
        "hit_rate": 0.55,
        "net_pnl_pct": 2.0
      }
    }
  ]
}
```

## Feature Bank (`meta/feature_bank.json`)
```json
{
  "features": {
    "rsi_14": {
      "symbols_using": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
      "avg_importance": 0.15,
      "max_importance": 0.25
    }
  },
  "symbols": {
    "BTCUSDT": {
      "feature_importance": {
        "rsi_14": 0.20,
        "ema_20": 0.15
      },
      "model_type": "xgboost",
      "updated_at": "2025-01-01T02:00:00Z"
    }
  }
}
```

## Usage

### Training Pipeline
```python
from src.cloud.training.pipelines.per_coin_training_pipeline import PerCoinTrainingPipeline

# Initialize pipeline
pipeline = PerCoinTrainingPipeline(config=config, dropbox_sync=dropbox_sync)

# Train all symbols
symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", ...]  # 400 symbols
results = pipeline.train_all_symbols(symbols, max_workers=8)
```

### Shared Encoder
```python
from src.cloud.training.services.shared_encoder import SharedEncoder

# Initialize encoder
encoder = SharedEncoder(encoder_type="pca", n_components=50)

# Train on all coin features
encoder.fit(all_features)

# Transform features for a coin
encoded_features = encoder.transform(coin_features)
```

### Roster Export
```python
from src.cloud.training.services.roster_exporter import RosterExporter

# Initialize exporter
exporter = RosterExporter()

# Export roster
exporter.export_roster(symbols_data)
```

## Hamilton Integration

Hamilton reads `champions/roster.json` and:
1. Filters by `trade_ok=true`
2. Ranks by `rank` (lower is better)
3. Takes top N symbols based on Telegram command `/trade N`
4. Applies user allowlist/blocklist from `runtime/overrides.json`
5. Subscribes only to selected symbol streams

### Telegram Commands
- `/trade 10` - Trade top 10 symbols
- `/trade 20` - Trade top 20 symbols
- `/allow BTCUSDT ETHUSDT` - Add to allowlist
- `/block DOGEUSDT` - Add to blocklist

## Benefits

1. **Per-Coin Models:** One tailored model per coin
2. **Cross-Coin Learning:** Shared encoder captures common patterns
3. **No Coupling:** Per-coin heads keep sensitivity to each book's quirks
4. **Cost-Aware:** After-cost metrics ensure profitable trading
5. **Hamilton Control:** Telegram commands control live trading count
6. **Scalable:** Job queue handles 400 coins efficiently

## Next Steps

1. Implement actual data loading and feature building
2. Implement actual model training (XGBoost/LightGBM)
3. Train shared encoder on all coins
4. Export roster.json for Hamilton
5. Test with 150 symbols first, then scale to 400

