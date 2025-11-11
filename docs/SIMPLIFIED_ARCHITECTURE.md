# Simplified Huracan Architecture

## Overview

Huracan is designed to be simple, clean, and easy to understand. Each module has a single, clear purpose.

## Folder Structure

```
huracan/
├── engine/           # Trains daily models
│   └── README.md
├── mechanic/         # Fine-tunes and promotes
│   └── README.md
├── hamilton/         # Trades live
│   └── README.md
├── archive/          # Stores models + logs
│   └── README.md
├── broadcaster/      # Sends Telegram updates
│   └── README.md
├── shared/           # Shared utils and config
├── config.yaml       # Single config file
├── run_daily.py      # Master script
└── README_SIMPLE.md  # User guide
```

## Module Responsibilities

### Engine
- **Purpose:** Train models nightly from historical data
- **Input:** Candle data from exchanges
- **Output:** Models in `Dropbox/Huracan/models/baselines/DATE/SYMBOL/`
- **Files:** `model.bin`, `metrics.json`, `feature_recipe.json`

### Mechanic
- **Purpose:** Fine-tune and promote models
- **Input:** Champion models from Engine
- **Output:** Updated `champion.json` when promotions occur
- **Files:** `challenger_*.bin`, `promotions.json`

### Hamilton
- **Purpose:** Trade live using best models
- **Input:** Champion models from `champion.json`
- **Output:** Trades logged to `trades/DATE/trades.csv`
- **Files:** `trades.csv`, `equity_curve.json`

### Archive
- **Purpose:** Store all models, metrics, and logs
- **Input:** All outputs from Engine, Mechanic, and Hamilton
- **Output:** Organized storage in `Dropbox/Huracan/`
- **Files:** Daily folders with all artifacts

### Broadcaster
- **Purpose:** Send updates to users
- **Input:** Daily reports and status updates
- **Output:** Telegram messages, Instagram posts
- **Files:** None (just sends messages)

## Configuration

All configuration is in a single `config.yaml` file:

```yaml
general:
  version: "2.0"
  symbols: ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

engine:
  lookback_days: 180
  parallel_tasks: 8

hamilton:
  edge_threshold_bps: 10
  daily_loss_cap_pct: 1.0
```

## Data Flow

1. **Engine** trains models → saves to `models/baselines/DATE/SYMBOL/`
2. **Mechanic** creates challengers → promotes best → updates `champion.json`
3. **Hamilton** loads `champion.json` → trades → logs to `trades/DATE/`
4. **Archive** stores everything → organized by date
5. **Broadcaster** reads reports → sends updates

## File Structure in Dropbox

```
Huracan/
├── models/
│   ├── BTCUSDT/
│   ├── ETHUSDT/
│   └── SOLUSDT/
├── reports/
│   └── 2025-11-11/
│       ├── daily_report.json
│       └── run_results.json
├── trades/
│   └── 2025-11-11.csv
├── logs/
│   └── engine.log
└── champion.json
```

## Key Files

### champion.json
```json
{
  "date": "2025-11-11",
  "models": {
    "BTCUSDT": "models/BTCUSDT/model.bin",
    "ETHUSDT": "models/ETHUSDT/model.bin"
  },
  "summary": {
    "total_symbols": 2,
    "updated_today": ["BTCUSDT"]
  }
}
```

### daily_report.json
```json
{
  "date": "2025-11-11",
  "models_trained": 18,
  "promotions": 3,
  "total_trades": 243,
  "avg_pnl_pct": 1.7,
  "hit_rate_pct": 61
}
```

### engine_status.json
```json
{
  "phase": "training",
  "symbols_done": 5,
  "symbols_failed": 0,
  "symbols_total": 20,
  "current_symbol": "BTCUSDT",
  "eta_seconds": 1200,
  "last_error": null,
  "updated_at": "2025-11-11T02:10:00Z"
}
```

## Logging

Logs are human-readable:

```
[02:00] Engine starting (v2.0)
[02:05] Training BTCUSDT (180 days)...
[02:10] ✅ Model saved. Net PnL +3.2% after fees.
[02:12] Training ETHUSDT...
[02:18] ⚠️ Low volume. Skipped.
[02:20] Engine finished. 1 model saved.
```

## Running the System

### Automated (Recommended)
```bash
python run_daily.py
```

### Manual
```bash
# Run Engine
python -m engine.run

# Run Mechanic
python -m mechanic.run

# Run Hamilton
python -m hamilton.run
```

## Benefits of This Structure

1. **Simple:** One config file, clear module separation
2. **Clean:** Human-readable logs and JSON files
3. **Obvious:** Each module has a single purpose
4. **Maintainable:** Easy to understand and modify
5. **Automated:** One script runs everything

## Next Steps

1. Implement simplified Engine module
2. Implement simplified Mechanic module
3. Implement simplified Hamilton module
4. Implement Archive storage
5. Implement Broadcaster notifications

