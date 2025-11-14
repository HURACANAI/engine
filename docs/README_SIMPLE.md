# Huracan System (Simple Version)

## Overview

Huracan is a complete trading system with five main components:

1. **Engine** - Trains models nightly using hybrid scheduler (per-coin training with shared encoder)
2. **Mechanic** - Improves models hourly (fine-tunes and promotes with strict rules)
3. **Hamilton** - Trades live using the best model (reads champion pointers and roster)
4. **Archive** - Stores everything (S3 with models, champions, summaries, live logs)
5. **Broadcaster** - Updates you on Telegram (start, progress, completion, failures)

## Key Features

- **One Engine Interface**: All 23 engines use the same interface
- **Shared Feature Builder**: Same recipe for cloud and Hamilton
- **Costs in the Loop**: Fees, spread, slippage per symbol, per bar
- **Regime Gate**: Only allow engines in regimes where they work
- **Meta Combiner**: Per-coin EMA weights by recent accuracy and net edge
- **Per-Coin Champion**: Latest.json per symbol, always valid
- **Hybrid Scheduler**: Batched parallel training (8-16 coins per GPU)
- **S3 Storage**: Models, champions, summaries, live logs
- **Database**: Models, metrics, promotions, live trades, daily equity
- **Telegram Control**: `/trade 10`, `/trade 20` commands

## Quick Start

```bash
# Run training with hybrid scheduler
python -m cloud.training.pipelines.daily_retrain_scheduler --mode hybrid --max_concurrent 12 --symbols top20

# Or use the master script
python run_daily.py
```

Results will appear in `s3://huracan/` (or local `models/`, `champion/`, `summaries/` directories)

## How It Works

### Engine
- Runs daily at 02:00 UTC
- Uses hybrid scheduler (sequential, parallel, or hybrid mode)
- Trains one model per symbol with shared encoder
- Saves models to `s3://huracan/models/SYMBOL/TIMESTAMP/`
- Outputs: `model.bin`, `config.json`, `metrics.json`, `costs.json`, `sha256.txt`
- Updates champion pointers: `s3://huracan/champion/SYMBOL/latest.json`
- Generates daily summary: `s3://huracan/summaries/daily/DATE.json`

### Mechanic
- Runs hourly
- Creates challengers from recent models
- Promotes better models to champion
- Updates `champion.json` when promotions occur

### Hamilton
- Runs continuously
- Loads champion models from `s3://huracan/champion/SYMBOL/latest.json`
- Reads symbols selector file (from Telegram commands)
- Makes trading decisions based on meta combiner output
- Applies costs and guardrails (net edge floor, spread threshold)
- Logs all trades to `s3://huracan/live_logs/trades/YYYYMMDD.parquet`

### Archive
- Stores all models, metrics, and trade logs
- One folder per day: `reports/YYYY-MM-DD/`
- Keeps promotion history in `promotions.json`

### Broadcaster
- Sends daily summaries to Telegram
- Notifies on promotions, errors, and milestones

## Configuration

Edit `config.yaml` to customize:

```yaml
general:
  symbols: ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

engine:
  lookback_days: 180
  parallel_tasks: 8

hamilton:
  edge_threshold_bps: 10
  daily_loss_cap_pct: 1.0
```

## Folder Structure

```
huracan/
├── engine/           # Trains daily models
├── mechanic/         # Fine-tunes and promotes
├── hamilton/         # Trades live
├── archive/          # Stores models + logs
├── broadcaster/      # Sends Telegram updates
├── shared/           # Shared utils and config
├── config.yaml       # Single config file
└── run_daily.py      # Master script
```

## Output Structure

```
s3://huracan/
├── models/
│   ├── BTCUSDT/
│   │   └── 20250101_020000Z/
│   │       ├── model.bin
│   │       ├── config.json
│   │       ├── metrics.json
│   │       ├── costs.json
│   │       └── sha256.txt
│   └── ETHUSDT/
│       └── ...
├── challengers/
│   ├── BTCUSDT/
│   │   └── 20250101_030000Z/
│   │       └── ...
│   └── ETHUSDT/
│       └── ...
├── champion/
│   ├── BTCUSDT/
│   │   └── latest.json
│   └── ETHUSDT/
│       └── latest.json
├── summaries/
│   └── daily/
│       └── 2025-01-01.json
└── live_logs/
    └── trades/
        └── 20250101.parquet
```

## Daily Report

Each day, a `daily_report.json` is generated:

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

## Logs

Logs are human-readable:

```
[02:00] Engine starting (v2.0)
[02:05] Training BTCUSDT (180 days)...
[02:10] ✅ Model saved. Net PnL +3.2% after fees.
[02:12] Training ETHUSDT...
[02:18] ⚠️ Low volume. Skipped.
[02:20] Engine finished. 1 model saved.
```

## Troubleshooting

- Check `logs/engine.log` for errors
- Verify Dropbox connection in `config.yaml`
- Check `engine_status.json` for current status
- Review `daily_report.json` for daily summary

## Support

For issues, check:
1. Logs in `logs/`
2. Status in `engine_status.json`
3. Daily report in `reports/YYYY-MM-DD/daily_report.json`

