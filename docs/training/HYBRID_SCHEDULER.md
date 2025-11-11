# Hybrid Training Scheduler

## Overview

The hybrid training scheduler enables efficient per-coin model training with three modes: sequential, parallel, and hybrid (batched parallel). It provides safe I/O, partial outputs, resumability, and production-ready error handling.

## Features

### Three Training Modes

1. **Sequential**: Train one coin at a time
2. **Parallel**: Train all coins in parallel (Ray or multiprocessing)
3. **Hybrid** (default): Batched parallel with concurrency cap

### Auto-Detection

- GPU detection: Automatically detects GPU availability
- Defaults: 12 concurrent on GPU nodes, 2 on CPU-only nodes
- Falls back to sequential if max_concurrent=1

### Safe I/O

- Unique work directories per symbol: `models/{SYMBOL}/YYYYMMDD_HHMMSSZ/`
- Partial outputs saved early:
  - `features.parquet`
  - `split_indices.json`
  - `training_log.json`
- Final artifacts:
  - `model.bin`
  - `config.json`
  - `metrics.json`
  - `sha256.txt` (integrity hash)

### Resumability

- Resume ledger: `runs/YYYYMMDDZ/status.json`
- Tracks per-symbol status: pending, running, success, failed, skipped
- Skips completed symbols unless `--force` flag
- Resumes from checkpoints if available

### Timeouts and Retries

- Per-coin timeout: Configurable (default: 45 minutes)
- Retries: Up to 2 retries with jittered backoff
- Error tracking: Error type and stack summary

### Cost Awareness

- Fetches per-symbol costs before training
- Saves to `costs.json`:
  - `taker_fee_bps`
  - `maker_fee_bps`
  - `median_spread_bps`
  - `slippage_bps_per_sigma`
- Passes costs to training for after-cost evaluation

### Storage Integration

- Storage abstraction: Dropbox (now), S3 (future)
- Upload on completion: Not just at the end
- Unique filenames: UUID and timestamp to prevent collisions
- Integrity checks: SHA256 hashes for all files

### Telemetry and Logs

- Structlog JSON logs
- Events: `job_started`, `coin_started`, `coin_partial_saved`, `coin_succeeded`, `coin_failed`, `upload_succeeded`, `upload_failed`, `job_completed`
- Logs to: `logs/YYYYMMDDZ/engine.log` and per-symbol folders

### Metrics and Summary

- Summary JSON: `summary/YYYYMMDDZ/engine_summary.json`
- Fields:
  - `total_symbols`
  - `succeeded`
  - `failed`
  - `skipped`
  - `avg_train_minutes`
  - `median_train_minutes`
  - `total_wall_minutes`
  - `by_symbol` (metrics path and net_bps_after_costs)

### Telegram Integration

- Start notification: Total symbols and mode
- Completion summary: Success/failed/skipped counts and duration
- Failure alerts: Compact alerts with symbol and reason

## Usage

### Basic Usage

```bash
# Hybrid mode (default)
python -m cloud.training.pipelines.daily_retrain_scheduler --mode hybrid --max_concurrent 12 --symbols top20

# Sequential mode
python -m cloud.training.pipelines.daily_retrain_scheduler --mode sequential --symbols top20

# Parallel mode
python -m cloud.training.pipelines.daily_retrain_scheduler --mode parallel --symbols top20
```

### Command Line Flags

- `--mode`: Training mode (`sequential`, `parallel`, `hybrid`) - default: `hybrid`
- `--max_concurrent`: Maximum concurrent workers - default: 12 on GPU, 2 on CPU
- `--symbols`: Symbols to train (`topN`, CSV file, or comma-separated list) - default: `top20`
- `--timeout_minutes`: Timeout per symbol in minutes - default: 45
- `--force`: Force retrain even if already completed
- `--driver`: Storage driver (`dropbox`, `s3`) - default: `dropbox`
- `--dry_run`: Dry run mode (no actual training)

### Environment Variables

- `DROPBOX_ACCESS_TOKEN`: Dropbox access token
- `TELEGRAM_TOKEN`: Telegram bot token
- `TELEGRAM_CHAT_ID`: Telegram chat ID
- `CUDA_VISIBLE_DEVICES`: GPU device IDs (for GPU detection)

### Symbol Selection

```bash
# Top N symbols
--symbols top20

# Comma-separated list
--symbols BTCUSDT,ETHUSDT,SOLUSDT

# CSV file
--symbols symbols.csv
```

## Folder Layout

```
models/
├── BTCUSDT/
│   └── 20250101_020000Z/
│       ├── features.parquet
│       ├── split_indices.json
│       ├── training_log.json
│       ├── model.bin
│       ├── config.json
│       ├── metrics.json
│       ├── costs.json
│       └── sha256.txt
└── ETHUSDT/
    └── 20250101_020100Z/
        └── ...

runs/
└── 20250101Z/
    └── status.json

logs/
└── 20250101Z/
    ├── engine.log
    └── BTCUSDT/
        └── training.log

summary/
└── 20250101Z/
    └── engine_summary.json
```

## Resume Rules

### Idempotency

- If `model.bin` and `metrics.json` exist with `status=ok`, skip retrain unless `--force`
- If only partials exist, resume from last checkpoint if available
- Resume ledger tracks all symbol statuses

### Status Tracking

- `pending`: Symbol not yet started
- `running`: Symbol currently training
- `success`: Symbol completed successfully
- `failed`: Symbol failed (with error details)
- `skipped`: Symbol skipped (with reason)

### Resuming Interrupted Runs

1. Rerun without `--force` flag
2. Completed symbols are skipped
3. Running symbols resume if checkpoint exists
4. Failed symbols are retried (up to 2 times)

## Known Limits

### Concurrency

- GPU nodes: Recommended max_concurrent=12
- CPU nodes: Recommended max_concurrent=2
- Memory: Each worker uses ~2-4GB RAM
- Disk: Each symbol uses ~100MB-1GB disk space

### Timeouts

- Default: 45 minutes per symbol
- Adjust based on data size and model complexity
- Timeout triggers retry (up to 2 retries)

### Storage

- Dropbox: Rate limits apply (600 requests/10 minutes)
- S3: Not yet implemented
- Local: Requires sufficient disk space

## Testing

### Unit Tests

```bash
# Run unit tests
pytest tests/test_scheduler.py
pytest tests/test_resume_ledger.py
pytest tests/test_storage.py
```

### Smoke Test

```bash
# Run smoke test with 3 fake symbols
python -m cloud.training.pipelines.daily_retrain_scheduler --mode hybrid --symbols BTCUSDT,ETHUSDT,SOLUSDT --dry_run
```

## Integration

### With Existing Training Pipeline

```python
from src.cloud.training.pipelines.scheduler import HybridTrainingScheduler, SchedulerConfig, TrainingMode
from src.cloud.training.pipelines.work_item import TrainResult

def train_symbol(symbol: str, cfg: Dict[str, Any]) -> TrainResult:
    # Your training logic here
    return TrainResult(symbol=symbol, status="success", ...)

config = SchedulerConfig(
    mode=TrainingMode.HYBRID,
    max_concurrent=12,
    timeout_minutes=45,
)

scheduler = HybridTrainingScheduler(config=config, train_func=train_symbol)
results = scheduler.schedule_symbols(["BTCUSDT", "ETHUSDT"])
```

## Troubleshooting

### Common Issues

1. **Timeout errors**: Increase `--timeout_minutes` or reduce `--max_concurrent`
2. **Storage upload failures**: Check `DROPBOX_ACCESS_TOKEN` and network connectivity
3. **Memory errors**: Reduce `--max_concurrent` or use sequential mode
4. **Resume not working**: Check `runs/YYYYMMDDZ/status.json` for status

### Debug Mode

```bash
# Enable debug logging
export STRUCTLOG_LEVEL=DEBUG
python -m cloud.training.pipelines.daily_retrain_scheduler --mode hybrid --symbols top20
```

## Next Steps

1. Integrate with actual training pipeline
2. Add S3 storage support
3. Add feature drift checks
4. Add profiler for performance tracking
5. Add unit cost logging

## References

- [Scheduler Implementation](../src/cloud/training/pipelines/scheduler.py)
- [Work Item](../src/cloud/training/pipelines/work_item.py)
- [Resume Ledger](../src/cloud/training/utils/resume_ledger.py)
- [Storage Client](../src/cloud/training/services/storage.py)

