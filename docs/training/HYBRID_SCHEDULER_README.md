# Hybrid Training Scheduler - Quick Start

## Overview

The hybrid training scheduler enables efficient per-coin model training with three modes, safe I/O, resumability, and production-ready error handling.

## Quick Start

### Installation

```bash
# Install dependencies
pip install dropbox structlog

# Set environment variables
export DROPBOX_ACCESS_TOKEN=your_token
export TELEGRAM_TOKEN=your_token  # Optional
export TELEGRAM_CHAT_ID=your_chat_id  # Optional
```

### Basic Usage

```bash
# Run with default settings (hybrid mode, top 20 symbols)
python -m cloud.training.pipelines.daily_retrain_scheduler --symbols top20

# Run in sequential mode
python -m cloud.training.pipelines.daily_retrain_scheduler --mode sequential --symbols top20

# Run in parallel mode
python -m cloud.training.pipelines.daily_retrain_scheduler --mode parallel --symbols top20

# Run with custom concurrency
python -m cloud.training.pipelines.daily_retrain_scheduler --mode hybrid --max_concurrent 8 --symbols top20
```

### Command Line Options

- `--mode`: Training mode (`sequential`, `parallel`, `hybrid`) - default: `hybrid`
- `--max_concurrent`: Maximum concurrent workers - default: 12 on GPU, 2 on CPU
- `--symbols`: Symbols to train (`topN`, CSV file, or comma-separated list) - default: `top20`
- `--timeout_minutes`: Timeout per symbol in minutes - default: 45
- `--force`: Force retrain even if already completed
- `--driver`: Storage driver (`dropbox`, `s3`) - default: `dropbox`
- `--dry_run`: Dry run mode (no actual training)

### Examples

```bash
# Train top 20 symbols
python -m cloud.training.pipelines.daily_retrain_scheduler --symbols top20

# Train specific symbols
python -m cloud.training.pipelines.daily_retrain_scheduler --symbols BTCUSDT,ETHUSDT,SOLUSDT

# Train from CSV file
python -m cloud.training.pipelines.daily_retrain_scheduler --symbols symbols.csv

# Force retrain
python -m cloud.training.pipelines.daily_retrain_scheduler --symbols top20 --force

# Dry run
python -m cloud.training.pipelines.daily_retrain_scheduler --symbols top20 --dry_run

# Custom timeout
python -m cloud.training.pipelines.daily_retrain_scheduler --symbols top20 --timeout_minutes 60
```

## Output Structure

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

## Resuming Interrupted Runs

The scheduler automatically resumes interrupted runs:

1. Rerun without `--force` flag
2. Completed symbols are skipped
3. Running symbols resume if checkpoint exists
4. Failed symbols are retried (up to 2 times)

```bash
# Resume interrupted run
python -m cloud.training.pipelines.daily_retrain_scheduler --symbols top20
```

## Monitoring

### Logs

Logs are written to:
- `logs/YYYYMMDDZ/engine.log` - Main engine log
- `logs/YYYYMMDDZ/{SYMBOL}/training.log` - Per-symbol logs

### Summary

Summary is written to:
- `summary/YYYYMMDDZ/engine_summary.json`

### Telegram

If `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID` are set:
- Start notification when training starts
- Completion summary when training completes
- Failure alerts when symbols fail

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
python -m cloud.training.pipelines.daily_retrain_scheduler --symbols top20
```

## Next Steps

1. Integrate with actual training pipeline
2. Add S3 storage support
3. Add unit tests
4. Add smoke test

## References

- [Full Documentation](./HYBRID_SCHEDULER.md)
- [Implementation Summary](./HYBRID_SCHEDULER_IMPLEMENTATION.md)

