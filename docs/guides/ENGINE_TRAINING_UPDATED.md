# Engine Training - Updated Architecture

## Overview

The Engine training system has been updated to use a **hybrid training scheduler** that enables efficient per-coin model training with resumability, cost awareness, and production-ready error handling.

## Key Changes

### 1. Hybrid Training Scheduler

The Engine now uses a hybrid training scheduler with three modes:

- **Sequential mode**: Train one coin at a time (safe, simple)
- **Parallel mode**: Train all coins at once (fast, requires more resources)
- **Hybrid mode** (default): Train coins in batches with concurrency cap (balanced)

### 2. Per-Coin Training

- One model per coin (tailored to each coin's characteristics)
- Shared encoder for cross-coin learning (captures common patterns)
- Per-coin heads (XGBoost/LightGBM) on top of encoder outputs
- Cost-aware evaluation (after-cost metrics)

### 3. Resumability

- Resume ledger tracks training status per symbol
- Can resume from where it left off if interrupted
- Skips completed symbols (unless `--force` flag)
- Resumes from checkpoints if available

### 4. Cost Awareness

- Fetches per-symbol costs before training
- Evaluates models after costs (fees, spread, slippage)
- Only saves models that pass after-cost quality gates

### 5. Storage Integration

- Storage client abstraction (Dropbox now, S3 future)
- Uploads artifacts as training completes
- Integrity checks (SHA256 hashes)

### 6. Telegram Integration

- Start notification when training starts
- Completion summary when training completes
- Failure alerts when symbols fail

## Training Flow

1. **Initialize Scheduler**
   - Select training mode (sequential, parallel, hybrid)
   - Set up resume ledger
   - Configure timeout and retries
   - Initialize storage client

2. **Load Symbols**
   - Read symbol list from config or command line
   - Filter completed symbols (unless `--force`)

3. **Train Each Coin**
   - Fetch costs before training
   - Create unique work directory
   - Load data and build features
   - Train shared encoder (first run)
   - Train per-coin model
   - Evaluate model (after-cost metrics)
   - Save final artifacts
   - Upload to storage
   - Update resume ledger

4. **Handle Failures**
   - Timeout protection (default: 45 minutes)
   - Retries (up to 2 times with backoff)
   - Error handling and logging

5. **Generate Summary**
   - Create summary JSON
   - Generate roster file
   - Update champion pointers

## File Structure

```
/Huracan/
├── models/
│   ├── BTCUSDT/
│   │   └── 20250101_020000Z/
│   │       ├── model.bin
│   │       ├── config.json
│   │       ├── metrics.json
│   │       ├── costs.json
│   │       ├── features.json
│   │       ├── sha256.txt
│   │       ├── features.parquet
│   │       ├── split_indices.json
│   │       └── training_log.json
│   └── ETHUSDT/
│       └── ...
├── champions/
│   ├── BTCUSDT.json
│   ├── ETHUSDT.json
│   └── roster.json
├── runs/
│   └── 20250101Z/
│       └── status.json
└── summary/
    └── 20250101Z/
        └── engine_summary.json
```

## Usage

### Command Line

```bash
# Hybrid mode (default)
python -m cloud.training.pipelines.daily_retrain_scheduler --mode hybrid --max_concurrent 12 --symbols top20

# Sequential mode
python -m cloud.training.pipelines.daily_retrain_scheduler --mode sequential --symbols top20

# Parallel mode
python -m cloud.training.pipelines.daily_retrain_scheduler --mode parallel --symbols top20
```

### Configuration

```yaml
engine:
  target_symbols: 400  # Train on 400 coins
  start_with_symbols: 150  # Start with 150 symbols
  parallel_tasks: 8
  shared_encoder:
    type: "pca"
    n_components: 50
    enabled: true
```

## Benefits

1. **Scalable**: Handles 400 coins efficiently with job queue
2. **Resumable**: Can resume from where it left off if interrupted
3. **Cost-Aware**: Evaluates models after costs (fees, spread, slippage)
4. **Pattern Sharing**: Shared encoder captures common patterns
5. **No Coupling**: Per-coin heads keep sensitivity to each book
6. **Production-Ready**: Timeout protection, retries, error handling
7. **Observable**: Comprehensive logging and Telegram integration

## Next Steps

1. Integrate with actual training pipeline
2. Add S3 storage support
3. Add unit tests
4. Add smoke test

## References

- [Hybrid Scheduler Documentation](../training/HYBRID_SCHEDULER.md)
- [Per-Coin Training Documentation](../training/PER_COIN_TRAINING_WITH_SHARED_ENCODER.md)
- [Implementation Summary](../training/HYBRID_SCHEDULER_IMPLEMENTATION.md)

