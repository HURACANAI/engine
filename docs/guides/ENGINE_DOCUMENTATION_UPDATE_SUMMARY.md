# Engine Documentation Update Summary

## Overview

All engine documentation has been updated to reflect the new **hybrid training scheduler** and **per-coin training architecture**.

## Updated Files

### 1. `HOW_THE_BOT_WORKS_SIMPLE.md`
- Updated STEP 6: Learn and Test section
- Added details about hybrid training scheduler
- Added per-coin training flow
- Updated STEP 7: Save Everything section
- Added per-coin artifacts structure
- Updated timing information
- Updated summary section

### 2. `HOW_THE_BOT_WORKS_DETAILED.md`
- Updated STEP 6: Learn and Test section
- Added comprehensive hybrid training scheduler details
- Added per-coin training loop
- Added failure handling and retries
- Added summary and roster generation
- Updated file structure section
- Updated Dropbox save structure

### 3. `HOW_IT_WORKS_SIMPLE.md`
- Updated "How Does It Learn?" section
- Added hybrid training scheduler details
- Added per-coin training steps
- Updated daily routine section
- Updated training phase section

### 4. `ENGINE_TRAINING_UPDATED.md` (New)
- Created new document explaining updated training architecture
- Includes key changes, training flow, file structure, usage, benefits
- References to other documentation

## Key Updates

### Hybrid Training Scheduler
- Three modes: sequential, parallel, hybrid (default)
- Auto-detection of GPU availability
- Defaults: 12 concurrent on GPU, 2 on CPU
- Timeout protection (default: 45 minutes per coin)
- Retries (up to 2 times with backoff)

### Per-Coin Training
- One model per coin
- Shared encoder for cross-coin learning
- Per-coin heads (XGBoost/LightGBM)
- Cost-aware evaluation
- After-cost metrics

### Resumability
- Resume ledger tracks training status
- Can resume from where it left off
- Skips completed symbols (unless `--force`)
- Resumes from checkpoints if available

### Storage Integration
- Storage client abstraction (Dropbox now, S3 future)
- Uploads artifacts as training completes
- Integrity checks (SHA256 hashes)

### File Structure
- Per-coin artifacts: `models/{SYMBOL}/YYYYMMDD_HHMMSSZ/`
- Champion pointers: `champions/{SYMBOL}.json`
- Roster file: `champions/roster.json`
- Resume ledger: `runs/YYYYMMDDZ/status.json`
- Summary: `summary/YYYYMMDDZ/engine_summary.json`

## Documentation Consistency

All documentation now consistently describes:
1. Hybrid training scheduler with three modes
2. Per-coin training with shared encoder
3. Cost-aware evaluation
4. Resumability and resume ledger
5. Storage integration (Dropbox/S3)
6. Telegram integration
7. Summary and roster generation

## Next Steps

1. Review documentation for accuracy
2. Add examples and use cases
3. Add troubleshooting sections
4. Add performance benchmarks
5. Add best practices

## References

- [Hybrid Scheduler Documentation](../training/HYBRID_SCHEDULER.md)
- [Per-Coin Training Documentation](../training/PER_COIN_TRAINING_WITH_SHARED_ENCODER.md)
- [Implementation Summary](../training/HYBRID_SCHEDULER_IMPLEMENTATION.md)
- [Engine Training Updated](../guides/ENGINE_TRAINING_UPDATED.md)

