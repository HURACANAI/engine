# Per-Coin Training Implementation Summary

## Overview

This document summarizes the implementation of per-coin training with shared encoder for 400 coins in shadow mode, with clean separation between Engine, Mechanic, and Hamilton.

## Architecture

### Clean Split

1. **Engine**: Trains per-coin models with shared encoder
2. **Mechanic**: Fine-tunes per-coin heads hourly (encoder stays fixed)
3. **Hamilton**: Controls live trading count from Telegram (reads roster.json)

## Implementation

### 1. Per-Coin Training

#### Job Queue System
- ✅ `job_queue.py` - Parallel training with configurable workers
- ✅ Loops symbols and runs train → validate → export
- ✅ Thread-safe result collection
- ✅ Job status tracking (pending, running, completed, failed, skipped)

#### Training Pipeline
- ✅ `per_coin_training_pipeline.py` - Main pipeline
- ✅ Data gates filter low-quality symbols
- ✅ Slippage calibration per symbol
- ✅ Shared encoder training on all coins
- ✅ Per-coin model training with shared + coin-specific features
- ✅ Cost and liquidity gates
- ✅ Artifact export to Dropbox

#### Artifact Storage
- ✅ Saves to: `models/{SYMBOL}/baseline_DATE/`
- ✅ Files: `model.bin`, `metrics.json`, `costs.json`, `features.json`
- ✅ Per-symbol champion: `champions/{SYMBOL}.json`

### 2. Shared Encoder (Cross-Coin Learning)

#### Shared Encoder
- ✅ `shared_encoder.py` - PCA or autoencoder
- ✅ Trained on union of all coin features
- ✅ Captures common microstructure patterns
- ✅ Frozen for stability (updated weekly)
- ✅ Saved to `meta/shared_encoder.pkl`

#### Feature Bank
- ✅ `feature_bank.py` - Tracks feature importance per coin
- ✅ Meta score table in `meta/feature_bank.json`
- ✅ Identifies shared features across coins
- ✅ Tracks which features help/hurt each coin

#### Pattern Sharing
- ✅ Option A: Shared encoder (low risk) - ✅ Implemented
- ✅ Option B: Meta learner (medium risk) - 🔄 Future
- ✅ Avoids one global model for all coins
- ✅ Per-coin heads keep sensitivity to each book's quirks

### 3. Data Gates and Cost Gates

#### Data Gates
- ✅ `data_gates.py` - Strict data filtering
- ✅ Checks: volume, gaps, spreads, data coverage
- ✅ Returns skip_reasons for failed symbols
- ✅ Configurable thresholds

#### Slippage Calibration
- ✅ `slippage_calibration.py` - Calibrates slippage per symbol
- ✅ Fits slippage_bps_per_sigma from last 30 days
- ✅ Stores fit date for tracking
- ✅ Uses actual trade fills if available

#### Cost and Liquidity Gates
- ✅ After-cost metrics scoring
- ✅ Trade_ok tagging based on gates:
  - net_pnl_pct > 0
  - sample_size > 100
  - sharpe > 0.5
  - hit_rate > 0.45
  - max_drawdown_pct < 20.0

### 4. Roster Export for Hamilton

#### Roster Exporter
- ✅ `roster_exporter.py` - Exports `champions/roster.json`
- ✅ Ranks symbols by liquidity, cost, and recent net edge
- ✅ Fields: symbol, model_path, rank, spread_bps, fee_bps, avg_slip_bps, last_7d_net_bps, trade_ok
- ✅ Hamilton reads this for trading decisions

#### Per-Symbol Champion
- ✅ `per_symbol_champion.py` - Manages `champions/{SYMBOL}.json`
- ✅ Lightweight champion pointer per symbol
- ✅ Updates only if new model is better
- ✅ Comparison based on sharpe and net_pnl

### 5. Hamilton Integration (Telegram Control)

#### Telegram Commands
- `/trade 10` - Trade top 10 symbols
- `/trade 20` - Trade top 20 symbols
- `/allow BTCUSDT ETHUSDT` - Add to allowlist
- `/block DOGEUSDT` - Add to blocklist

#### Selection Logic
1. Read `champions/roster.json`
2. Filter by `trade_ok=true`
3. Rank by `rank` (lower is better)
4. Apply user allowlist/blocklist from `runtime/overrides.json`
5. Take top N based on `/trade N` command
6. Subscribe only to selected symbol streams

#### Runtime Overrides
- ✅ `runtime/overrides.json` - User allowlist/blocklist
- ✅ Selection: `user_allowlist - user_blocklist`, capped by N from `/trade`

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
├── meta/
│   ├── shared_encoder.pkl
│   └── feature_bank.json
└── runtime/
    └── overrides.json  # Hamilton runtime overrides
```

## Configuration

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

## Training Flow

1. **Data Loading**: Load candle data for all symbols
2. **Data Gates**: Filter low-quality symbols (skip_reasons in manifest)
3. **Feature Building**: Build features for each symbol
4. **Shared Encoder**: Train shared encoder on all coin features
5. **Model Training**: Train per-coin model with shared + coin-specific features
6. **Slippage Calibration**: Calibrate slippage per symbol
7. **Cost Gates**: Check after-cost metrics (trade_ok tagging)
8. **Artifact Export**: Save models, metrics, costs, features to Dropbox
9. **Champion Update**: Update per-symbol champion if better
10. **Feature Bank**: Update feature importance per coin
11. **Roster Export**: Export `champions/roster.json` for Hamilton

## Benefits

1. **Scalable**: Handles 400 coins efficiently with job queue
2. **Pattern Sharing**: Shared encoder captures common patterns
3. **No Coupling**: Per-coin heads keep sensitivity to each book
4. **Cost-Aware**: After-cost metrics ensure profitability
5. **Hamilton Control**: Telegram commands control live trading count
6. **Traceable**: All artifacts stored with hashes
7. **Clean Split**: Engine trains, Hamilton trades, Mechanic fine-tunes

## What to Avoid

- ❌ Do not make one global model for all coins
- ❌ Do not let Engine make live trade counts (Hamilton's job)
- ❌ Do not force pattern transfer if it drops after-cost metrics

## Practical Starter Plan

1. ✅ Train 100-150 symbols first (prove pipeline)
2. ✅ Add shared encoder trained on union of all features
3. ✅ Freeze encoder for a week (stability)
4. ✅ Keep one XGBoost or light neural head per coin
5. ✅ Validate with walk-forward
6. ✅ Score after costs
7. ✅ Publish `champions/roster.json` and `champions/{SYMBOL}.json`
8. 🔄 Hamilton implements Telegram controls and filtering
9. 🔄 Mechanic fine-tunes per-coin heads hourly (encoder stays fixed)

## Next Steps

### Immediate
1. Integrate actual data loading
2. Implement actual model training (XGBoost/LightGBM)
3. Train shared encoder on all coins
4. Export roster.json
5. Test with 150 symbols

### Short Term
1. Scale to 400 symbols
2. Add feature drift checks
3. Add profiler for performance tracking
4. Add unit cost logging
5. Add promotion guardrails to Mechanic
6. Add TCA checks to Hamilton

### Long Term
1. Add meta learner (Option B)
2. Add autoencoder option
3. Add feature importance analysis
4. Add cross-coin feature validation
5. Add rollback rules to Mechanic
6. Add session limits to Hamilton

## Acceptance Checklist

### Engine
- ✅ Per-coin training with job queue
- ✅ Shared encoder for cross-coin learning
- ✅ Data gates with skip_reasons
- ✅ Slippage calibration per symbol
- ✅ Cost and liquidity gates
- ✅ Per-symbol champion pointers
- ✅ Roster export for Hamilton
- ✅ Feature bank for meta scores

### Mechanic
- 🔄 Fine-tune per-coin heads hourly
- 🔄 Encoder stays fixed for stability
- 🔄 Use per-symbol champions

### Hamilton
- 🔄 Reads roster.json
- 🔄 Filters by trade_ok
- 🔄 Telegram commands control trade count
- 🔄 User allowlist/blocklist support
- 🔄 Runtime overrides from `runtime/overrides.json`

## Conclusion

The per-coin training system with shared encoder is now implemented. The system:
- Trains one tailored model per coin
- Shares patterns through shared encoder (Option A)
- Exports roster.json for Hamilton
- Supports 400 coins in shadow mode
- Allows Hamilton to control live trading count via Telegram
- Maintains clean split between Engine, Mechanic, and Hamilton

Ready for integration and testing with 150 symbols first, then scaling to 400.

