# Huracan Simplification Progress

## Completed

### 1. Unified Configuration
- ✅ Created `config.yaml` with all settings in one place
- ✅ Created `config_loader.py` for loading configuration
- ✅ Environment variable support (${VAR_NAME})

### 2. Simplified Structure
- ✅ Created `README_SIMPLE.md` with user-friendly guide
- ✅ Created module READMEs (Engine, Mechanic, Hamilton, Archive, Broadcaster)
- ✅ Created `run_daily.py` master script
- ✅ Created documentation in `SIMPLIFIED_ARCHITECTURE.md`

### 3. Core Utilities
- ✅ Created `daily_report.py` for daily summaries
- ✅ Created `champion_manager.py` for champion.json management
- ✅ Created `status_manager.py` for engine_status.json
- ✅ Created `simple_logger.py` for human-readable logs

### 4. Enhanced Services
- ✅ Created `slippage_calibration.py` for per-symbol slippage calibration
- ✅ Created `data_gates.py` for symbol filtering
- ✅ Created `integrity_verifier.py` for model verification
- ✅ Created `mechanic_service.py` for challenger creation and promotion
- ✅ Created `hamilton_service.py` for model loading and trading

### 5. Per-Coin Contracts
- ✅ Created `per_coin.py` with all contract dataclasses
- ✅ Created `paths.py` for Dropbox path helpers
- ✅ Created `writer.py` for contract writing
- ✅ Created `per_coin_training.py` for per-coin training service

## In Progress

### 1. Engine Integration
- 🔄 Integrate data gates into training pipeline
- 🔄 Integrate slippage calibration into cost model
- 🔄 Add code_hash and data_hash to artifacts
- 🔄 Add profiler for CPU/GPU time and memory
- 🔄 Add unit cost logging per symbol

### 2. Mechanic Enhancements
- 🔄 Add promotion guardrails (min hours, min trades)
- 🔄 Add rollback rules (drawdown floor, win rate floor)
- 🔄 Add staggered work (round robin symbols)
- 🔄 Add shadow A/B testing

### 3. Hamilton Enhancements
- 🔄 Add TCA enforcement (edge > cost + margin)
- 🔄 Add pre-trade checks (balance, notional, step size)
- 🔄 Add session limits (daily loss cap, trade count cap)
- 🔄 Add latency meter (tick to order, order to fill)

### 4. Archive Enhancements
- 🔄 Add trades table/CSV per symbol with TCA breakdown
- 🔄 Add daily equity curve snapshots
- 🔄 Add promotions log table
- 🔄 Add integrity checks on startup

### 5. Observability
- 🔄 Build engine_status.json with phase, symbols, ETA
- 🔄 Add heartbeats for Mechanic and Hamilton
- 🔄 Wire alerts for failed training, uploads, promotions

## To Do

### 1. Data Quality
- ⏳ Add candle repair step (forward fill gaps)
- ⏳ Detect split/symbol change events
- ⏳ Add feature drift checks (PSI/KS)

### 2. Risk and Compliance
- ⏳ Add funding and borrow costs tracking
- ⏳ Add exposure caps by correlated clusters
- ⏳ Add sector grouping (L1 chain, sector)

### 3. Scaling
- ⏳ Add sharding by symbol group
- ⏳ Add backpressure for Dropbox uploads
- ⏳ Add memory caps per symbol

### 4. Governance
- ⏳ Add version rules (bump engine_version on changes)
- ⏳ Add promotion review reports (human-friendly)
- ⏳ Add backward compatibility checks

### 5. Security
- ⏳ Add read-only keys for Hamilton downloads
- ⏳ Add secrets rotation
- ⏳ Add secret version logging (never log values)

### 6. Acceptance Tests
- ⏳ Cost gate test (edge < cost is skipped)
- ⏳ Rollback test (underperformance triggers revert)
- ⏳ Hash test (tampered model fails integrity)
- ⏳ Drift test (synthetic drift triggers flag)
- ⏳ Latency test (high latency pauses trading)

### 7. KPIs
- ⏳ Net PnL per 100 trades after costs
- ⏳ Hit rate and average trade bps by symbol
- ⏳ Max intraday drawdown by account and symbol
- ⏳ Promotion win rate (promotions that outperform)
- ⏳ Cache hit rates and average train time per symbol

## Next Steps

### This Week
1. ✅ Implement slippage calibration per symbol
2. ✅ Add data gates with skip_reasons
3. ✅ Add integrity verifier
4. 🔄 Integrate into training pipeline
5. 🔄 Add promotion guardrails to Mechanic
6. 🔄 Add TCA checks to Hamilton

### Next Week
1. Add feature drift checks
2. Add profiler for performance tracking
3. Add unit cost logging
4. Add rollback rules to Mechanic
5. Add session limits to Hamilton

### Future
1. Add scaling features (sharding, backpressure)
2. Add governance features (versioning, reviews)
3. Add security features (read-only keys, rotation)
4. Add acceptance tests
5. Add KPI tracking

## File Structure

```
huracan/
├── config.yaml                 # ✅ Single config file
├── run_daily.py                # ✅ Master script
├── README_SIMPLE.md            # ✅ User guide
├── engine/
│   ├── README.md               # ✅ Module docs
│   └── run.py                  # ✅ Simplified runner
├── shared/
│   ├── config_loader.py        # ✅ Config loading
│   ├── daily_report.py         # ✅ Daily reports
│   ├── champion_manager.py     # ✅ Champion management
│   ├── status_manager.py       # ✅ Status management
│   └── simple_logger.py        # ✅ Human-readable logs
├── src/
│   ├── shared/contracts/       # ✅ Per-coin contracts
│   └── cloud/training/services/
│       ├── slippage_calibration.py  # ✅ Slippage calibration
│       ├── data_gates.py            # ✅ Data gates
│       ├── integrity_verifier.py    # ✅ Integrity verification
│       ├── mechanic_service.py      # ✅ Mechanic service
│       └── hamilton_service.py      # ✅ Hamilton service
└── docs/
    ├── SIMPLIFIED_ARCHITECTURE.md   # ✅ Architecture docs
    └── SIMPLIFICATION_PROGRESS.md   # ✅ This file
```

## Benefits

1. **Simple:** One config file, clear module separation
2. **Clean:** Human-readable logs and JSON files
3. **Obvious:** Each module has a single purpose
4. **Maintainable:** Easy to understand and modify
5. **Automated:** One script runs everything
6. **Traceable:** All artifacts stored with hashes
7. **Resilient:** Data gates, integrity checks, rollback rules
8. **Observable:** Status files, heartbeats, daily reports

## Usage

```bash
# Run the entire system
python run_daily.py

# Run individual modules
python -m engine.run
python -m mechanic.run
python -m hamilton.run

# Check status
cat engine_status.json

# View daily report
cat reports/2025-11-11/daily_report.json

# View champion models
cat champion.json
```

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

