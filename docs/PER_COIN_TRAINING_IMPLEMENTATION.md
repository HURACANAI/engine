# Per-Coin Training Implementation

## Overview

This document describes the implementation of per-coin training contracts and infrastructure for the Engine. The implementation follows the requirements to train one model per coin, export per-coin artifacts, and route live trading by symbol.

## Completed Work

### 1. Contract Helpers (`src/shared/contracts/per_coin.py`)

Created dataclasses for all per-coin contracts:

- **RunManifest**: Manifest per run with per-coin artifacts and metrics
- **ChampionPointer**: Champion pointer per coin with model paths
- **FeatureRecipe**: Feature recipe per coin with configuration and hash
- **PerCoinMetrics**: Metrics per coin with cost-aware performance metrics
- **CostModel**: Cost model per coin with trading fees and execution costs
- **Heartbeat**: Engine heartbeat with current status and progress
- **FailureReport**: Failure report with error details and suggestions

All contracts include:
- JSON serialization/deserialization
- Type validation
- Hash computation for reproducibility

### 2. Path Helpers (`src/shared/contracts/paths.py`)

Created utility functions for Dropbox path generation:

- `get_manifest_path()`: Path to manifest.json
- `get_champion_pointer_path()`: Path to latest.json
- `get_model_path()`: Path to model.bin per symbol
- `get_config_path()`: Path to config.json per symbol
- `get_metrics_path()`: Path to metrics.json per symbol
- `get_feature_recipe_path()`: Path to feature_recipe.json per symbol
- `get_heartbeat_path()`: Path to engine.json
- `get_failure_report_path()`: Path to failure_report.json
- `format_date_str()`: Format date as YYYYMMDD
- `format_date_iso()`: Format date as YYYY-MM-DD

### 3. Contract Writer (`src/shared/contracts/writer.py`)

Created `ContractWriter` class for writing contracts to Dropbox:

- `write_manifest()`: Write run manifest
- `write_champion_pointer()`: Write champion pointer
- `write_feature_recipe()`: Write feature recipe
- `write_metrics()`: Write metrics
- `write_cost_model()`: Write cost model
- `write_heartbeat()`: Write heartbeat
- `write_failure_report()`: Write failure report
- `write_model_file()`: Write model binary file

All methods:
- Handle temporary file creation
- Upload to Dropbox via DropboxSync
- Clean up temporary files
- Return Dropbox path on success

### 4. Per-Coin Training Service (`src/cloud/training/services/per_coin_training.py`)

Created `PerCoinTrainingService` class that:

- Creates contract instances from training results
- Converts `TrainingTaskResult` to `PerCoinMetrics`
- Converts `CostBreakdown` to `CostModel`
- Exports training results as per-coin artifacts
- Writes heartbeats and failure reports

Key methods:
- `create_run_manifest()`: Create run manifest from training results
- `create_champion_pointer()`: Create champion pointer
- `create_feature_recipe()`: Create feature recipe
- `create_per_coin_metrics()`: Create per-coin metrics
- `create_cost_model()`: Create cost model
- `export_training_result()`: Export training result as per-coin artifacts

### 5. Configuration (`config/base.yaml` and `src/cloud/training/config/settings.py`)

Added per-coin training configuration:

```yaml
training:
  per_coin:
    symbols_allowed: []  # Empty list = all symbols from universe
    per_symbol_costs: {}  # Per-symbol cost overrides
    parallel_tasks: 2  # Number of symbols to train in parallel
    time_budget_per_symbol_minutes: 30  # Time budget per symbol
    gates:
      min_sharpe: 1.0
      min_hit_rate: 0.50
      max_drawdown_pct: 15.0
      min_net_pnl_pct: 1.0
      min_sample_size: 100
    promotion_rules:
      min_hit_rate_improvement: 0.01
      min_sharpe_improvement: 0.2
      max_drawdown_tolerance: 0.0
      min_net_pnl_improvement: 0.01
```

Added settings classes:
- `PerCoinGatesSettings`: Gate thresholds for model promotion
- `PerCoinPromotionRulesSettings`: Promotion rules for challenger promotion
- `PerCoinTrainingSettings`: Per-coin training configuration

## Directory Structure

The implementation follows the required Dropbox directory structure:

```
huracan/
├── YYYYMMDD/
│   ├── manifest.json
│   └── logs/
│       └── failure_report.json
├── models/
│   └── baselines/
│       └── YYYYMMDD/
│           └── SYMBOL/
│               ├── model.bin
│               ├── config.json
│               ├── metrics.json
│               ├── costs.json
│               └── feature_recipe.json
├── champion/
│   └── latest.json
├── heartbeats/
│   └── engine.json
└── reports/
    └── daily/
        └── YYYYMMDD/
            └── summary.json
```

## Next Steps

### 1. Wire Engine Outputs

Integrate `PerCoinTrainingService` into the training pipeline:

- Modify `TrainingOrchestrator.run()` to use `PerCoinTrainingService`
- Export per-coin artifacts after each symbol is trained
- Write manifest.json after all symbols are trained
- Write champion pointer if gates pass

### 2. Add Per-Symbol Training Loop

Modify the training loop to:

- Train one model per symbol
- Respect memory caps
- Write partial outputs per symbol early
- Handle failures gracefully

### 3. Implement Cost-Aware Scoring

Add cost-aware scoring:

- Fetch per-symbol costs from configuration or exchange
- Score net after costs
- Gate baselines on net metrics
- Use cost model for each symbol

### 4. Add Champion Pointer Writer

Update champion pointer:

- Read existing champion pointer
- Compare new models with champions
- Update champion pointer only if gates pass
- Append promotions log

### 5. Add Heartbeat Updates

Implement heartbeat updates:

- Update heartbeat every 30-60 seconds
- Include phase, current_symbol, progress, last_error
- Write to Dropbox via ContractWriter

### 6. Add Failure Bundle Writer

Implement failure reporting:

- Write failure_report.json on any exception
- Include step, exception_type, message, last_files_written, suggestions
- Store in YYYYMMDD/logs/failure_report.json

### 7. Add Unit Tests

Create unit tests for:

- Contract helpers (per_coin.py)
- Path helpers (paths.py)
- Contract writer (writer.py)
- Per-coin training service (per_coin_training.py)

## Usage Example

```python
from src.shared.contracts.writer import ContractWriter
from src.cloud.training.services.per_coin_training import PerCoinTrainingService
from src.cloud.training.integrations.dropbox_sync import DropboxSync

# Initialize Dropbox sync
dropbox_sync = DropboxSync(
    access_token="your_token",
    app_folder="huracan",
)

# Initialize per-coin training service
per_coin_service = PerCoinTrainingService(
    dropbox_sync=dropbox_sync,
    base_folder="huracan",
    engine_version="1.0.0",
)

# Export training result
exported_paths = per_coin_service.export_training_result(
    symbol="BTCUSDT",
    result=training_result,
    date_str="20250101",
    feature_recipe=feature_recipe,
    sample_size=1000,
)

# Write heartbeat
per_coin_service.write_heartbeat(
    phase="training",
    current_symbol="BTCUSDT",
    progress=0.5,
)

# Create and write run manifest
manifest = per_coin_service.create_run_manifest(
    run_id="run_20250101_020000",
    utc_started=datetime.now(timezone.utc),
    symbols_trained=["BTCUSDT", "ETHUSDT"],
    artifacts_map={"BTCUSDT": "/huracan/models/baselines/20250101/BTCUSDT/model.bin"},
    metrics_map={"BTCUSDT": {...}},
    costs_map={"BTCUSDT": {...}},
    feature_recipe_hash_map={"BTCUSDT": "abc123"},
)

per_coin_service.contract_writer.write_manifest(manifest, "20250101")
```

## Testing

To test the implementation:

1. Create a test Dropbox folder
2. Initialize DropboxSync with test token
3. Create test training results
4. Export artifacts using PerCoinTrainingService
5. Verify files are written to Dropbox
6. Verify contract structure matches requirements

## Notes

- All contracts are Dropbox-compatible now
- S3 and Postgres ready for future migration
- Contracts follow the required structure exactly
- Path helpers generate correct Dropbox paths
- Configuration supports per-coin training settings

## References

- Contract specifications in user requirements
- Dropbox directory structure requirements
- Promotion rules specifications
- Cost model requirements

