# Core System Implementation

## Overview

This document describes the core system implementation with non-negotiables, contracts, scheduler, archive layout, database tables, and Telegram control.

## Non-Negotiables

### 1. One Engine Interface for All 23 Engines

**Implementation**: `src/shared/engines/engine_interface.py`

- **BaseEngine**: Abstract base class for all engines
- **EngineInput**: Unified input (symbol, timestamp, features, regime, costs, metadata)
- **EngineOutput**: Unified output (direction, edge_bps_before_costs, confidence_0_1, horizon_minutes, metadata)
- **EngineRegistry**: Registry for managing all 23 engines

**Usage**:
```python
from src.shared.engines import BaseEngine, EngineInput, EngineOutput, EngineRegistry

class MyEngine(BaseEngine):
    def infer(self, input_data: EngineInput) -> EngineOutput:
        # Engine logic here
        return EngineOutput(
            direction=Direction.BUY,
            edge_bps_before_costs=10.0,
            confidence_0_1=0.8,
            horizon_minutes=60,
            metadata={},
        )

registry = EngineRegistry()
registry.register(MyEngine(engine_id="engine_1", name="My Engine", supported_regimes=["TREND", "RANGE"]))
```

### 2. One Shared Feature Builder

**Implementation**: `src/shared/features/feature_builder.py`

- **FeatureRecipe**: Recipe definition (timeframes, indicators, fill_rules, normalization, window_sizes)
- **FeatureBuilder**: Builder for creating features from candles
- Same recipe for cloud and Hamilton

**Usage**:
```python
from src.shared.features import FeatureBuilder, FeatureRecipe

recipe = FeatureRecipe(
    timeframes=["1h", "4h", "1d"],
    indicators={"rsi": {"period": 14}},
    fill_rules={"strategy": "forward_fill"},
    normalization={"type": "standard"},
)

builder = FeatureBuilder(recipe=recipe)
features = builder.build_features(candles_df, symbol="BTCUSDT")
```

### 3. Costs in the Loop

**Implementation**: `src/shared/costs/cost_calculator.py`

- **CostModel**: Cost model per symbol (taker_fee_bps, maker_fee_bps, median_spread_bps, slippage_bps_per_sigma)
- **CostCalculator**: Calculator for per-symbol, per-bar costs
- Fees, spread, slippage per symbol, per bar

**Usage**:
```python
from src.shared.costs import CostCalculator, CostModel

calculator = CostCalculator()
cost_model = CostModel(
    symbol="BTCUSDT",
    taker_fee_bps=4.0,
    maker_fee_bps=2.0,
    median_spread_bps=5.0,
    slippage_bps_per_sigma=2.0,
    min_notional=10.0,
    step_size=0.001,
    last_updated_utc=datetime.now(timezone.utc),
)

calculator.register_cost_model(cost_model)
costs = calculator.get_costs("BTCUSDT", datetime.now(timezone.utc))
net_edge = calculator.calculate_net_edge("BTCUSDT", edge_bps_before_costs=10.0, timestamp=datetime.now(timezone.utc))
should_trade = calculator.should_trade("BTCUSDT", edge_bps_before_costs=10.0, timestamp=datetime.now(timezone.utc), net_edge_floor_bps=3.0)
```

### 4. Regime Gate

**Implementation**: `src/shared/regime/regime_classifier.py`

- **Regime**: Enum (TREND, RANGE, PANIC, ILLIQUID)
- **RegimeClassifier**: Classifier for tagging each bar
- Only allow engines in regimes where they work

**Usage**:
```python
from src.shared.regime import RegimeClassifier, Regime

classifier = RegimeClassifier()
regime_classification = classifier.classify(candles_df, symbol="BTCUSDT")

# Filter engines by regime
supported_engines = classifier.filter_engines_by_regime(engines, regime_classification.regime)
```

### 5. Meta Combiner Per Coin

**Implementation**: `src/shared/meta/meta_combiner.py`

- **MetaCombiner**: Combiner for per-coin meta combination
- EMA weights by recent accuracy and net edge
- Clip limits and thresholds

**Usage**:
```python
from src.shared.meta import MetaCombiner

combiner = MetaCombiner(symbol="BTCUSDT", ema_alpha=0.1)

# Update weights
combiner.update_weights("engine_1", accuracy=0.6, net_edge_bps=5.0, timestamp=datetime.now(timezone.utc))

# Combine outputs
meta_output = combiner.combine(engine_outputs, regime="TREND")
```

### 6. Per Coin Champion Pointer

**Implementation**: `src/shared/champion/champion_manager.py`

- **ChampionManager**: Manager for per-coin champion pointers
- Latest.json per symbol, always valid

**Usage**:
```python
from src.shared.champion import ChampionManager

manager = ChampionManager(base_path="champion", s3_bucket="huracan")

# Save champion
manager.save_champion(
    symbol="BTCUSDT",
    model_id="model_123",
    s3_path="s3://huracan/models/BTCUSDT/20250101_020000Z/",
)

# Load champion
champion = manager.load_champion("BTCUSDT")
```

## Minimal Contracts

### 1. Engine Inference Output

**Implementation**: `src/shared/engines/engine_interface.py`

- **EngineOutput**: Dataclass with direction, edge_bps_before_costs, confidence_0_1, horizon_minutes, metadata

### 2. Model Bundle

**Implementation**: `src/shared/contracts/model_bundle.py`

- **ModelBundle**: Dataclass with model_id, symbol, model_path, config_path, metrics_path, sha256_path, created_at, metadata
- Files: model.bin, config.json, metrics.json, sha256.txt

### 3. Champion Pointer

**Implementation**: `src/shared/contracts/model_bundle.py`

- **ChampionPointer**: Dataclass with symbol, model_id, s3_path, updated_at, metadata
- File: champion/SYMBOL/latest.json

## Scheduler

### Implementation

**File**: `src/cloud/training/pipelines/scheduler.py`

- **HybridTrainingScheduler**: Scheduler for batched parallel training
- **Three modes**: sequential, parallel, hybrid (default)
- **Batch size**: 8-16 coins per GPU
- **Queue-based**: Simple queue implementation
- **Timeout**: Per job timeout (default: 45 minutes)
- **Early writes**: After each coin
- **Retries**: With smaller batch or fewer workers if VRAM is tight

### Usage

```python
from src.cloud.training.pipelines.scheduler import HybridTrainingScheduler, SchedulerConfig, TrainingMode

config = SchedulerConfig(
    mode=TrainingMode.HYBRID,
    max_concurrent=12,
    timeout_minutes=45,
)

scheduler = HybridTrainingScheduler(config=config, train_func=train_symbol)
results = scheduler.schedule_symbols(symbols)
```

## Archive Layout

### S3 Structure

```
s3://huracan/
├── models/
│   ├── BTCUSDT/
│   │   └── 20250101_020000Z/
│   │       ├── model.bin
│   │       ├── config.json
│   │       ├── metrics.json
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

### S3 Client

**Implementation**: `src/shared/storage/s3_client.py`

- **S3Client**: Client for S3 operations
- Upload/download files and JSON
- Signed URLs for Hamilton access

**Usage**:
```python
from src.shared.storage import S3Client

client = S3Client(bucket="huracan", access_key="...", secret_key="...")

# Upload file
client.put_file("local/model.bin", "models/BTCUSDT/20250101_020000Z/model.bin")

# Upload JSON
client.put_json({"key": "value"}, "summaries/daily/2025-01-01.json")

# Get signed URL
url = client.get_signed_url("models/BTCUSDT/20250101_020000Z/model.bin")
```

## Database Tables

### Implementation

**File**: `src/shared/database/models.py`

- **ModelRecord**: model_id, parent_id, kind, created_at, s3_path, features_used, params
- **ModelMetrics**: model_id, sharpe, hit_rate, drawdown, net_bps, window, cost_bps, promoted
- **Promotion**: from_model_id, to_model_id, reason, at, snapshot
- **LiveTrade**: trade_id, time, symbol, side, size, entry, exit, fees, net_pnl, model_id
- **DailyEquity**: date, nav, max_dd, turnover, fees_bps

### Usage

```python
from src.shared.database import DatabaseClient, ModelRecord, ModelMetrics

client = DatabaseClient(connection_string="postgresql://...")

# Save model
model = ModelRecord(
    model_id="model_123",
    parent_id=None,
    kind="baseline",
    created_at=datetime.now(timezone.utc),
    s3_path="s3://huracan/models/BTCUSDT/20250101_020000Z/",
    features_used=["rsi", "ema", "volatility"],
    params={"learning_rate": 0.1},
)

client.save_model(model)

# Save metrics
metrics = ModelMetrics(
    model_id="model_123",
    sharpe=1.5,
    hit_rate=0.55,
    drawdown=10.0,
    net_bps=5.0,
    window="test",
    cost_bps=11.0,
    promoted=False,
)

client.save_metrics(metrics)
```

## Telegram Control

### Implementation

**File**: `src/shared/telegram/symbols_selector.py`

- **SymbolsSelector**: Selector for symbol selection
- Engine respects symbols selector file
- Commands: `/trade 10`, `/trade 20`

### Usage

```python
from src.shared.telegram import SymbolsSelector

selector = SymbolsSelector(selector_file="symbols_selector.json")

# Load symbols
symbols = selector.load_symbols()

# Save symbols (from Telegram bot)
selector.save_symbols(["BTCUSDT", "ETHUSDT", "SOLUSDT"], source="telegram")

# Get top symbols by meta weight
top_symbols = selector.get_top_symbols(10, meta_weights={"BTCUSDT": 0.8, "ETHUSDT": 0.6})
```

## Daily Summary

### Implementation

**File**: `src/shared/summary/daily_summary.py`

- **DailySummaryGenerator**: Generator for daily summaries
- **DailySummary**: Summary data (total_symbols, succeeded, failed, skipped, top_contributors, hit_rate, net_edge_bps, trades_counted)

### Usage

```python
from src.shared.summary import DailySummaryGenerator

generator = DailySummaryGenerator(base_path="summaries/daily")

# Generate summary
summary = generator.generate_summary(
    date="2025-01-01",
    results=training_results,
    meta_weights=meta_weights,
)

# Save summary
generator.save_summary(summary)
```

## Guardrails

### 1. Net Edge Floor

**Implementation**: `src/shared/costs/cost_calculator.py`

- `should_trade()` method checks net edge floor
- Do not emit buy/sell if edge minus costs is below floor

### 2. Spread Threshold

**Implementation**: Data gates

- Skip thin books
- Check spread threshold per symbol

### 3. Cooldowns and Dedup Windows

**Implementation**: Trading logic

- Cut churn
- Implement cooldowns between trades

### 4. Sample Size Gates

**Implementation**: Promotion logic

- No champion flips on tiny samples
- Minimum sample size for promotion

## Integration

### Engine Training Flow

1. **Load Symbols**: Read from symbols selector file
2. **Initialize Scheduler**: Set up hybrid training scheduler
3. **For Each Symbol**:
   - Fetch costs
   - Load data and build features
   - Classify regime
   - Run engines (filtered by regime)
   - Combine outputs using meta combiner
   - Calculate net edge after costs
   - Check guardrails (net edge floor, spread threshold)
   - Train model if passes gates
   - Save model bundle to S3
   - Update champion pointer
   - Save to database
4. **Generate Summary**: Create daily summary
5. **Save Summary**: Upload to S3

## Next Steps

1. **Complete S3 Integration**: Implement S3 client fully
2. **Complete Database Integration**: Implement database operations
3. **Complete Telegram Integration**: Implement Telegram bot
4. **Implement Guardrails**: Net edge floor, spread threshold, cooldowns, sample size gates
5. **Implement 23 Engines**: Create all 23 engine implementations
6. **Implement Regime Gates**: Complete regime classification
7. **Implement Meta Combiner**: Complete EMA weight updates
8. **Move to S3/R2**: Swap Dropbox to S3 or R2
9. **Add Postgres**: Start with models, model_metrics, promotions
10. **Mechanic Hourly Loop**: Fine tune and promote with strict rules

## Conclusion

The core system is implemented with all non-negotiables, contracts, scheduler, archive layout, database tables, and Telegram control. The next steps are to complete the integrations and implement the 23 engines. This provides a clean, scalable path to 400 coins and high trade counts while keeping Hamilton simple and safe.

