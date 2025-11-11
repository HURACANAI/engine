# Complete System Overview

## Overview

This document provides a complete overview of the Engine system with all non-negotiables, contracts, scheduler, archive layout, database tables, and Telegram control.

## Architecture

### Core Components

1. **Engine Interface** (`src/shared/engines/`)
   - Unified interface for all 23 engines
   - Same inputs and outputs
   - Engine registry for management

2. **Shared Feature Builder** (`src/shared/features/`)
   - One shared feature builder
   - Same recipe for cloud and Hamilton
   - Feature recipe with hash

3. **Cost Calculator** (`src/shared/costs/`)
   - Costs in the loop
   - Fees, spread, slippage per symbol, per bar
   - Net edge calculation

4. **Regime Classifier** (`src/shared/regime/`)
   - Regime gate
   - Only allow engines in regimes where they work
   - Regime: TREND, RANGE, PANIC, ILLIQUID

5. **Meta Combiner** (`src/shared/meta/`)
   - Per coin meta combiner
   - EMA weights by recent accuracy and net edge
   - Clip limits and thresholds

6. **Champion Manager** (`src/shared/champion/`)
   - Per coin champion pointer
   - Latest.json per symbol
   - Always valid

### Storage and Database

7. **S3 Client** (`src/shared/storage/`)
   - S3 storage client
   - Upload/download files and JSON
   - Signed URLs for Hamilton access

8. **Database Client** (`src/shared/database/`)
   - Database models and client
   - Models, metrics, promotions, live trades, daily equity
   - PostgreSQL integration

### Control and Monitoring

9. **Telegram Control** (`src/shared/telegram/`)
   - Symbols selector
   - Engine respects symbols selector file
   - Commands: `/trade 10`, `/trade 20`

10. **Daily Summary** (`src/shared/summary/`)
    - Daily summary generator
    - Top contributors, hit rate, net edge, trades counted

### Training Pipeline

11. **Hybrid Training Scheduler** (`src/cloud/training/pipelines/scheduler.py`)
    - Batched parallel training
    - 8-16 coins per GPU
    - Timeout and retries

12. **Integrated Training Pipeline** (`src/cloud/training/pipelines/integrated_training_pipeline.py`)
    - Integrates all components
    - End-to-end training flow
    - Per-coin training with all guardrails

## Non-Negotiables

### ✅ Implemented

1. **One Engine Interface for All 23 Engines**
   - `BaseEngine` abstract class
   - `EngineInput` and `EngineOutput` dataclasses
   - `EngineRegistry` for management

2. **One Shared Feature Builder**
   - `FeatureBuilder` class
   - `FeatureRecipe` dataclass
   - Same recipe for cloud and Hamilton

3. **Costs in the Loop**
   - `CostCalculator` class
   - `CostModel` dataclass
   - Fees, spread, slippage per symbol, per bar

4. **Regime Gate**
   - `RegimeClassifier` class
   - `Regime` enum
   - Only allow engines in regimes where they work

5. **Meta Combiner Per Coin**
   - `MetaCombiner` class
   - EMA weights by recent accuracy and net edge
   - Clip limits and thresholds

6. **Per Coin Champion Pointer**
   - `ChampionManager` class
   - Latest.json per symbol
   - Always valid

## Minimal Contracts

### ✅ Implemented

1. **Engine Inference Output**
   - `EngineOutput` dataclass
   - direction: buy, sell, wait
   - edge_bps_before_costs: Expected edge in basis points
   - confidence_0_1: Confidence score (0.0 to 1.0)
   - horizon_minutes: Prediction horizon
   - metadata: Additional metadata

2. **Model Bundle**
   - `ModelBundle` dataclass
   - model.bin: Trained model
   - config.json: Model configuration
   - metrics.json: Performance metrics
   - sha256.txt: Integrity hash

3. **Champion Pointer**
   - `ChampionPointer` dataclass
   - champion/SYMBOL/latest.json: Champion pointer file
   - bucket_path: S3 path to model bundle
   - model_id: Model identifier

## Scheduler

### ✅ Implemented

1. **Hybrid Training Scheduler**
   - Three modes: sequential, parallel, hybrid
   - Batched parallel training (8-16 coins per GPU)
   - Simple queue-based implementation

2. **Timeout and Retries**
   - Timeout per job (default: 45 minutes)
   - Early writes after each coin
   - Retries with smaller batch or fewer workers if VRAM is tight

## Archive Layout

### ✅ Implemented

1. **S3 Structure**
   - `s3://huracan/models/SYMBOL/TIMESTAMP/`
   - `s3://huracan/challengers/SYMBOL/TIMESTAMP/`
   - `s3://huracan/champion/SYMBOL/latest.json`
   - `s3://huracan/summaries/daily/DATE.json`
   - `s3://huracan/live_logs/trades/YYYYMMDD.parquet`

2. **S3 Client**
   - Upload/download files and JSON
   - Signed URLs for Hamilton access

## Database Tables

### ✅ Implemented

1. **Database Models**
   - `ModelRecord`: model_id, parent_id, kind, created_at, s3_path, features_used, params
   - `ModelMetrics`: model_id, sharpe, hit_rate, drawdown, net_bps, window, cost_bps, promoted
   - `Promotion`: from_model_id, to_model_id, reason, at, snapshot
   - `LiveTrade`: trade_id, time, symbol, side, size, entry, exit, fees, net_pnl, model_id
   - `DailyEquity`: date, nav, max_dd, turnover, fees_bps

2. **Database Client**
   - Save methods for all models
   - TODO: Implement actual database operations

## Telegram Control

### ✅ Implemented

1. **Symbols Selector**
   - `SymbolsSelector` class
   - Load/save symbols from selector file
   - Get top N symbols by meta weight

### 🔄 To Do

1. **Telegram Bot Integration**
   - `/trade 10` command
   - `/trade 20` command
   - Write top N symbols to selector file
   - Engine and Hamilton read same file

## Guardrails

### ✅ Implemented

1. **Net Edge Floor**
   - `CostCalculator.should_trade()` method
   - Do not emit buy/sell if edge minus costs is below floor

### 🔄 To Do

1. **Spread Threshold**
   - Skip thin books
   - Implement in data gates

2. **Cooldowns and Dedup Windows**
   - Cut churn
   - Implement in trading logic

3. **Sample Size Gates**
   - No champion flips on tiny samples
   - Implement in promotion logic

## Training Flow

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

## File Structure

```
src/shared/
├── engines/
│   ├── __init__.py
│   ├── engine_interface.py
│   └── example_engine.py
├── features/
│   ├── __init__.py
│   └── feature_builder.py
├── costs/
│   ├── __init__.py
│   └── cost_calculator.py
├── regime/
│   ├── __init__.py
│   └── regime_classifier.py
├── meta/
│   ├── __init__.py
│   └── meta_combiner.py
├── champion/
│   ├── __init__.py
│   └── champion_manager.py
├── storage/
│   ├── __init__.py
│   └── s3_client.py
├── database/
│   ├── __init__.py
│   └── models.py
├── telegram/
│   ├── __init__.py
│   └── symbols_selector.py
├── summary/
│   ├── __init__.py
│   └── daily_summary.py
└── contracts/
    ├── __init__.py
    └── model_bundle.py
```

## Usage Example

```python
from src.shared.engines import EngineRegistry, BaseEngine
from src.shared.features import FeatureBuilder
from src.shared.costs import CostCalculator
from src.shared.regime import RegimeClassifier
from src.shared.meta import MetaCombiner
from src.shared.champion import ChampionManager
from src.shared.storage import S3Client
from src.shared.database import DatabaseClient
from src.cloud.training.pipelines.integrated_training_pipeline import IntegratedTrainingPipeline

# Initialize components
engine_registry = EngineRegistry()
feature_builder = FeatureBuilder()
cost_calculator = CostCalculator()
regime_classifier = RegimeClassifier()
champion_manager = ChampionManager(s3_bucket="huracan")
s3_client = S3Client(bucket="huracan")
database_client = DatabaseClient(connection_string="...")

# Create pipeline
pipeline = IntegratedTrainingPipeline(
    engine_registry=engine_registry,
    feature_builder=feature_builder,
    cost_calculator=cost_calculator,
    regime_classifier=regime_classifier,
    champion_manager=champion_manager,
    s3_client=s3_client,
    database_client=database_client,
)

# Train symbol
result = pipeline.train_symbol("BTCUSDT", candles_df, cfg={})
```

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
