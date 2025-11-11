# Implementation Roadmap

## Overview

This document outlines the implementation roadmap for the Engine system with non-negotiables, contracts, scheduler, archive layout, database tables, and Telegram control.

## Non-Negotiables

### ✅ Implemented

1. **Engine Interface** - One unified interface for all 23 engines
   - `src/shared/engines/engine_interface.py`
   - `BaseEngine` abstract class
   - `EngineInput` and `EngineOutput` dataclasses
   - `EngineRegistry` for managing engines

2. **Shared Feature Builder** - One shared feature builder
   - `src/shared/features/feature_builder.py`
   - `FeatureRecipe` dataclass
   - `FeatureBuilder` class
   - Same recipe for cloud and Hamilton

3. **Cost Calculator** - Costs in the loop
   - `src/shared/costs/cost_calculator.py`
   - `CostModel` dataclass
   - `CostCalculator` class
   - Fees, spread, slippage per symbol, per bar

4. **Regime Classifier** - Regime gate
   - `src/shared/regime/regime_classifier.py`
   - `Regime` enum (TREND, RANGE, PANIC, ILLIQUID)
   - `RegimeClassifier` class
   - Only allow engines in regimes where they work

5. **Meta Combiner** - Per coin meta combiner
   - `src/shared/meta/meta_combiner.py`
   - `MetaCombiner` class
   - EMA weights by recent accuracy and net edge

6. **Champion Manager** - Per coin champion pointer
   - `src/shared/champion/champion_manager.py`
   - `ChampionManager` class
   - Latest.json per symbol, always valid

### 🔄 In Progress

1. **S3 Storage Client** - S3 client for storage
   - `src/shared/storage/s3_client.py`
   - `S3Client` class
   - Upload/download files and JSON

2. **Database Models** - Database tables
   - `src/shared/database/models.py`
   - Model records, metrics, promotions, live trades, daily equity

3. **Telegram Control** - Symbols selector
   - `src/shared/telegram/symbols_selector.py`
   - `SymbolsSelector` class
   - Engine respects symbols selector file

4. **Daily Summary** - Daily summary generator
   - `src/shared/summary/daily_summary.py`
   - `DailySummaryGenerator` class
   - Top contributors, hit rate, net edge, trades counted

## Minimal Contracts

### ✅ Implemented

1. **Engine Inference Output**
   - `EngineOutput` dataclass
   - `direction`: buy, sell, wait
   - `edge_bps_before_costs`: Expected edge in basis points
   - `confidence_0_1`: Confidence score (0.0 to 1.0)
   - `horizon_minutes`: Prediction horizon
   - `metadata`: Additional metadata

2. **Model Bundle**
   - `ModelBundle` dataclass
   - `model.bin`: Trained model
   - `config.json`: Model configuration
   - `metrics.json`: Performance metrics
   - `sha256.txt`: Integrity hash

3. **Champion Pointer**
   - `ChampionPointer` dataclass
   - `champion/SYMBOL/latest.json`: Champion pointer file
   - `bucket_path`: S3 path to model bundle
   - `model_id`: Model identifier

## Scheduler

### ✅ Implemented

1. **Hybrid Training Scheduler**
   - `src/cloud/training/pipelines/scheduler.py`
   - Three modes: sequential, parallel, hybrid
   - Batched parallel training (8-16 coins per GPU)
   - Simple queue-based implementation

2. **Timeout and Retries**
   - Timeout per job (default: 45 minutes)
   - Early writes after each coin
   - Retries with smaller batch or fewer workers if VRAM is tight

### 🔄 To Do

1. **VRAM Management**
   - Monitor VRAM usage
   - Adjust batch size dynamically
   - Reduce workers if VRAM is tight

## Archive Layout

### ✅ Implemented

1. **S3 Structure**
   - `s3://huracan/models/SYMBOL/TIMESTAMP/`
   - `s3://huracan/challengers/SYMBOL/TIMESTAMP/`
   - `s3://huracan/champion/SYMBOL/latest.json`
   - `s3://huracan/summaries/daily/DATE.json`
   - `s3://huracan/live_logs/trades/YYYYMMDD.parquet`

2. **S3 Client**
   - `S3Client` class
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
   - `DatabaseClient` class
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

### 🔄 To Do

1. **Net Edge Floor**
   - Do not emit buy/sell if edge minus costs is below floor
   - Implement in `CostCalculator.should_trade()`

2. **Spread Threshold**
   - Skip thin books
   - Implement in data gates

3. **Cooldowns and Dedup Windows**
   - Cut churn
   - Implement in trading logic

4. **Sample Size Gates**
   - No champion flips on tiny samples
   - Implement in promotion logic

## Next Steps

### Immediate

1. **Complete S3 Integration**
   - Implement S3 client fully
   - Test upload/download
   - Add signed URLs

2. **Complete Database Integration**
   - Implement database operations
   - Create tables
   - Test save/load operations

3. **Complete Telegram Integration**
   - Implement Telegram bot
   - Add `/trade` commands
   - Test symbol selection

4. **Implement Guardrails**
   - Net edge floor
   - Spread threshold
   - Cooldowns
   - Sample size gates

### Short Term

1. **Implement 23 Engines**
   - Create all 23 engine implementations
   - Register in engine registry
   - Test each engine

2. **Implement Regime Gates**
   - Complete regime classification
   - Filter engines by regime
   - Test regime gates

3. **Implement Meta Combiner**
   - Complete EMA weight updates
   - Test meta combination
   - Validate weights

### Long Term

1. **Move to S3/R2**
   - Swap Dropbox to S3 or R2
   - Keep same folder shape
   - Add signed URLs

2. **Add Postgres**
   - Start with models, model_metrics, promotions
   - Add live tables later
   - Test database operations

3. **Mechanic Hourly Loop**
   - Fine tune and promote with strict rules
   - Implement promotion logic
   - Test promotion flow

## Acceptance Criteria

### ✅ Completed

1. Engine interface for all 23 engines
2. Shared feature builder
3. Cost calculator
4. Regime classifier
5. Meta combiner
6. Champion manager
7. Model bundle contract
8. Champion pointer contract
9. Hybrid training scheduler
10. S3 client structure
11. Database models
12. Symbols selector
13. Daily summary generator

### 🔄 In Progress

1. S3 client implementation
2. Database client implementation
3. Telegram bot integration
4. Guardrails implementation

### ⏳ Pending

1. 23 engine implementations
2. Regime gate implementation
3. Meta combiner weight updates
4. Promotion logic
5. Live trade tracking
6. Daily equity tracking

## Conclusion

The core infrastructure is in place. The next steps are to complete the S3 and database integrations, implement the 23 engines, and add the guardrails. This will provide a clean, scalable path to 400 coins and high trade counts while keeping Hamilton simple and safe.

