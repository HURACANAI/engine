# Engine Scope Documentation

**Last Updated:** 2025-01-XX  
**Status:** Engine-Only Development Phase

---

## Overview

This codebase is **ONLY for building the Engine** (Cloud Training Box).  
The Engine builds daily baseline models through full retraining on historical data.

---

## What the Engine IS

The Engine is the **Cloud Training Box** that:

1. **Daily Baseline Training**
   - Runs once per day at 02:00 UTC
   - Trains on 3-6 months of historical market data
   - Builds clean baseline models from scratch
   - Validates with walk-forward testing
   - Saves models to S3 and registers in Postgres

2. **Shadow Trading for Learning**
   - Performs paper trades on historical data
   - Learns from trade outcomes
   - Trains RL agents and ML models
   - **NO real money, NO live trading**

3. **Model Export**
   - Exports trained models for other components
   - Creates model artifacts (weights, configs, metrics)
   - Publishes to S3 for Archive/Mechanic/Pilot to use

4. **Monitoring & Health Checks**
   - Monitors training progress
   - Tracks model performance metrics
   - Generates daily reports

---

## What the Engine is NOT

The Engine does **NOT** do:

1. **Hourly Incremental Updates** (that's Mechanic)
   - Engine does full daily retraining
   - Mechanic does hourly fine-tuning (future component)

2. **Live Trading Execution** (that's Pilot)
   - Engine does shadow trading for learning only
   - Pilot does live trading with real money (future component)

3. **Real-Time Order Management** (that's Pilot)
   - Engine doesn't manage live positions
   - Pilot manages live positions and risk (future component)

---

## Main Entry Point

**File:** `src/cloud/training/pipelines/daily_retrain.py`

**Function:** `run_daily_retrain()`

**Usage:**
```bash
# Via Python module
python -m cloud.training.pipelines.daily_retrain

# Via Poetry script
cloud-training-daily-retrain
```

**Scheduling:** Runs daily at 02:00 UTC (configured via APScheduler)

---

## Engine-Only Components

These components are **actively used** by the Engine:

### Core Training
- `src/cloud/training/pipelines/daily_retrain.py` - Main entry point
- `src/cloud/training/pipelines/rl_training_pipeline.py` - RL training
- `src/cloud/training/pipelines/enhanced_rl_pipeline.py` - Enhanced RL
- `src/cloud/training/pipelines/progressive_training.py` - Progressive training

### Data Processing
- `src/cloud/engine/labeling/` - Triple-barrier labeling
- `src/cloud/engine/data_quality/` - Data cleaning pipeline
- `src/cloud/engine/costs/` - Transaction cost analysis
- `src/cloud/engine/multi_window/` - Multi-window training
- `src/cloud/engine/walk_forward.py` - Walk-forward validation

### Shadow Trading (Learning)
- `src/cloud/training/backtesting/shadow_trader.py` - Shadow trading for learning
- `src/cloud/training/agents/` - RL agents
- `src/cloud/training/analyzers/` - Trade analysis

### Models & Features
- `src/cloud/training/models/` - ML models (most are Engine)
- `src/shared/features/` - Feature engineering

### Services
- `src/cloud/training/services/exchange.py` - Exchange client
- `src/cloud/training/services/model_registry.py` - Model registry
- `src/cloud/training/services/artifacts.py` - Artifact publishing
- `src/cloud/training/services/orchestration.py` - Training orchestration

### Monitoring
- `src/cloud/training/monitoring/` - Health monitoring (Engine uses for daily checks)

### Database
- `src/cloud/training/memory/store.py` - Trade memory storage
- `src/cloud/training/database/pool.py` - Connection pooling

---

## FUTURE Components (Not Used in Engine)

These components are marked as **FUTURE** and will be used when building Mechanic/Pilot:

### Mechanic Components (Hourly Updates)

**Location:** `src/cloud/engine/incremental/`

- `incremental_labeler.py` - [FUTURE/MECHANIC] Incremental labeling for hourly updates
- `delta_detector.py` - [FUTURE/MECHANIC] Detects changes between runs
- `__init__.py` - [FUTURE/MECHANIC] Incremental training system

**Why Not Used:** Engine does full daily retraining, not incremental updates

### Pilot Components (Live Trading)

**Location:** `src/cloud/training/models/`

- `shadow_deployment.py` - [FUTURE/PILOT] Shadow deployment for live trading
- `shadow_promotion.py` - [FUTURE/PILOT] Promotion criteria for live deployment
- `trading_coordinator.py` - [FUTURE/PILOT] Live trading coordination

**Why Not Used:** Engine does shadow trading for learning, not live deployment

**Note:** `shadow_trader.py` is **USED by Engine** for learning (different purpose)

### Contracts

**Location:** `src/cloud/training/services/orchestration.py`

- `MechanicContract` - [FUTURE/MECHANIC] Contract for Mechanic integration
- `PilotContract` - [FUTURE/PILOT] Contract for Pilot integration

**Why Not Used:** These are for future integration with Mechanic/Pilot components

---

## Important Distinctions

### Shadow Trading: Engine vs Pilot

| Aspect | Engine Shadow Trading | Pilot Shadow Deployment |
|--------|---------------------|------------------------|
| **Purpose** | Learning from historical data | Testing before production |
| **Data** | Historical candles | Live market data |
| **Outcome** | Trains models | Validates models |
| **File** | `shadow_trader.py` | `shadow_deployment.py` |
| **Status** | ✅ USED | ❌ FUTURE |

### Training: Engine vs Mechanic

| Aspect | Engine | Mechanic |
|--------|--------|----------|
| **Frequency** | Daily (02:00 UTC) | Hourly |
| **Method** | Full retraining | Incremental updates |
| **Data Window** | 3-6 months | Last few hours |
| **Purpose** | Build baseline | Fine-tune baseline |
| **Status** | ✅ CURRENT | ❌ FUTURE |

---

## File Markers

Files are marked with clear headers:

- `[FUTURE/MECHANIC - NOT USED IN ENGINE]` - For Mechanic component
- `[FUTURE/PILOT - NOT USED IN ENGINE]` - For Pilot component
- `[ENGINE - USED FOR LEARNING]` - Used by Engine for learning

---

## Development Guidelines

### When Adding New Code

1. **Ask:** Is this for Engine daily training?
   - ✅ Yes → Add to Engine components
   - ❌ No → Mark as FUTURE/MECHANIC or FUTURE/PILOT

2. **Check:** Does it run hourly or do live trading?
   - Hourly → Mark as FUTURE/MECHANIC
   - Live trading → Mark as FUTURE/PILOT

3. **Verify:** Does it integrate with daily_retrain.py?
   - ✅ Yes → Engine component
   - ❌ No → May be FUTURE component

### When Reviewing Code

- Look for `[FUTURE/` markers to identify non-Engine code
- Check imports for `MechanicContract` or `PilotContract`
- Verify `daily_retrain.py` is the main entry point

---

## Architecture Flow

```
Daily Training Cycle (02:00 UTC)
    ↓
daily_retrain.py (MAIN ENTRY POINT)
    ↓
TrainingOrchestrator
    ↓
├─→ Data Loading (historical candles)
├─→ Feature Engineering
├─→ Model Training (LightGBM, RL agents)
├─→ Walk-Forward Validation
├─→ Shadow Trading (for learning)
├─→ Model Export (S3 + Postgres)
└─→ Report Generation
```

---

## Questions?

If unsure whether code belongs in Engine:

1. Does it run daily at 02:00 UTC? → Engine
2. Does it train models on historical data? → Engine
3. Does it do shadow trading for learning? → Engine
4. Does it run hourly? → FUTURE/MECHANIC
5. Does it execute live trades? → FUTURE/PILOT

---

## Summary

- **Engine = Daily baseline training** ✅ CURRENT
- **Mechanic = Hourly incremental updates** ❌ FUTURE
- **Pilot = Live trading execution** ❌ FUTURE

**Focus:** Build Engine only. Mark everything else as FUTURE.

