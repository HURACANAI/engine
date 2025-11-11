# Dropbox Folder Structure

## Overview

Every engine run creates a **dated folder** (YYYY-MM-DD format) containing all output data for that day. This makes it easy to track what happened on each day and access historical data.

## Folder Structure

```
Dropbox/
└── Runpodhuracan/
    ├── 2025-11-08/                    # Today's run (November 8, 2025)
    │   ├── data/
    │   │   └── candles/               # Historical coin data (parquet files)
    │   │       ├── BTC-USDT.parquet
    │   │       ├── ETH-USDT.parquet
    │   │       └── ...
    │   ├── learning/                  # Everything the engine learned
    │   │   ├── learning_snapshot_20251108_020000.json
    │   │   ├── learning_snapshot_20251108_030000.json
    │   │   └── ...
    │   ├── models/                    # Trained models (for Hamilton to use)
    │   │   ├── BTC-USDT_model.pkl
    │   │   ├── ETH-USDT_model.pkl
    │   │   └── ...
    │   ├── logs/                      # All engine logs
    │   │   ├── engine_monitoring_20251108_020000.log
    │   │   ├── training.log
    │   │   └── ...
    │   ├── monitoring/                # Monitoring data (JSON)
    │   │   ├── health_check_20251108.json
    │   │   ├── performance_metrics.json
    │   │   └── ...
    │   ├── reports/                   # Reports and analytics
    │   │   ├── training_report.json
    │   │   ├── performance_analysis.csv
    │   │   └── ...
    │   ├── exports/                   # COMPREHENSIVE DATA EXPORTS (A-Z)
    │   │   ├── trade_history_2025-11-08.csv
    │   │   ├── all_trades_complete_2025-11-08.csv
    │   │   ├── model_performance_2025-11-08.csv
    │   │   ├── win_loss_analysis_2025-11-08.json
    │   │   ├── pattern_library_2025-11-08.csv
    │   │   ├── pattern_performance_2025-11-08.csv
    │   │   ├── post_exit_tracking_2025-11-08.csv
    │   │   ├── regime_analysis_2025-11-08.csv
    │   │   ├── model_evolution_2025-11-08.csv
    │   │   ├── observability_trades_2025-11-08.csv
    │   │   ├── observability_models_2025-11-08.csv
    │   │   ├── observability_model_deltas_2025-11-08.csv
    │   │   ├── comprehensive_metrics_2025-11-08.json
    │   │   ├── performance_summary_2025-11-08.json
    │   │   └── ... (all other exports)
    │   └── config/                    # Configuration files used
    │       ├── base.yaml
    │       ├── local.yaml
    │       └── ...
    ├── 2025-11-07/                    # Previous day's run
    │   └── ... (same structure)
    └── 2025-11-06/                    # Even earlier run
        └── ... (same structure)
```

## Folder Contents

### 📊 `data/candles/`
**Historical coin data** - All downloaded candle data (OHLCV) for training
- **Format**: Parquet files (`.parquet`)
- **Naming**: `{SYMBOL}.parquet` (e.g., `BTC-USDT.parquet`)
- **Contains**: Historical price, volume, and market data
- **Sync Frequency**: Every 2 hours
- **Purpose**: Training data for models

### 📚 `learning/`
**Everything the engine learned** - All insights, patterns, and discoveries
- **Format**: JSON files (`.json`)
- **Naming**: `learning_snapshot_{timestamp}.json`
- **Contains**:
  - Pattern detections
  - Model improvements
  - Feature importance changes
  - Error learnings
  - Performance insights
- **Sync Frequency**: Every 5 minutes
- **Purpose**: Track what the engine learned over time

### 🤖 `models/`
**Trained models** - Models ready for Hamilton to use
- **Format**: Pickle files (`.pkl`)
- **Naming**: `{SYMBOL}_model.pkl`
- **Contains**: Trained ML models with weights and parameters
- **Sync Frequency**: Every 30 minutes
- **Purpose**: Models for live trading (Hamilton)

### 📝 `logs/`
**All engine logs** - Complete log of everything that happened
- **Format**: Log files (`.log`)
- **Naming**: `engine_monitoring_{timestamp}.log`, `training.log`, etc.
- **Contains**:
  - Training progress
  - Errors and warnings
  - System status
  - Debug information
- **Sync Frequency**: Every 5 minutes
- **Purpose**: Debugging and monitoring

### 📈 `monitoring/`
**Monitoring data** - Health checks and performance metrics
- **Format**: JSON files (`.json`)
- **Naming**: `health_check_{date}.json`, `performance_metrics.json`
- **Contains**:
  - Health check results
  - Performance metrics
  - System status
  - Alert information
- **Sync Frequency**: Every 5 minutes
- **Purpose**: System health monitoring

### 📋 `reports/`
**Reports and analytics** - Generated reports and analysis
- **Format**: JSON, CSV, HTML, PDF files
- **Contains**:
  - Training reports
  - Performance analysis
  - Model evaluation reports
  - Analytics dashboards
- **Sync Frequency**: Every 5 minutes (with logs)
- **Purpose**: Analysis and reporting

### ⚙️ `config/`
**Configuration files** - Config files used for this run
- **Format**: YAML, JSON, TOML files
- **Contains**:
  - Base configuration
  - Environment-specific config
  - Settings used for training
- **Sync Frequency**: On initial sync only
- **Purpose**: Reproducibility and debugging

### 📦 `exports/`
**COMPREHENSIVE DATA EXPORTS** - EVERYTHING A-Z exported to files
- **Format**: CSV, JSON files
- **Contains**:
  - **Trade History**: All trades (today's + complete history)
  - **Model Performance**: Daily performance metrics
  - **Win/Loss Analysis**: Detailed analysis of every win and loss
  - **Pattern Library**: All learned patterns with performance
  - **Pattern Performance**: Pattern-specific metrics
  - **Post-Exit Tracking**: What happened after we exited trades
  - **Regime Analysis**: Performance by market regime
  - **Model Evolution**: How models evolved over time
  - **Observability Data**: SQLite journal data (trades, models, deltas)
  - **Learning Snapshots**: Everything the engine learned
  - **Backtest Results**: All backtest outcomes
  - **Training Artifacts**: Model metadata and component models
  - **Comprehensive Metrics**: Complete metrics summary
  - **Performance Summary**: Export summary and metadata
- **Sync Frequency**: Every 30 minutes (with logs sync)
- **Purpose**: Complete backup and analysis of ALL engine data

## Benefits of This Structure

1. ✅ **Daily Organization**: Each day's data is self-contained
2. ✅ **Easy Access**: Find data by date quickly
3. ✅ **Complete History**: Everything that happened is stored
4. ✅ **Reproducibility**: Can replay any day's training
5. ✅ **Analysis**: Easy to compare different days
6. ✅ **Backup**: Complete backup of all engine outputs

## What Gets Synced

| Folder | Content | Sync Frequency | Purpose |
|--------|---------|----------------|---------|
| `data/candles/` | Historical coin data | Every 2 hours | Training data |
| `learning/` | Engine learnings | Every 5 minutes | Track insights |
| `models/` | Trained models | Every 30 minutes | For Hamilton |
| `logs/` | Engine logs | Every 5 minutes | Debugging |
| `monitoring/` | Monitoring data | Every 5 minutes | Health checks |
| `reports/` | Reports & analytics | Every 5 minutes | Analysis |
| `exports/` | **COMPREHENSIVE EXPORTS (A-Z)** | Every 30 minutes | Complete backup |
| `config/` | Config files | On startup | Reproducibility |

## Accessing Data

### Find Data by Date
```
/Runpodhuracan/2025-11-08/learning/    # Today's learnings
/Runpodhuracan/2025-11-08/models/      # Today's models
/Runpodhuracan/2025-11-08/logs/        # Today's logs
```

### Find Specific Data
```
/Runpodhuracan/2025-11-08/data/candles/BTC-USDT.parquet    # BTC historical data
/Runpodhuracan/2025-11-08/models/BTC-USDT_model.pkl        # BTC model
/Runpodhuracan/2025-11-08/learning/learning_snapshot_*.json # Learning snapshots
```

## Summary

Every engine run creates a **dated folder** containing:
- ✅ **Coin data** (`data/candles/`) - Historical market data
- ✅ **Learning** (`learning/`) - Everything the engine learned
- ✅ **Models** (`models/`) - Trained models for Hamilton
- ✅ **Logs** (`logs/`) - Complete log of everything
- ✅ **Monitoring** (`monitoring/`) - Health and performance data
- ✅ **Reports** (`reports/`) - Generated reports and analytics
- ✅ **Exports** (`exports/`) - **COMPREHENSIVE A-Z EXPORTS** (trade history, win/loss, patterns, metrics, etc.)
- ✅ **Config** (`config/`) - Configuration files used

**Everything is organized by date, making it easy to track and analyze the engine's progress!** 🚀

## 🎯 Comprehensive Exports (A-Z Coverage)

The `exports/` folder contains **COMPLETE exports of ALL engine data**:

### PostgreSQL Database Exports
- ✅ **Trade History** - All trades (today's + complete history)
- ✅ **Model Performance** - Daily performance metrics
- ✅ **Win/Loss Analysis** - Detailed analysis of every win and loss
- ✅ **Pattern Library** - All learned patterns
- ✅ **Pattern Performance** - Pattern-specific metrics
- ✅ **Post-Exit Tracking** - What happened after we exited
- ✅ **Regime Analysis** - Performance by market regime
- ✅ **Model Evolution** - How models evolved over time

### SQLite Observability Exports
- ✅ **Observability Trades** - Trade records from journal
- ✅ **Observability Models** - Model records from journal
- ✅ **Model Deltas** - Model change tracking

### File System Exports
- ✅ **Learning Snapshots** - Everything the engine learned
- ✅ **Backtest Results** - All backtest outcomes
- ✅ **Training Artifacts** - Model metadata and components

### Metrics & Summaries
- ✅ **Comprehensive Metrics** - Complete metrics summary
- ✅ **Performance Summary** - Export summary and metadata

**This ensures COMPLETE backup and analysis of EVERYTHING the engine does!** 📊

