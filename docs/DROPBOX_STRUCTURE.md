# Dropbox File Structure Documentation

This document describes the organized Dropbox file structure for the Huracan Engine system, designed for easy access by Hamilton and other modules.

## Overview

The Dropbox structure is organized to:
- **Separate shared data** from dated/archived data
- **Provide clear paths** for Hamilton to access models and configs
- **Organize data by type** (models, data, configs, logs, etc.)
- **Enable easy programmatic access** for all modules

## Structure

```
/{app_folder}/                              # Default: "Runpodhuracan"
  ├── data/                                 # Shared data (persists across days)
  │   ├── candles/                         # Historical candle data
  │   │   ├── BTCUSDT/                     # Organized by symbol
  │   │   │   ├── BTCUSDT_1d_20250615_20251112.parquet
  │   │   │   └── ...
  │   │   ├── ETHUSDT/
  │   │   │   └── ...
  │   │   └── ...
  │   ├── features/                        # Feature data
  │   └── market_data/                     # Other market data
  │
  ├── models/                              # Trained models
  │   ├── champions/                      # Champion models (latest, best)
  │   │   ├── latest/                     # Latest champion for each coin
  │   │   │   ├── BTCUSDT.bin
  │   │   │   ├── ETHUSDT.bin
  │   │   │   └── ...
  │   │   └── archive/                    # Historical champions
  │   │       ├── 2025-11-12/
  │   │       │   ├── BTCUSDT.bin
  │   │       │   └── ...
  │   │       └── ...
  │   └── training/                       # Training artifacts
  │       ├── 2025-11-12/                # Dated training runs
  │       │   ├── BTCUSDT/
  │       │   │   ├── model/
  │       │   │   │   ├── model.bin
  │       │   │   │   └── model_metadata.json
  │       │   │   ├── metrics/
  │       │   │   │   ├── training_metrics.json
  │       │   │   │   └── costs.json
  │       │   │   ├── features/
  │       │   │   │   └── features.json
  │       │   │   └── data/
  │       │   │       └── candles.parquet
  │       │   └── ...
  │       └── ...
  │
  ├── hamilton/                           # Hamilton-specific exports (live trading)
  │   ├── roster.json                    # Current roster (ranked coins)
  │   ├── champion.json                  # Current champion pointer
  │   ├── configs/                       # Hamilton configs
  │   │   ├── BTCUSDT.json
  │   │   ├── ETHUSDT.json
  │   │   └── ...
  │   ├── manifests/                     # Run manifests
  │   │   └── ...
  │   └── active/                        # Active model pointers
  │       ├── BTCUSDT.txt                # Active model ID for each symbol
  │       ├── ETHUSDT.txt
  │       └── ...
  │
  ├── exports/                           # Comprehensive exports
  │   ├── trades/                        # Trade history
  │   │   ├── 2025-11-12/
  │   │   │   └── ...
  │   │   └── ...
  │   ├── metrics/                       # Performance metrics
  │   │   └── ...
  │   └── reports/                       # Reports
  │       └── ...
  │
  ├── logs/                              # Logs (dated)
  │   ├── 2025-11-12/
  │   │   └── ...
  │   └── ...
  │
  ├── config/                            # Configuration files
  │   ├── base.yaml
  │   ├── prod.yaml
  │   └── ...
  │
  └── monitoring/                        # Monitoring data
      ├── 2025-11-12/
      │   └── ...
      └── ...
```

## Key Directories

### `/data/` - Shared Data
- **Purpose**: Data that persists across days and is shared between modules
- **Structure**: Organized by data type and symbol
- **Usage**: Historical data, features, market data
- **Access**: All modules can read from here

### `/models/champions/latest/` - Latest Champion Models
- **Purpose**: Latest champion model for each symbol
- **Structure**: `{SYMBOL}.bin`
- **Usage**: Hamilton loads models from here for live trading
- **Access**: Hamilton reads, Engine writes

### `/models/champions/archive/` - Historical Champions
- **Purpose**: Archive of previous champion models
- **Structure**: `{YYYY-MM-DD}/{SYMBOL}.bin`
- **Usage**: Historical reference, rollback capability
- **Access**: Engine writes (archives on new champion)

### `/models/training/` - Training Artifacts
- **Purpose**: Complete training run artifacts
- **Structure**: `{YYYY-MM-DD}/{SYMBOL}/{type}/`
- **Usage**: Training history, analysis, debugging
- **Access**: Engine writes, Mechanic/Hamilton can read

### `/hamilton/` - Hamilton Exports
- **Purpose**: Hamilton-specific files for live trading
- **Structure**: Organized by file type (roster, configs, active, etc.)
- **Usage**: Hamilton reads all files from here
- **Access**: Engine writes, Hamilton reads

### `/hamilton/roster.json` - Trading Roster
- **Purpose**: Ranked list of coins for trading decisions
- **Format**: JSON with ranked symbols, metrics, trade_ok flags
- **Usage**: Hamilton uses this to decide which coins to trade
- **Access**: Engine writes, Hamilton reads

### `/hamilton/champion.json` - Champion Pointer
- **Purpose**: Points to current champion model
- **Format**: JSON with champion model ID and metadata
- **Usage**: Hamilton uses this to find the best model
- **Access**: Engine writes, Hamilton reads

### `/hamilton/configs/` - Hamilton Configs
- **Purpose**: Per-symbol configuration files
- **Structure**: `{SYMBOL}.json`
- **Usage**: Hamilton loads configs for each symbol
- **Access**: Engine writes, Hamilton reads

### `/hamilton/active/` - Active Model Pointers
- **Purpose**: Active model ID for each symbol
- **Structure**: `{SYMBOL}.txt` (contains model ID)
- **Usage**: Hamilton checks which model is active for each symbol
- **Access**: Engine writes (on activation), Hamilton reads

## Usage Examples

### For Hamilton (Live Trading)

```python
from src.cloud.training.integrations.dropbox_sync import DropboxSync

# Initialize Dropbox sync
sync = DropboxSync(
    access_token="your_token",
    app_folder="Runpodhuracan"
)

# Load roster
roster = sync.list_hamilton_roster()
if roster:
    trade_ok_symbols = [s["symbol"] for s in roster["symbols"] if s.get("trade_ok", False)]
    print(f"Trading {len(trade_ok_symbols)} symbols")

# Get active model for a symbol
active_model_id = sync.get_active_model("BTCUSDT")
if active_model_id:
    print(f"Active model for BTCUSDT: {active_model_id}")

# List all champion models
champions = sync.list_champion_models()
print(f"Available champions: {champions}")
```

### For Engine (Training)

```python
from src.cloud.training.integrations.dropbox_sync import DropboxSync

# Initialize Dropbox sync
sync = DropboxSync(
    access_token="your_token",
    app_folder="Runpodhuracan"
)

# Export Hamilton roster
sync.export_hamilton_roster("champions/roster.json")

# Upload champion model
sync.upload_champion_model(
    symbol="BTCUSDT",
    model_path="models/BTCUSDT/model.bin",
    archive_previous=True
)

# Set active model
sync.set_active_model("BTCUSDT", "btc_v48")

# Upload training artifacts
sync.upload_training_artifact(
    symbol="BTCUSDT",
    run_date=date.today(),
    artifact_path="models/BTCUSDT/model.bin",
    artifact_type="model"
)

# Upload candle data
sync.upload_candle_data(
    symbol="BTCUSDT",
    candle_path="data/candles/BTCUSDT_1d.parquet"
)
```

### For Data Access

```python
from src.cloud.training.integrations.dropbox_sync import DropboxSync

# Initialize Dropbox sync
sync = DropboxSync(
    access_token="your_token",
    app_folder="Runpodhuracan"
)

# Restore data cache from Dropbox
restored_count = sync.restore_data_cache(
    data_cache_dir="data/candles",
    use_latest_dated_folder=False  # Use shared location
)

print(f"Restored {restored_count} files from Dropbox")
```

## Migration Guide

### From Old Structure to New Structure

The new organized structure is backward compatible. To migrate:

1. **Use new methods**: Use the new organized structure methods (e.g., `upload_champion_model`, `export_hamilton_roster`)
2. **Enable organized structure**: Set `use_organized_structure=True` in `export_coin_results`
3. **Organize data by symbol**: Set `organize_by_symbol=True` in `upload_data_cache`

### Legacy Structure Support

The old structure is still supported for backward compatibility:
- Old structure: `/{app_folder}/coins/{SYMBOL}/{DATE}/`
- New structure: `/{app_folder}/models/training/{DATE}/{SYMBOL}/`

Set `use_organized_structure=False` to use the legacy structure.

## Best Practices

1. **Use organized structure**: Always use the new organized structure for new code
2. **Organize by symbol**: Organize data by symbol for easy access
3. **Archive champions**: Archive previous champions before uploading new ones
4. **Use shared data location**: Use shared data location for data that persists across days
5. **Use dated folders for logs**: Use dated folders for logs and monitoring data
6. **Keep Hamilton exports separate**: Keep Hamilton-specific exports in `/hamilton/` directory

## Path Conventions

- **Symbols**: Uppercase, no slashes (e.g., "BTCUSDT", "ETH-USDT" → "ETHUSDT")
- **Dates**: ISO format (YYYY-MM-DD)
- **File extensions**: 
  - Models: `.bin` or `.pkl`
  - Configs: `.json` or `.yaml`
  - Data: `.parquet` or `.json`
  - Logs: `.log`

## Access Patterns

### Hamilton (Live Trading)
1. Read `/hamilton/roster.json` to get ranked coins
2. Read `/hamilton/champion.json` to get champion pointer
3. Read `/hamilton/active/{SYMBOL}.txt` to get active model ID
4. Load model from `/models/champions/latest/{SYMBOL}.bin`
5. Load config from `/hamilton/configs/{SYMBOL}.json`

### Engine (Training)
1. Upload training artifacts to `/models/training/{DATE}/{SYMBOL}/`
2. Upload champion model to `/models/champions/latest/{SYMBOL}.bin`
3. Archive previous champion to `/models/champions/archive/{DATE}/{SYMBOL}.bin`
4. Export roster to `/hamilton/roster.json`
5. Export champion to `/hamilton/champion.json`
6. Set active model in `/hamilton/active/{SYMBOL}.txt`

### Data Access
1. Read candle data from `/data/candles/{SYMBOL}/`
2. Read features from `/data/features/`
3. Read market data from `/data/market_data/`

## Security

- **Access tokens**: Store access tokens securely (environment variables, secrets manager)
- **Read-only access**: Use read-only tokens for Hamilton if possible
- **Path validation**: Validate all paths to prevent directory traversal
- **Symbol sanitization**: Sanitize symbols to prevent path injection

## Troubleshooting

### Files Not Found
- Check if Dropbox sync is enabled
- Verify access token is valid
- Check if file exists in Dropbox
- Verify path normalization

### Upload Failures
- Check file size limits
- Verify network connection
- Check Dropbox API rate limits
- Verify access token permissions

### Path Issues
- Verify symbol format (uppercase, no slashes)
- Check date format (YYYY-MM-DD)
- Verify path normalization
- Check for special characters in paths

## References

- [Dropbox API Documentation](https://www.dropbox.com/developers/documentation)
- [Dropbox Python SDK](https://dropbox-sdk-python.readthedocs.io/)
- [Huracan Engine Documentation](../README.md)

