# Dropbox Sync Configuration Guide

## Overview

The engine now uses **optimized sync intervals** for different types of data, ensuring efficient use of bandwidth while keeping important data backed up frequently.

## What Gets Synced

### ✅ **Learning Data** (`logs/learning/*.json`)
- **What**: All insights, patterns, and learnings the engine discovers
- **Includes**:
  - Pattern detections
  - Model improvements
  - Error learnings
  - Feature importance changes
  - Performance insights
- **Sync Frequency**: Every **5 minutes** (300 seconds)
- **Why**: Capture insights quickly so they're never lost

### ✅ **Historical Coin Data** (`data/candles/*.parquet`)
- **What**: All downloaded historical candle data (OHLCV)
- **Includes**:
  - Daily/hourly candle data for all coins
  - Historical price data
  - Volume data
- **Sync Frequency**: 
  - **Newly downloaded files**: Synced within **5 minutes** ⚡
  - **Full sync**: Every **2 hours** (7200 seconds)
- **Why**: 
  - Newly downloaded coin data is synced immediately (within 5 minutes) to ensure no data loss
  - Full sync runs every 2 hours for all files (large files, don't change often)
- **Special Feature**: 
  - Automatically restored from Dropbox on startup (no need to re-download!)
  - **Immediate sync for new downloads**: When coins are downloaded, they're automatically synced to Dropbox within 5 minutes

### ✅ **Models** (`models/*.pkl`)
- **What**: Trained model files
- **Sync Frequency**: Every **30 minutes** (1800 seconds)
- **Why**: Models don't change as frequently as logs, but we want backups

### ✅ **Logs & Monitoring** (`logs/*.log`, `logs/*.json`)
- **What**: Engine logs and monitoring data
- **Sync Frequency**: Every **5 minutes** (300 seconds)
- **Why**: Important for debugging and real-time monitoring

## Default Sync Intervals

| Data Type | Interval | Reason |
|-----------|----------|--------|
| **Learning Data** | 5 minutes | Capture insights quickly |
| **Logs & Monitoring** | 5 minutes | Real-time debugging |
| **Models** | 30 minutes | Models change less frequently |
| **Historical Data (New)** | 5 minutes ⚡ | Newly downloaded coins synced immediately |
| **Historical Data (Full)** | 2 hours | Large files, full sync less frequently |

## Configuration

You can customize sync intervals in `src/cloud/training/config/settings.py`:

```python
class DropboxSettings(BaseModel):
    # Sync intervals in seconds
    sync_interval_learning_seconds: int = 300  # 5 minutes
    sync_interval_logs_seconds: int = 300  # 5 minutes
    sync_interval_models_seconds: int = 1800  # 30 minutes
    sync_interval_data_cache_seconds: int = 7200  # 2 hours
    
    # Enable/disable specific syncs
    sync_learning: bool = True
    sync_data_cache: bool = True
    sync_logs: bool = True
    sync_models: bool = True
    sync_monitoring: bool = True
    
    # Restore historical data on startup
    restore_data_cache_on_startup: bool = True
```

## Recommended Sync Frequencies

### For Development/Testing
- **Learning**: 1-2 minutes (capture everything quickly)
- **Logs**: 1-2 minutes (see issues immediately)
- **Models**: 5-10 minutes (frequent backups)
- **Historical Data**: 1 hour (if actively downloading)

### For Production
- **Learning**: 5-10 minutes (balanced approach)
- **Logs**: 5-10 minutes (good for monitoring)
- **Models**: 30 minutes (models change less often)
- **Historical Data**: 2-4 hours (large files, change rarely)

### For Bandwidth-Constrained Environments
- **Learning**: 10-15 minutes (still capture insights)
- **Logs**: 15-30 minutes (less frequent)
- **Models**: 1-2 hours (models are stable)
- **Historical Data**: 4-6 hours (only sync when needed)

## How It Works

1. **Multiple Sync Threads**: Each data type has its own background thread
2. **Independent Intervals**: Each thread syncs at its own frequency
3. **Automatic Restore**: Historical data is automatically restored from Dropbox on startup (if available)
4. **First Startup**: If Dropbox is empty, data is downloaded from exchange during training (normal behavior)
5. **Smart Sync**: Only syncs files that have changed (uses file hashing)

### First Startup Flow

**On first startup (no data in Dropbox):**
1. ✅ Engine starts up
2. ✅ Tries to restore from Dropbox → finds nothing (this is OK!)
3. ✅ Training begins
4. ✅ For each coin, `CandleDataLoader` checks local cache
5. ✅ If cache is empty, downloads from exchange
6. ✅ Saves to local cache (`data/candles/`)
7. ✅ **Automatic sync**: Newly downloaded coin data synced to Dropbox within **5 minutes** ⚡
8. ✅ Continuous sync also does full sync every 2 hours

**On subsequent startups (data exists in Dropbox):**
1. ✅ Engine starts up
2. ✅ Restores historical data from Dropbox
3. ✅ Training begins
4. ✅ Uses restored data (no re-download needed!)
5. ✅ Only downloads new/missing data

**Key Point**: Historical data WILL be downloaded on first startup - it happens automatically during training if the cache is empty!

## Data Storage Structure

```
Dropbox/
└── Runpodhuracan/
    ├── 2025-11-08/              # Dated folder (daily)
    │   ├── logs/                # Daily logs
    │   ├── models/              # Daily models
    │   ├── monitoring/          # Daily monitoring data
    │   └── learning/            # Daily learning insights
    └── data/
        └── candles/             # Shared historical data (reused across days)
            ├── BTC-USDT.parquet
            ├── ETH-USDT.parquet
            └── ...
```

## Benefits

1. ✅ **Historical Data Restore**: No need to re-download coin history - it's restored from Dropbox
2. ✅ **Optimized Bandwidth**: Large files sync less frequently
3. ✅ **Fast Insight Capture**: Learning data syncs frequently
4. ✅ **Automatic Backup**: Everything is backed up automatically
5. ✅ **Configurable**: Adjust intervals based on your needs

## Monitoring

The engine logs sync activity:
```
🔄 Starting Dropbox continuous sync with optimized intervals...
   📚 Learning data: every 5 min
   📝 Logs & monitoring: every 5 min
   🤖 Models: every 30 min
   📊 Historical data: every 120 min
```

Check logs for sync status:
- `learning_data_synced` - Learning data uploaded
- `data_cache_synced` - Historical data uploaded
- `models_synced` - Models uploaded
- `logs_synced` - Logs uploaded

## Troubleshooting

### Historical Data Not Restoring
- Check `restore_data_cache_on_startup` is `True`
- Verify data exists in Dropbox at `/Runpodhuracan/data/candles/`
- Check logs for `data_cache_restored` messages

### Sync Not Working
- Verify Dropbox token is valid
- Check sync intervals are > 0
- Ensure data directories exist
- Check logs for sync errors

### Too Much Bandwidth Usage
- Increase sync intervals
- Disable sync for data types you don't need
- Set `sync_data_cache_seconds` to a higher value (e.g., 6 hours)

## Summary

The engine now syncs:
- ✅ **Learning data** every 5 minutes (captures insights quickly)
- ✅ **Historical coin data** every 2 hours (large files, change rarely)
- ✅ **Models** every 30 minutes (good balance)
- ✅ **Logs** every 5 minutes (real-time monitoring)

**Historical data is automatically restored from Dropbox on startup**, so you don't need to re-download it!

All intervals are configurable - adjust based on your needs! 🚀

