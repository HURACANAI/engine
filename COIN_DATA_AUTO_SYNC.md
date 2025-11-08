# Coin Data Automatic Sync to Dropbox

## Overview

**Every time coin data is downloaded and updated, it is automatically synced to Dropbox within 5 minutes!** ⚡

## How It Works

### 1. **Download Process**
When the engine downloads coin data:
1. Data is downloaded from the exchange
2. Data is saved to local cache (`data/candles/`)
3. File modification time is updated to current time
4. Log message: `"Coin data saved - will be synced to Dropbox within 5 minutes"`

### 2. **Automatic Sync Detection**
The Dropbox sync system:
1. Checks for new files every **5 minutes**
2. Identifies files modified in the last **10 minutes**
3. Immediately syncs those files to Dropbox
4. Logs: `"Recently downloaded coin data synced immediately"`

### 3. **Sync Behavior**

| Scenario | Sync Behavior |
|----------|---------------|
| **Newly downloaded coin** | Synced within **5 minutes** ⚡ |
| **Updated coin data** | Synced within **5 minutes** ⚡ |
| **All coin data** | Full sync every **2 hours** |

## Benefits

✅ **No Data Loss**: Newly downloaded coins are synced immediately
✅ **Automatic**: No manual intervention needed
✅ **Fast**: Synced within 5 minutes of download
✅ **Efficient**: Only newly downloaded files are synced immediately (not all files)
✅ **Reliable**: Full sync every 2 hours ensures nothing is missed

## Technical Details

### Sync Mechanism

The sync system uses a two-tier approach:

1. **Quick Check (Every 5 minutes)**:
   - Scans for files modified in the last 10 minutes
   - Immediately syncs those files to Dropbox
   - Logs each file synced

2. **Full Sync (Every 2 hours)**:
   - Syncs all files in `data/candles/`
   - Ensures complete backup
   - Catches any files that might have been missed

### File Detection

Files are detected as "recent" if:
- File modification time is within the last 10 minutes
- File is a `.parquet` file
- File is in the `data/candles/` directory

### Dropbox Location

All coin data is synced to:
```
/Runpodhuracan/YYYY-MM-DD/data/candles/
```

Where `YYYY-MM-DD` is the date of the engine run.

## Log Messages

When coin data is downloaded:
```
coin_data_cached: symbol=BTC/USDT, cache_path=data/candles/BTC-USDT_1m_20250101_20250108.parquet, rows=10080, message="Coin data saved - will be synced to Dropbox within 5 minutes"
```

When coin data is synced:
```
coin_data_synced_immediately: file=data/candles/BTC-USDT_1m_20250101_20250108.parquet, remote_path=/Runpodhuracan/2025-01-08/data/candles/BTC-USDT_1m_20250101_20250108.parquet, message="Newly downloaded coin data synced to Dropbox"
```

## Configuration

The sync intervals are configured in `src/cloud/training/config/settings.py`:

```python
class DropboxSettings(BaseModel):
    # Historical data cache: sync less frequently (every 2 hours) - large files, don't change often
    sync_interval_data_cache_seconds: int = 7200  # 2 hours (full sync)
    # Quick check for new files: every 5 minutes (automatic)
```

**Note**: The 5-minute quick check interval is hardcoded in the sync loop for optimal performance. The 2-hour interval is configurable.

## Summary

✅ **Automatic**: Every time coins are downloaded, they're automatically synced to Dropbox
✅ **Fast**: Synced within 5 minutes of download
✅ **Reliable**: Full sync every 2 hours ensures complete backup
✅ **Efficient**: Only newly downloaded files are synced immediately
✅ **No Data Loss**: All downloaded coin data is backed up to Dropbox

**You never have to worry about losing downloaded coin data - it's automatically backed up to Dropbox!** 🚀

