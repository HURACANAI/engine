# Historical Data Persistence Fix

## Problem

Historical coin data was being stored in **dated folders** (`/Runpodhuracan/2025-11-08/data/candles/`), which meant:
- Data was separated by day
- On restart, the engine couldn't find data from previous days
- Data had to be re-downloaded on every startup
- Wasted bandwidth and time downloading the same data repeatedly

## Solution

Historical coin data is now stored in a **SHARED location** (`/Runpodhuracan/data/candles/`) that persists across days:

### Changes Made

1. **Default Storage Location Changed**
   - `upload_data_cache()` now defaults to `use_dated_folder=False`
   - Data goes to `/Runpodhuracan/data/candles/` (shared) instead of dated folders
   - This ensures data persists across days

2. **Restore Logic Updated**
   - `restore_data_cache()` now always uses shared location by default
   - Recursively restores ALL files from Dropbox on startup
   - Only downloads new/missing data from exchange

3. **Continuous Sync Updated**
   - Full sync uses shared location
   - Quick sync (recent files) also uses shared location
   - All new downloads automatically sync to shared location

4. **Startup Flow**
   - On startup, engine checks Dropbox for existing data
   - Restores ALL historical data from shared location
   - Only downloads new/missing data from exchange
   - No more re-downloading existing data!

## How It Works

### First Startup (No Data in Dropbox)

1. ✅ Engine starts up
2. ✅ Checks Dropbox for data → finds nothing (OK!)
3. ✅ Training begins
4. ✅ Downloads data from exchange
5. ✅ Saves to local cache (`data/candles/`)
6. ✅ Automatically syncs to Dropbox (`/Runpodhuracan/data/candles/`)
7. ✅ Data is now backed up and will be restored on next startup

### Subsequent Startups (Data Exists in Dropbox)

1. ✅ Engine starts up
2. ✅ **Restores ALL historical data from Dropbox** (`/Runpodhuracan/data/candles/`)
3. ✅ Training begins
4. ✅ Checks local cache first → finds restored data
5. ✅ Only downloads NEW/MISSING data from exchange
6. ✅ New data is automatically synced to Dropbox

## Benefits

✅ **No Re-downloading**: Existing data is restored from Dropbox  
✅ **Faster Startup**: No waiting for data downloads (only new data)  
✅ **Bandwidth Savings**: Only download what's new  
✅ **Persistent Storage**: Data persists across days  
✅ **Automatic Sync**: New downloads automatically backed up  
✅ **Incremental Updates**: Only download missing/new data  

## Dropbox Structure

```
Dropbox/
└── Runpodhuracan/
    ├── 2025-11-08/              # Dated folder (daily logs, models, etc.)
    │   ├── logs/
    │   ├── models/
    │   └── monitoring/
    └── data/                     # SHARED location (persists across days)
        └── candles/              # Historical coin data
            ├── BTC-USDT.parquet
            ├── ETH-USDT.parquet
            └── ...
```

## Configuration

In `settings.py`:
- `restore_data_cache_on_startup: bool = True` - Enable restore on startup
- `sync_data_cache: bool = True` - Enable automatic sync
- `sync_interval_data_cache_seconds: int = 7200` - Full sync every 2 hours

## Files Changed

- `src/cloud/training/integrations/dropbox_sync.py`:
  - `upload_data_cache()` - Changed default to use shared location
  - `restore_data_cache()` - Always uses shared location, recursive restore
  - Continuous sync loop - Uses shared location

- `src/cloud/training/pipelines/daily_retrain.py`:
  - Updated restore call to use shared location
  - Updated initial sync to use shared location

## Testing

After deployment:

1. **First Run**: 
   - Data should be downloaded from exchange
   - Data should sync to `/Runpodhuracan/data/candles/`
   - Check Dropbox to verify files are there

2. **Second Run**:
   - Engine should restore data from Dropbox
   - Should see: "✅ Restored X historical data files from Dropbox"
   - Should NOT re-download existing data
   - Only new/missing data should be downloaded

3. **Verify**:
   - Check Dropbox: `/Runpodhuracan/data/candles/` should have all coin data
   - Check logs: Should see restore messages on startup
   - Check training: Should skip downloading existing data

## Migration Note

If you have existing data in dated folders (`/Runpodhuracan/2025-11-08/data/candles/`), you can:
1. Manually move it to shared location in Dropbox
2. Or let the engine re-download (one-time only)
3. Or run a migration script to move existing data

The engine will automatically use the shared location going forward.


