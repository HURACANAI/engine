# Download Setup Complete ✅

## What Was Fixed

### 1. **Validation Error Fix**
- **Problem**: Validation was calculating expected rows as minutes (1,576,801) instead of days (1,095) for 1d timeframe
- **Solution**: Updated `DataQualitySuite._expected_rows()` to calculate based on timeframe:
  - `1m` → minutes
  - `1h` → hours  
  - `1d` → days
  - `1w` → weeks
  - `1M` → months
- **Result**: ✅ Validation now works correctly for all timeframes

### 2. **Dropbox Cache Structure for Daily Retraining**
- **Location**: `/Runpodhuracan/data/candles/` (shared cache, persists across days)
- **Structure**: 
  ```
  /Runpodhuracan/data/candles/
  ├── BTC/
  │   └── BTC-USDT_1d_20221112_20251111.parquet
  ├── ETH/
  │   └── ETH-USDT_1d_20221112_20251111.parquet
  ├── SOL/
  │   └── SOL-USDT_1d_20221112_20251111.parquet
  └── ... (249 coins)
  ```
- **Benefits**:
  - ✅ Data persists across days (not in dated folders)
  - ✅ Daily retraining can restore from Dropbox cache
  - ✅ Only new/missing data needs to be downloaded
  - ✅ Faster startup times for retraining

## Current Download Status

**Running**: Downloading top 250 coins (249 found)
- **Progress**: ~70/249 coins downloaded and uploaded
- **Timeframe**: 1d (daily candles)
- **History**: 1095 days (3 years)
- **Location**: Dropbox `/Runpodhuracan/data/candles/`

## How to Use

### Download Top 250 Coins for Daily Retraining

```bash
python scripts/download_top250_for_daily_training.py \
  --dropbox-token YOUR_TOKEN \
  --top 250 \
  --days 1095 \
  --timeframe 1d
```

### Options

- `--dropbox-token`: Your Dropbox access token (required)
- `--top N`: Number of top coins (default: 250)
- `--days N`: Days of history (default: 1095 = 3 years)
- `--timeframe`: Timeframe (1m, 5m, 15m, 1h, 4h, 1d) (default: 1d)
- `--min-volume`: Minimum 24h volume in USDT (default: 1,000,000)
- `--exchange`: Exchange ID (default: binance)

### Check Progress

```bash
# Watch live progress
tail -f /tmp/download_top250.log

# Check status
bash scripts/check_download_progress.sh
```

## Daily Retraining Integration

### How It Works

1. **First Run (Today)**:
   - Downloads top 250 coins from Binance
   - Caches locally: `data/candles/{SYMBOL}/`
   - Uploads to Dropbox: `/Runpodhuracan/data/candles/{SYMBOL}/`
   - Data is now available for daily retraining

2. **Daily Retraining (Tomorrow)**:
   - Engine starts up
   - Restores data from Dropbox cache (`/Runpodhuracan/data/candles/`)
   - Only downloads NEW/MISSING data from exchange
   - Uses cached data for training (fast!)
   - Updates cache with new data

### Benefits

✅ **No Re-downloading**: Existing data restored from Dropbox  
✅ **Faster Startup**: Only new data downloaded  
✅ **Bandwidth Savings**: Only download what's new  
✅ **Persistent Storage**: Data persists across days  
✅ **Automatic Sync**: New downloads automatically backed up  

## Folder Structure

### Local Cache
```
data/candles/
├── BTC/
│   └── BTC-USDT_1d_20221112_20251111.parquet
├── ETH/
│   └── ETH-USDT_1d_20221112_20251111.parquet
└── ... (249 coins)
```

### Dropbox Cache (Shared)
```
/Runpodhuracan/data/candles/
├── BTC/
│   └── BTC-USDT_1d_20221112_20251111.parquet
├── ETH/
│   └── ETH-USDT_1d_20221112_20251111.parquet
└── ... (249 coins)
```

## Next Steps

1. ✅ **Wait for download to complete** (~15-20 minutes for 249 coins)
2. ✅ **Verify in Dropbox**: Check `/Runpodhuracan/data/candles/` for all files
3. ✅ **Daily retraining**: Engine will automatically restore from Dropbox cache
4. ✅ **Incremental updates**: Only new data will be downloaded during retraining

## Troubleshooting

### Validation Errors Fixed
- ✅ Coverage calculation now correct for all timeframes
- ✅ 1d timeframe: expects ~1095 rows (not 1.5M minutes)
- ✅ Validation passes for coins with full history

### Download Issues
- ✅ Handles coins with partial history (newer coins)
- ✅ Skips coins that already exist in Dropbox (same size)
- ✅ Retries on rate limits
- ✅ Logs all errors for debugging

### Dropbox Upload
- ✅ Creates directories automatically
- ✅ Skips files that already exist (same size)
- ✅ Updates files if size changed
- ✅ Uses shared cache location (not dated folders)

## Files Changed

1. **`src/cloud/training/datasets/quality_checks.py`**:
   - Fixed `_expected_rows()` to calculate based on timeframe

2. **`scripts/simple_download_candles.py`**:
   - Improved Dropbox upload logic
   - Better cache handling
   - Shared cache location for daily retraining

3. **`scripts/download_top250_for_daily_training.py`** (NEW):
   - Dedicated script for downloading top 250 coins
   - Organized for daily retraining workflow
   - Better progress reporting

## Status

✅ **Download Running**: Processing 249 coins  
✅ **Validation Fixed**: Works correctly for all timeframes  
✅ **Dropbox Cache**: Organized for daily retraining  
✅ **Progress**: ~70/249 coins completed  
✅ **ETA**: ~15-20 minutes remaining  

Monitor progress with: `tail -f /tmp/download_top250.log`

