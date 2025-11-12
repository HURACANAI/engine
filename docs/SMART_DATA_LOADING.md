# Smart Data Loading & Caching System

## Overview

The training bot now features an **intelligent data loading system** that ensures you always train on the **most recent market data** while efficiently reusing cached historical data.

## Key Features

### 1. **Always Fetch Latest Data**
Every time the bot runs, it automatically:
- Updates the end time to **NOW** (current UTC time)
- Downloads any new candles since last run
- Ensures you're training on the freshest market data

### 2. **Smart Cache Reuse**
The system intelligently reuses existing cache files:
- Finds existing cache for the symbol/timeframe
- Checks date range coverage
- Only downloads missing gaps (historical or recent)
- Merges new data with cached data
- Updates the same cache file (no duplicate files)

### 3. **Automatic Gap Filling**
If cached data has gaps, the system automatically:
- Detects missing historical data (before cache start)
- Detects missing recent data (after cache end)
- Downloads both gaps in one operation
- Merges everything into a single, complete dataset

### 4. **Dropbox Integration**
After updating local cache, the system:
- Automatically uploads updated cache to Dropbox
- Keeps your data safe in the cloud
- Allows training continuation from any machine

## How It Works

### Example Scenario

**First Run (Day 1)**:
```
Request: 90 days of SOL/USDT data
Cache: None
Action: Download 2,160 candles (90 days × 24 hours)
Result: data/candles/SOL/SOL-USDT_1h_20250814_20251112.parquet
```

**Second Run (Day 2)**:
```
Request: 90 days of SOL/USDT data
Cache: Has data from 2025-08-14 to 2025-11-12
Current Time: 2025-11-13 08:00 UTC

Detection:
- Cache END: 2025-11-12 23:00
- Requested END: 2025-11-13 08:00 (NOW)
- Gap: 9 hours of new data

Action: Download 9 new candles (2025-11-12 23:00 to NOW)
Merge: 2,160 old + 9 new = 2,169 total candles
Update: Same cache file (SOL-USDT_1h_20250814_20251112.parquet)
Upload: Updated file to Dropbox
```

**Third Run (Day 30)**:
```
Request: 90 days of SOL/USDT data
Cache: Has data from 2025-08-14 to 2025-11-13
Current Time: 2025-12-12 14:00 UTC

Detection:
- Cache START: 2025-08-14
- Requested START: 2025-09-13 (90 days before NOW)
- Gap: 30 days of older data no longer needed

- Cache END: 2025-11-13
- Requested END: 2025-12-12 14:00 (NOW)
- Gap: 29 days of new data

Action: Download 29 days × 24 hours = 696 new candles
Merge: Keep only last 90 days of data
Update: Same cache file
Upload: Updated file to Dropbox
```

## Cache File Strategy

### Old Behavior (Before Enhancement)
- Created new cache file for every date range
- Result: `SOL-USDT_1h_20250101_20250401.parquet`, `SOL-USDT_1h_20250201_20250501.parquet`, etc.
- Problem: Multiple files, wasted space, confusion

### New Behavior (After Enhancement)
- **Reuses existing cache file** for same symbol/timeframe
- Finds: `SOL-USDT_1h_*.parquet` (any dates)
- Uses most recently modified file
- Updates same file with new data
- Result: **Single, continuously updated cache file per symbol/timeframe**

## Data Freshness Guarantees

### Training Always Uses Latest Data
```python
# Before (potentially stale data):
query = CandleQuery(
    symbol="SOL/USDT",
    start_at=90_days_ago,
    end_at=yesterday,  # ❌ Stale by 1 day
)

# After (always fresh):
query = CandleQuery(
    symbol="SOL/USDT",
    start_at=90_days_ago,
    end_at=datetime.now(tz=timezone.utc),  # ✅ Always NOW
)
```

### Automatic NOW Update
The system automatically updates `end_at` to NOW:
```
⏰ [SOL/USDT] Updated end time to NOW: 2025-11-12 16:49:23 UTC
```

This ensures every training run includes the most recent market movements.

## Visual Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  Training Bot Starts                                        │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  Request: 90 days of SOL/USDT, END=NOW                     │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
        ┌────────────────┐
        │ Cache exists?  │
        └───┬────────┬───┘
            │        │
          YES       NO
            │        │
            ▼        ▼
    ┌────────────┐  ┌─────────────────┐
    │ Read cache │  │ Download all    │
    │ Check dates│  │ 90 days from    │
    └─────┬──────┘  │ exchange        │
          │         └────────┬────────┘
          ▼                  │
    ┌────────────┐           │
    │ Gap found? │           │
    └───┬────┬───┘           │
        │    │               │
       YES  NO               │
        │    │               │
        ▼    │               │
┌──────────────┐             │
│ Download gap │             │
│ (historical  │             │
│  or recent)  │             │
└─────┬────────┘             │
      │                      │
      ▼                      │
┌─────────────┐              │
│ Merge data  │              │
│ Remove dups │              │
└─────┬───────┘              │
      │                      │
      ├──────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Update cache file   │
│ (same filename)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Upload to Dropbox   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Return fresh data   │
│ for training        │
└─────────────────────┘
```

## Log Output Examples

### Cache Hit with Top-Up Needed
```
📦 [SOL/USDT] Checking local cache: SOL-USDT_1h_20250814_20251112.parquet
♻️  [SOL/USDT] Found existing cache file: SOL-USDT_1h_20250814_20251112.parquet
✅ [SOL/USDT] Found cached data: 2025-08-14 to 2025-11-12 (2,160 rows)
📅 [SOL/USDT] Requested range: 2025-08-14 to 2025-11-13
⬇️  [SOL/USDT] Need to download newer data: 2025-11-12 to 2025-11-13
📊 [SOL/USDT] Fetching 0 days and 16 hours of new data
🔄 [SOL/USDT] Top-up needed: Downloading missing data from binance
  ✅ [SOL/USDT] Download complete: 1 batches, 16 rows
✅ [SOL/USDT] Downloaded 16 new rows from binance
✅ [SOL/USDT] Merged data: 2,176 total rows (2,160 cached + 16 new)
💾 [SOL/USDT] Updated cache: SOL-USDT_1h_20250814_20251112.parquet
☁️  [SOL/USDT] Uploaded to Dropbox: SOL-USDT_1h_20250814_20251112.parquet
```

### Cache Miss - Full Download
```
📦 [SOL/USDT] Checking local cache: SOL-USDT_1h_20250814_20251112.parquet
⬇️  [SOL/USDT] Downloading from binance: 2025-08-14 to 2025-11-12
  ⬇️  [SOL/USDT] Downloading... 10 batches, 10,000 rows (46%)
  ✅ [SOL/USDT] Download complete: 22 batches, 2,160 rows
✅ [SOL/USDT] Downloaded 2,160 rows from binance
💾 [SOL/USDT] Saved to cache: SOL-USDT_1h_20250814_20251112.parquet
☁️  [SOL/USDT] Uploaded to Dropbox: SOL-USDT_1h_20250814_20251112.parquet
```

### Cache Complete - No Download Needed
```
📦 [SOL/USDT] Checking local cache: SOL-USDT_1h_20250814_20251112.parquet
♻️  [SOL/USDT] Found existing cache file: SOL-USDT_1h_20250814_20251112.parquet
✅ [SOL/USDT] Found cached data: 2025-08-14 to 2025-11-12 (2,160 rows)
📅 [SOL/USDT] Requested range: 2025-08-14 to 2025-11-12
✅ [SOL/USDT] Cache has all requested data - using cached data
```

## Benefits

### 1. **Data Freshness**
- Always includes latest candles up to NOW
- Never trains on stale data
- Captures most recent market movements

### 2. **Efficiency**
- Reuses existing cache files
- Only downloads missing gaps
- Minimizes API calls to exchange
- Faster training startup

### 3. **Reliability**
- Automatic gap detection
- Handles timestamp corruption
- Validates data quality
- Fallback to full re-download if needed

### 4. **Cloud Backup**
- Automatic Dropbox sync
- Safe from local disk failures
- Train from any machine
- Recover easily

### 5. **Storage Optimization**
- Single cache file per symbol/timeframe
- No duplicate data
- Organized by coin folders
- Easy to manage

## Implementation Details

### Key Changes in `data_loader.py`

#### 1. Always Update to NOW (Lines 91-102)
```python
# CRITICAL: Always update query.end_at to NOW
now = datetime.now(tz=timezone.utc)
if query.end_at < now:
    query = CandleQuery(
        symbol=query.symbol,
        timeframe=query.timeframe,
        start_at=query.start_at,
        end_at=now,  # Always fetch up to NOW
    )
    print(f"⏰ [{query.symbol}] Updated end time to NOW: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
```

#### 2. Reuse Existing Cache Files (Lines 435-445)
```python
# Check if there's already a cache file for this symbol/timeframe
existing_pattern = f"{symbol_safe}_{query.timeframe}_*.parquet"
existing_files = list(coin_dir.glob(existing_pattern))

if existing_files:
    # Use the most recently modified cache file
    most_recent = max(existing_files, key=lambda p: p.stat().st_mtime)
    print(f"♻️  [{query.symbol}] Found existing cache file: {most_recent.name}")
    return most_recent
```

#### 3. Smart Gap Detection (Lines 150-161)
```python
if cache_end < requested_end:
    needs_topup = True
    topup_start = cache_end + timedelta(minutes=1)
    print(f"⬇️  [{query.symbol}] Need to download newer data: {topup_start.date()} to {requested_end.date()}")
    # Show how many hours/days of new data we're fetching
    time_diff = requested_end - cache_end
    if time_diff.days > 0:
        print(f"📊 [{query.symbol}] Fetching {time_diff.days} days and {time_diff.seconds // 3600} hours of new data")
```

## Configuration

No configuration needed! The system works automatically.

However, you can customize:

### Cache Directory
```python
data_loader = CandleDataLoader(
    exchange_client=exchange,
    cache_dir=Path("/custom/cache/location"),  # Default: ./data/candles
)
```

### Disable Cache
```python
data = data_loader.load(query, use_cache=False)  # Always download fresh
```

### Lookback Period
```python
query = CandleQuery(
    symbol="SOL/USDT",
    timeframe="1h",
    start_at=datetime.now(tz=timezone.utc) - timedelta(days=180),  # 180 days instead of 90
)
```

## Monitoring

### Check Cache Status
```bash
# List cache files for SOL
ls -lh data/candles/SOL/

# Output:
# -rw-r--r--  1 user  staff   1.2M Nov 12 16:49 SOL-USDT_1h_20250814_20251112.parquet
```

### Inspect Cache Contents
```python
import polars as pl

# Load cache file
cache = pl.read_parquet("data/candles/SOL/SOL-USDT_1h_20250814_20251112.parquet")

# Check date range
print(f"Start: {cache['ts'].min()}")
print(f"End: {cache['ts'].max()}")
print(f"Total candles: {cache.height}")

# Output:
# Start: 2025-08-14 00:00:00
# End: 2025-11-12 23:00:00
# Total candles: 2160
```

### Monitor Dropbox Uploads
Watch training logs for upload confirmation:
```
☁️  [SOL/USDT] Uploaded to Dropbox: SOL-USDT_1h_20250814_20251112.parquet
```

## Troubleshooting

### Cache Not Updating
- Check file permissions on cache directory
- Verify Dropbox credentials are valid
- Check available disk space

### Missing Recent Data
- Ensure system clock is accurate (UTC)
- Check exchange API connectivity
- Verify symbol and timeframe are correct

### Duplicate Cache Files
Old files from before enhancement may exist. Safe to delete:
```bash
# Keep only the most recent cache file per symbol/timeframe
cd data/candles/SOL
ls -t SOL-USDT_1h_*.parquet | tail -n +2 | xargs rm
```

### Cache Corruption
System automatically detects and fixes:
```
⚠️  [SOL/USDT] Cache has corrupted timestamps (1970 dates) - re-downloading
```

## Future Enhancements

Planned improvements:
- [ ] Multi-timeframe synchronization (align 1h and 4h caches)
- [ ] Cross-symbol gap detection (ensure BTC, ETH, SOL have same date ranges)
- [ ] Automatic cache pruning (remove very old data beyond lookback window)
- [ ] Cache compression statistics (show space saved)
- [ ] Cache hit rate metrics (track download vs cache usage)

---

**Last Updated**: 2025-11-12
**Version**: 2.0 (Enhanced with Smart Top-Up)
