# Local Candle Download & Upload Guide

## Overview

Download historical candle data locally and upload to Dropbox, so the RunPod engine can restore it instead of downloading from the exchange.

**Benefits:**
- ✅ Avoid rate limits on RunPod
- ✅ Faster startup (download from Dropbox is faster than exchange API)
- ✅ Centralized data management
- ✅ Can download on your local machine with better internet/API access
- ✅ No need to wait for exchange downloads on every training run

## How It Works

### 1. Local Download & Upload

1. **Run script locally** to download candles:
   ```bash
   python scripts/download_and_upload_candles.py --symbols BTC/USDT ETH/USDT --days 150
   ```

2. **Script downloads** data from exchange to local `data/candles/` directory

3. **Script uploads** data to Dropbox (`/Runpodhuracan/data/candles/`)

### 2. RunPod Engine Restore

1. **Engine starts up** on RunPod
2. **Checks Dropbox** for existing data
3. **Restores** all historical data from Dropbox to local cache
4. **Skips exchange download** if data exists in cache
5. **Only downloads new/missing data** from exchange

## Usage

### Download Specific Symbols

```bash
# Download BTC and ETH with 150 days of data
python scripts/download_and_upload_candles.py --symbols BTC/USDT ETH/USDT --days 150

# Download with custom timeframe
python scripts/download_and_upload_candles.py --symbols BTC/USDT --days 365 --timeframe 1h

# Download with custom exchange
python scripts/download_and_upload_candles.py --symbols BTC/USDT --days 150 --exchange binance
```

### Download All Symbols from Universe

```bash
# Download all symbols from universe (uses universe selector from settings)
python scripts/download_and_upload_candles.py --all-symbols --days 150
```

### Custom Dropbox Token

```bash
# Use custom Dropbox token (if different from settings)
python scripts/download_and_upload_candles.py --symbols BTC/USDT --days 150 --dropbox-token YOUR_TOKEN
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--symbols` | List of symbols to download (e.g., BTC/USDT ETH/USDT) | Required (or use --all-symbols) |
| `--all-symbols` | Download all symbols from universe | False |
| `--days` | Number of days of historical data | 150 |
| `--exchange` | Exchange ID (binance, coinbasepro, etc.) | binance |
| `--timeframe` | Timeframe for candles (1m, 5m, 1h, 1d, etc.) | 1m |
| `--dropbox-token` | Dropbox access token | From settings or environment |
| `--app-folder` | Dropbox app folder name | Runpodhuracan |

## Examples

### Example 1: Download Top Coins

```bash
python scripts/download_and_upload_candles.py \
  --symbols BTC/USDT ETH/USDT SOL/USDT BNB/USDT \
  --days 150 \
  --timeframe 1m
```

### Example 2: Download All Universe Symbols

```bash
python scripts/download_and_upload_candles.py \
  --all-symbols \
  --days 150 \
  --timeframe 1m
```

### Example 3: Download with Different Timeframe

```bash
# Download hourly candles (smaller files, faster)
python scripts/download_and_upload_candles.py \
  --symbols BTC/USDT ETH/USDT \
  --days 365 \
  --timeframe 1h
```

## Dropbox Structure

Data is uploaded to:
```
Dropbox/
└── Runpodhuracan/
    └── data/
        └── candles/              # Shared location (persists across days)
            ├── BTC-USDT_1m_20250611_20251108.parquet
            ├── ETH-USDT_1m_20250611_20251108.parquet
            └── ...
```

## RunPod Engine Behavior

### First Run (No Data in Dropbox)

1. ✅ Engine starts
2. ✅ Checks Dropbox → finds nothing
3. ✅ Downloads data from exchange
4. ✅ Saves to local cache
5. ✅ Uploads to Dropbox

### Subsequent Runs (Data in Dropbox)

1. ✅ Engine starts
2. ✅ **Restores data from Dropbox** (fast!)
3. ✅ Checks local cache → finds restored data
4. ✅ **Skips exchange download** (no API calls needed!)
5. ✅ Only downloads new/missing data

### After Local Upload

1. ✅ You upload data locally to Dropbox
2. ✅ Engine restores from Dropbox on startup
3. ✅ **No exchange downloads needed** (unless new data is needed)
4. ✅ Fast startup, no rate limits!

## File Naming

Files are named using this format:
```
{symbol}_{timeframe}_{start_date}_{end_date}.parquet
```

Example:
- `BTC-USDT_1m_20250611_20251108.parquet`
- `ETH-USDT_1h_20240611_20251108.parquet`

Where:
- `symbol`: Symbol with `/` replaced with `-` (BTC/USDT → BTC-USDT)
- `timeframe`: Timeframe (1m, 5m, 1h, 1d, etc.)
- `start_date`: Start date in YYYYMMDD format
- `end_date`: End date in YYYYMMDD format

## Requirements

1. **Dropbox Access Token**: Set `DROPBOX_ACCESS_TOKEN` environment variable or configure in settings
2. **Exchange Credentials**: Configure in settings (if needed for authenticated API calls)
3. **Python Dependencies**: All engine dependencies must be installed

## Troubleshooting

### Dropbox Upload Fails

- Check Dropbox token is valid
- Check token has write permissions
- Check network connectivity

### Download Fails

- Check exchange API is accessible
- Check exchange credentials (if needed)
- Check symbol format is correct (e.g., BTC/USDT not BTCUSDT)

### Restore Doesn't Work on RunPod

- Check Dropbox token is configured on RunPod
- Check `restore_data_cache_on_startup` is enabled in settings
- Check data was uploaded to correct location (`/Runpodhuracan/data/candles/`)

## Workflow

### Recommended Workflow

1. **Download locally** once (or periodically):
   ```bash
   python scripts/download_and_upload_candles.py --all-symbols --days 150
   ```

2. **Upload to Dropbox** (automatic in script)

3. **RunPod engine** automatically restores on startup

4. **Update data** periodically (weekly/monthly):
   ```bash
   # Update with latest data
   python scripts/download_and_upload_candles.py --all-symbols --days 150
   ```

## Benefits Summary

✅ **No Rate Limits**: Download locally, avoid exchange API limits on RunPod  
✅ **Faster Startup**: Restore from Dropbox is faster than exchange API  
✅ **Centralized Data**: All data in one place (Dropbox)  
✅ **Flexible**: Download on your machine with better internet/API access  
✅ **Automated**: Engine automatically uses Dropbox data if available  





