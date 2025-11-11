# Robust Downloader for Top 250 Coins

## Overview

The robust downloader (`robust_download_top250.py`) is designed to reliably download all 250 coins with the following features:

✅ **Resume Capability**: Saves progress and can resume from where it left off  
✅ **Retry Logic**: Automatically retries failed downloads (up to 3 attempts)  
✅ **Rate Limit Handling**: Exponential backoff for rate limits (up to 5 minutes)  
✅ **Error Recovery**: Continues even if some coins fail  
✅ **Progress Tracking**: Saves progress to `data/download_progress.json`  
✅ **Skip Completed**: Skips coins that are already downloaded and uploaded  
✅ **Better Error Handling**: Handles network errors, timeouts, and API errors gracefully  

## Usage

### Basic Usage

```bash
python scripts/robust_download_top250.py \
  --dropbox-token YOUR_TOKEN \
  --top 250 \
  --days 1095 \
  --timeframe 1d
```

### Resume from Previous Run

```bash
python scripts/robust_download_top250.py \
  --dropbox-token YOUR_TOKEN \
  --top 250 \
  --days 1095 \
  --timeframe 1d \
  --resume
```

### Options

- `--dropbox-token`: Dropbox access token (required)
- `--top N`: Number of top coins (default: 250)
- `--days N`: Days of history (default: 1095 = 3 years)
- `--timeframe`: Timeframe (1m, 5m, 15m, 1h, 4h, 1d) (default: 1d)
- `--min-volume`: Minimum 24h volume in USDT (default: 1,000,000)
- `--delay`: Delay between coins in seconds (default: 0.5)
- `--resume`: Resume from previous progress (skips completed coins)

## Features

### 1. Progress Tracking

Progress is saved to `data/download_progress.json`:

```json
{
  "completed": ["BTC/USDT", "ETH/USDT", ...],
  "failed": ["SYMBOL1/USDT", ...],
  "started_at": "2025-11-11T19:12:00Z",
  "last_updated": "2025-11-11T19:15:00Z"
}
```

### 2. Resume Capability

If the download stops or is interrupted:
1. Run again with `--resume` flag
2. Script will skip already completed coins
3. Only downloads remaining coins
4. Retries failed coins

### 3. Retry Logic

- **Max Retries**: 3 attempts per coin
- **Rate Limits**: Exponential backoff (up to 5 minutes)
- **Network Errors**: Exponential backoff (up to 1 minute)
- **Other Errors**: Exponential backoff (up to 30 seconds)

### 4. Error Handling

The script handles:
- ✅ Rate limits (429 errors)
- ✅ Network timeouts
- ✅ Connection errors
- ✅ Invalid symbols
- ✅ API errors
- ✅ Dropbox upload errors

### 5. Skip Completed Coins

- Checks if coin is already in Dropbox
- Verifies file size matches
- Skips if already uploaded (saves time)
- Marks as completed in progress file

## Monitoring

### Check Progress

```bash
# Monitor download
bash scripts/monitor_download.sh

# Watch live log
tail -f /tmp/robust_download.log

# Check progress file
cat data/download_progress.json
```

### Progress File

The progress file (`data/download_progress.json`) contains:
- `completed`: List of successfully downloaded coins
- `failed`: List of failed coins (with errors)
- `started_at`: When download started
- `last_updated`: Last update timestamp

## Troubleshooting

### Download Keeps Stopping

1. **Check if process is running**:
   ```bash
   ps aux | grep robust_download_top250.py
   ```

2. **Check for errors**:
   ```bash
   tail -100 /tmp/robust_download.log | grep -i error
   ```

3. **Resume from where it left off**:
   ```bash
   python scripts/robust_download_top250.py \
     --dropbox-token YOUR_TOKEN \
     --resume
   ```

### Rate Limits

If you hit rate limits:
1. The script will automatically wait (exponential backoff)
2. Wait time increases with each retry (up to 5 minutes)
3. Script continues after rate limit clears
4. No manual intervention needed

### Failed Coins

If some coins fail:
1. Check `data/download_progress.json` for failed coins
2. Run again with `--resume` to retry failed coins
3. Failed coins will be retried with backoff
4. Check logs for specific error messages

### Network Issues

If network issues occur:
1. Script will retry with exponential backoff
2. Continues with next coin if retries fail
3. Failed coins are marked in progress file
4. Can resume later to retry failed coins

## Benefits Over Simple Downloader

| Feature | Simple Downloader | Robust Downloader |
|---------|------------------|-------------------|
| Resume Capability | ❌ | ✅ |
| Progress Tracking | ❌ | ✅ |
| Retry Logic | Basic | Advanced (3 retries) |
| Rate Limit Handling | Basic (10s wait) | Exponential backoff (up to 5min) |
| Error Recovery | Stops on error | Continues on error |
| Skip Completed | ❌ | ✅ |
| Progress File | ❌ | ✅ |

## Example Run

```bash
# Start download
python scripts/robust_download_top250.py \
  --dropbox-token YOUR_TOKEN \
  --top 250 \
  --days 1095 \
  --timeframe 1d \
  --resume

# Monitor progress
bash scripts/monitor_download.sh

# If interrupted, resume
python scripts/robust_download_top250.py \
  --dropbox-token YOUR_TOKEN \
  --resume
```

## Output

### Dropbox Structure

```
/Runpodhuracan/data/candles/  (shared cache)
├── BTC/
│   └── BTC-USDT_1d_20221112_20251111.parquet
├── ETH/
│   └── ETH-USDT_1d_20221112_20251111.parquet
└── ... (249 coins)
```

### Local Cache

```
data/candles/
├── BTC/
│   └── BTC-USDT_1d_20221112_20251111.parquet
├── ETH/
│   └── ETH-USDT_1d_20221112_20251111.parquet
└── ... (249 coins)
```

### Progress File

```
data/download_progress.json
```

## Best Practices

1. **Use `--resume` flag**: Always use `--resume` to skip completed coins
2. **Monitor progress**: Check progress regularly with `monitor_download.sh`
3. **Check logs**: Review logs if downloads fail
4. **Retry failed coins**: Run again with `--resume` to retry failed coins
5. **Verify Dropbox**: Check Dropbox to verify all coins are uploaded

## Next Steps

After download completes:
1. ✅ Verify all coins in Dropbox
2. ✅ Check progress file for any failed coins
3. ✅ Retry failed coins if needed
4. ✅ Data is ready for daily retraining

