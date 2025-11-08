# Automated Candle Sync to Dropbox

## Overview

Simple scripts to automatically download all required candles and upload them to Dropbox.

## Quick Start

### Option 1: Simple Bash Script (Recommended)

```bash
# Download all symbols with 150 days of data
./scripts/sync_all_candles_to_dropbox.sh

# Or specify days
./scripts/sync_all_candles_to_dropbox.sh 365
```

### Option 2: Python Script with More Options

```bash
# Download all symbols
python scripts/auto_sync_candles.py --days 150

# Update only (skip existing files)
python scripts/auto_sync_candles.py --days 150 --update-only

# Custom timeframe
python scripts/auto_sync_candles.py --days 150 --timeframe 1h
```

### Option 3: Direct Python Script

```bash
# Download all symbols from universe
python scripts/download_and_upload_candles.py --all-symbols --days 150

# Download specific symbols
python scripts/download_and_upload_candles.py --symbols BTC/USDT ETH/USDT --days 150
```

## What It Does

1. **Loads symbols from universe** - Gets all symbols configured in your universe settings
2. **Downloads candle data** - Downloads historical data from exchange (default: Binance)
3. **Uploads to Dropbox** - Automatically uploads to `/Runpodhuracan/data/candles/`
4. **Skips existing files** - Won't re-download if file already exists (optional)

## Automation

### Cron Job (Run Daily)

Add to your crontab to run daily:

```bash
# Edit crontab
crontab -e

# Add this line to run daily at 2 AM
0 2 * * * cd /path/to/engine && ./scripts/sync_all_candles_to_dropbox.sh 150 >> /var/log/candle_sync.log 2>&1
```

### Systemd Timer (Linux)

Create a systemd service and timer to run periodically:

```ini
# /etc/systemd/system/candle-sync.service
[Unit]
Description=Sync candles to Dropbox
After=network.target

[Service]
Type=oneshot
User=your_user
WorkingDirectory=/path/to/engine
ExecStart=/path/to/engine/scripts/sync_all_candles_to_dropbox.sh 150
Environment="HURACAN_ENV=local"
```

```ini
# /etc/systemd/system/candle-sync.timer
[Unit]
Description=Run candle sync daily
Requires=candle-sync.service

[Timer]
OnCalendar=daily
OnCalendar=02:00
Persistent=true

[Install]
WantedBy=timers.target
```

Then enable and start:
```bash
sudo systemctl enable candle-sync.timer
sudo systemctl start candle-sync.timer
```

## Configuration

### Environment Variables

```bash
# Set environment (local, prod, runpod)
export HURACAN_ENV=local

# Set Dropbox token (if not in settings)
export DROPBOX_ACCESS_TOKEN=your_token_here
```

### Settings

The script uses settings from `config/base.yaml` and `config/local.yaml` (or your environment).

Key settings:
- `dropbox.access_token` - Dropbox access token
- `dropbox.app_folder` - Dropbox app folder name (default: Runpodhuracan)
- `exchange.primary` - Primary exchange (default: binance)
- `universe.target_size` - Number of symbols to download

## Output

The script provides:
- Progress for each symbol
- Download status (success/failure)
- Upload status (success/failure)
- Summary at the end

Example output:
```
📊 Downloading historical candle data
   Symbols: BTC/USDT, ETH/USDT, ...
   Exchange: binance
   Timeframe: 1m
   Days: 150

📥 Downloading BTC/USDT...
   ✅ Downloaded 216,000 rows (5.2 MB)
   📤 Uploading to Dropbox...
   ✅ Uploaded to Dropbox: /Runpodhuracan/data/candles/BTC-USDT_1m_20250611_20251108.parquet

============================================================
📊 Summary
============================================================
   ✅ Downloaded: 20/20
   📤 Uploaded: 20/20
   ❌ Failed: 0/20
============================================================
```

## Dropbox Structure

Files are uploaded to:
```
Dropbox/
└── Runpodhuracan/
    └── data/
        └── candles/
            ├── BTC-USDT_1m_20250611_20251108.parquet
            ├── ETH-USDT_1m_20250611_20251108.parquet
            └── ...
```

## RunPod Integration

Once candles are uploaded to Dropbox:

1. **RunPod engine starts**
2. **Restores data from Dropbox** (automatic on startup)
3. **Uses cached data** (no exchange download needed)
4. **Only downloads new data** if needed

## Troubleshooting

### Missing Dependencies

```bash
# Install dependencies
pip install -r scripts/install_minimal_requirements.txt
```

### Config Files Missing

Make sure `config/base.yaml` and `config/local.yaml` exist.

### Dropbox Token Issues

```bash
# Set token
export DROPBOX_ACCESS_TOKEN=your_token

# Or pass directly
python scripts/auto_sync_candles.py --days 150 --dropbox-token your_token
```

### Download Failures

- Check internet connection
- Check exchange API is accessible
- Check symbol names are correct
- Check exchange credentials (if needed)

## Best Practices

1. **Run daily** - Keep data up to date
2. **Use update-only mode** - Faster on subsequent runs
3. **Monitor logs** - Check for failures
4. **Backup regularly** - Dropbox is your backup, but consider additional backups
5. **Test first** - Run with a few symbols first to verify setup

## Examples

### Initial Download (All Symbols, 150 Days)

```bash
./scripts/sync_all_candles_to_dropbox.sh 150
```

### Daily Update (Update Only)

```bash
python scripts/auto_sync_candles.py --days 150 --update-only
```

### Weekly Full Sync (All Symbols)

```bash
python scripts/auto_sync_candles.py --days 150
```

### Custom Timeframe (Hourly Data)

```bash
python scripts/auto_sync_candles.py --days 365 --timeframe 1h
```

