#!/usr/bin/env python3
"""
Download top 250 coins and cache them in Dropbox for daily retraining.

This script:
1. Downloads top 250 coins by 24h volume from Binance
2. Caches data locally in data/candles/{SYMBOL}/
3. Uploads to Dropbox at /Runpodhuracan/data/candles/{SYMBOL}/
4. Uses shared cache location (persists across days for daily retraining)
5. Skips already-downloaded coins if they exist in Dropbox

Usage:
    python scripts/download_top250_for_daily_training.py --dropbox-token YOUR_TOKEN
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.simple_download_candles import download_and_upload, get_top_coins_from_binance


def main():
    parser = argparse.ArgumentParser(
        description="Download top 250 coins and cache in Dropbox for daily retraining"
    )
    parser.add_argument(
        "--dropbox-token",
        type=str,
        required=True,
        help="Dropbox access token",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=250,
        help="Number of top coins to download (default: 250)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1095,
        help="Number of days of history (default: 1095 = 3 years)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1d",
        choices=["1m", "5m", "15m", "1h", "4h", "1d"],
        help="Timeframe for candles (default: 1d for daily retraining)",
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=1_000_000,
        help="Minimum 24h volume in USDT (default: 1,000,000)",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default="binance",
        help="Exchange ID (default: binance)",
    )
    
    args = parser.parse_args()
    
    # Set Dropbox token as environment variable
    os.environ["DROPBOX_ACCESS_TOKEN"] = args.dropbox_token
    
    print("=" * 70)
    print("🚀 DOWNLOADING TOP COINS FOR DAILY RETRAINING")
    print("=" * 70)
    print(f"Top N: {args.top}")
    print(f"Days: {args.days}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Min Volume: ${args.min_volume:,.0f} USDT")
    print(f"Exchange: {args.exchange}")
    print(f"Dropbox Location: /Runpodhuracan/data/candles/ (shared cache)")
    print("=" * 70)
    print()
    
    # Get top coins
    print(f"📊 Fetching top {args.top} coins from Binance...")
    symbols = get_top_coins_from_binance(
        top_n=args.top,
        min_volume_usdt=args.min_volume
    )
    print(f"✅ Found {len(symbols)} coins")
    print()
    
    # Download and upload
    print(f"📥 Downloading and caching {len(symbols)} coins...")
    print(f"   Local cache: data/candles/")
    print(f"   Dropbox cache: /Runpodhuracan/data/candles/")
    print(f"   (This cache persists across days for daily retraining)")
    print()
    
    download_and_upload(
        symbols=symbols,
        days=args.days,
        exchange_id=args.exchange,
        timeframe=args.timeframe,
        dropbox_token=args.dropbox_token,
        use_adaptive=False,  # Use fixed days for consistency
    )
    
    print()
    print("=" * 70)
    print("✅ DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"✅ Downloaded {len(symbols)} coins")
    print(f"✅ Cached locally: data/candles/")
    print(f"✅ Cached in Dropbox: /Runpodhuracan/data/candles/")
    print()
    print("📝 Next Steps:")
    print("   1. Data is now cached in Dropbox for daily retraining")
    print("   2. Daily retraining will restore this data from Dropbox")
    print("   3. Only new/missing data will be downloaded during retraining")
    print("=" * 70)


if __name__ == "__main__":
    main()

