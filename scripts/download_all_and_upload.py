#!/usr/bin/env python3
"""
Download all top 20 coins and upload to Dropbox with rate limit protection.

This script:
1. Downloads missing coins one at a time (sequential to avoid rate limits)
2. Uploads everything to Dropbox when complete
3. Shows progress and final status
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.fast_download_candles import download_symbol
from scripts.upload_local_candles_to_dropbox import upload_local_candles

TOP_20_COINS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "SHIB/USDT", "TON/USDT",
    "TRX/USDT", "DOT/USDT", "LINK/USDT", "MATIC/USDT", "NEAR/USDT",
    "UNI/USDT", "ICP/USDT", "LTC/USDT", "FIL/USDT", "ATOM/USDT"
]

def check_downloaded_coins():
    """Check which coins are already downloaded."""
    downloaded = set()
    for coin_dir in Path("data/candles").iterdir():
        if coin_dir.is_dir():
            files = list(coin_dir.glob("*.parquet"))
            if files:
                coin = coin_dir.name
                downloaded.add(coin)
    return downloaded

def get_missing_coins():
    """Get list of missing coins."""
    downloaded = check_downloaded_coins()
    missing = []
    for symbol in TOP_20_COINS:
        coin = symbol.split("/")[0]
        if coin not in downloaded:
            missing.append(symbol)
    return missing

def main():
    print("=" * 60)
    print("📊 Download All Top 20 Coins & Upload to Dropbox")
    print("=" * 60)
    print()
    
    # Check current status
    downloaded = check_downloaded_coins()
    missing = get_missing_coins()
    
    print(f"Already downloaded: {len(downloaded)}/20 coins")
    print(f"Missing: {len(missing)} coins")
    if missing:
        print(f"  {', '.join(missing)}")
    print()
    
    if not missing:
        print("✅ All coins already downloaded!")
    else:
        print(f"📥 Downloading {len(missing)} missing coins...")
        print("   (Sequential download with delays to avoid rate limits)")
        print()
        
        success_count = 0
        failed_count = 0
        
        for i, symbol in enumerate(missing, 1):
            print(f"[{i}/{len(missing)}] {symbol}...")
            
            # Add delay between coins (except first)
            if i > 1:
                time.sleep(3.0)  # 3 second delay to avoid rate limits
            
            coin, success, message = download_symbol(
                symbol=symbol,
                days=150,
                timeframe="1m",
                exchange_id="binance",
            )
            
            if success:
                success_count += 1
                print(f"   ✅ {message}")
            else:
                failed_count += 1
                print(f"   ❌ {message}")
            
            # Extra delay after each coin to be safe
            time.sleep(1.0)
        
        print()
        print("=" * 60)
        print("📊 Download Summary")
        print("=" * 60)
        print(f"   ✅ Success: {success_count}/{len(missing)}")
        print(f"   ❌ Failed: {failed_count}/{len(missing)}")
        print("=" * 60)
        print()
    
    # Upload everything to Dropbox
    print("=" * 60)
    print("📤 Uploading All Files to Dropbox")
    print("=" * 60)
    print()
    
    upload_local_candles()
    
    # Final status
    print()
    print("=" * 60)
    print("✅ Complete!")
    print("=" * 60)
    
    final_downloaded = check_downloaded_coins()
    print(f"Coins in Dropbox: {len(final_downloaded)}/20")
    print(f"Coins: {', '.join(sorted(final_downloaded))}")

if __name__ == "__main__":
    main()

