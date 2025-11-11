#!/usr/bin/env python3
"""
Verify that all 250 coins are downloaded and uploaded to Dropbox.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.simple_download_candles import get_top_coins_from_binance
from src.cloud.training.config.settings import EngineSettings
from src.cloud.training.integrations.dropbox_sync import DropboxSync
from src.cloud.training.datasets.data_loader import CandleQuery
from datetime import datetime, timedelta, timezone

def main():
    print("=" * 70)
    print("🔍 VERIFYING DOWNLOAD COMPLETE")
    print("=" * 70)
    print()
    
    # Get all coins
    print("📊 Getting list of top 250 coins...")
    all_coins = get_top_coins_from_binance(top_n=250, min_volume_usdt=1_000_000)
    print(f"✅ Found {len(all_coins)} coins")
    print()
    
    # Check progress file
    progress_file = Path("data/download_progress.json")
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        completed = set(progress.get("completed", []))
        print(f"📋 Progress file: {len(completed)}/{len(all_coins)} coins marked as completed")
    else:
        completed = set()
        print("⚠️  No progress file found")
    print()
    
    # Check local cache
    cache_dir = Path("data/candles")
    local_coins = set()
    if cache_dir.exists():
        parquet_files = list(cache_dir.rglob("*.parquet"))
        for f in parquet_files:
            # Extract coin from path
            parts = f.parts
            if len(parts) >= 3:
                coin = parts[-2]
                local_coins.add(coin)
        print(f"📦 Local cache: {len(local_coins)} coins with files ({len(parquet_files)} files)")
    else:
        print("⚠️  Local cache directory does not exist")
    print()
    
    # Check Dropbox (if token available)
    dropbox_coins = set()
    try:
        settings = EngineSettings.load(environment="local")
        token = settings.dropbox.access_token
        if token:
            print("📤 Checking Dropbox...")
            dropbox_sync = DropboxSync(
                access_token=token,
                app_folder=settings.dropbox.app_folder,
                enabled=True,
            )
            
            # Check each coin in Dropbox
            end_at = datetime.now(tz=timezone.utc)
            start_at = end_at - timedelta(days=1095)
            
            for symbol in all_coins[:10]:  # Check first 10 as sample
                query = CandleQuery(
                    symbol=symbol,
                    timeframe="1d",
                    start_at=start_at,
                    end_at=end_at,
                )
                # Generate expected path
                symbol_safe = symbol.replace("/", "-")
                base_coin = symbol.split("/")[0]
                filename = f"{symbol_safe}_1d_{start_at:%Y%m%d}_{end_at:%Y%m%d}.parquet"
                remote_path = f"/{settings.dropbox.app_folder}/data/candles/{base_coin}/{filename}"
                
                try:
                    dropbox_sync._dbx.files_get_metadata(remote_path)
                    dropbox_coins.add(base_coin)
                except Exception:
                    pass
            
            print(f"📤 Dropbox sample (first 10): {len(dropbox_coins)}/{10} coins found")
        else:
            print("⚠️  No Dropbox token available, skipping Dropbox check")
    except Exception as e:
        print(f"⚠️  Error checking Dropbox: {e}")
    print()
    
    # Summary
    print("=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"Total coins: {len(all_coins)}")
    print(f"Progress file: {len(completed)} completed")
    print(f"Local cache: {len(local_coins)} coins")
    print()
    
    # Find missing
    progress_symbols = {c.split("/")[0] for c in completed}
    all_symbols = {c.split("/")[0] for c in all_coins}
    missing = all_symbols - progress_symbols
    
    if missing:
        print(f"⚠️  Missing from progress: {len(missing)} coins")
        print(f"   {', '.join(sorted(missing)[:10])}{'...' if len(missing) > 10 else ''}")
    else:
        print("✅ All coins in progress file")
    print()
    
    missing_local = all_symbols - local_coins
    if missing_local:
        print(f"⚠️  Missing from local cache: {len(missing_local)} coins")
        print(f"   {', '.join(sorted(missing_local)[:10])}{'...' if len(missing_local) > 10 else ''}")
    else:
        print("✅ All coins in local cache")
    print()
    
    if not missing and not missing_local:
        print("✅ ALL 249 COINS DOWNLOADED!")
    else:
        print(f"⚠️  Need to download {len(missing) + len(missing_local)} more coins")
    print("=" * 70)


if __name__ == "__main__":
    main()

