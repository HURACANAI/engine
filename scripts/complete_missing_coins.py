#!/usr/bin/env python3
"""
Download missing coins to complete the 250 coin dataset.

This script:
1. Gets the list of top 250 coins
2. Checks which ones are already downloaded (from progress file and Dropbox)
3. Downloads only the missing coins
4. Uploads them to Dropbox
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Set
from datetime import datetime, timedelta, timezone

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.robust_download_top250 import (
    download_symbol_with_retry,
    load_progress,
    mark_completed,
    mark_failed,
    get_top_coins_from_binance,
)
from src.cloud.training.config.settings import EngineSettings
from src.cloud.training.datasets.data_loader import CandleDataLoader, CandleQuery
from src.cloud.training.datasets.quality_checks import DataQualitySuite
from src.cloud.training.services.exchange import ExchangeClient
from src.cloud.training.integrations.dropbox_sync import DropboxSync


def get_missing_coins(
    all_coins: List[str],
    progress: dict,
    dropbox_sync: DropboxSync,
    settings: EngineSettings,
    loader: CandleDataLoader,
    query_template: CandleQuery,
) -> List[str]:
    """Get list of coins that are missing (not completed)."""
    completed = set(progress.get("completed", []))
    missing = []
    
    for symbol in all_coins:
        if symbol in completed:
            # Double-check it's actually in Dropbox
            query = CandleQuery(
                symbol=symbol,
                timeframe=query_template.timeframe,
                start_at=query_template.start_at,
                end_at=query_template.end_at,
            )
            cache_path = loader._cache_path(query)
            rel_path = cache_path.relative_to(Path("data/candles"))
            remote_path = f"/{settings.dropbox.app_folder}/data/candles/{rel_path.as_posix()}"
            
            try:
                dropbox_sync._dbx.files_get_metadata(remote_path)
                # File exists in Dropbox, skip
                continue
            except Exception:
                # File doesn't exist, need to download
                missing.append(symbol)
        else:
            # Not in completed list, need to download
            missing.append(symbol)
    
    return missing


def main():
    parser = argparse.ArgumentParser(
        description="Download missing coins to complete the 250 coin dataset"
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
        help="Number of top coins (default: 250)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1095,
        help="Days of history (default: 1095 = 3 years)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1d",
        choices=["1m", "5m", "15m", "1h", "4h", "1d"],
        help="Timeframe (default: 1d)",
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=1_000_000,
        help="Minimum 24h volume in USDT (default: 1,000,000)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between coins in seconds (default: 0.5)",
    )
    
    args = parser.parse_args()
    
    # Set Dropbox token
    os.environ["DROPBOX_ACCESS_TOKEN"] = args.dropbox_token
    
    print("=" * 70)
    print("🔍 FINDING MISSING COINS")
    print("=" * 70)
    print()
    
    # Initialize
    settings = EngineSettings.load(environment=os.getenv("HURACAN_ENV", "local"))
    
    # Dropbox sync
    dropbox_sync = DropboxSync(
        access_token=args.dropbox_token,
        app_folder=settings.dropbox.app_folder,
        enabled=True,
    )
    
    # Exchange client
    exchange = ExchangeClient("binance", credentials={}, sandbox=settings.exchange.sandbox)
    
    # Data loader
    quality_suite = DataQualitySuite(coverage_threshold=0.01)
    loader = CandleDataLoader(
        exchange_client=exchange,
        quality_suite=quality_suite,
        cache_dir=Path("data/candles"),
        fallback_exchanges=[],
    )
    
    # Get all coins
    print(f"📊 Fetching top {args.top} coins from Binance...")
    all_coins = get_top_coins_from_binance(top_n=args.top, min_volume_usdt=args.min_volume)
    print(f"✅ Found {len(all_coins)} coins")
    print()
    
    # Load progress
    progress = load_progress()
    completed = set(progress.get("completed", []))
    print(f"📋 Progress: {len(completed)}/{len(all_coins)} coins completed")
    print()
    
    # Calculate date range
    end_at = datetime.now(tz=timezone.utc)
    start_at = end_at - timedelta(days=args.days)
    query_template = CandleQuery(
        symbol="BTC/USDT",  # Template, will be replaced
        timeframe=args.timeframe,
        start_at=start_at,
        end_at=end_at,
    )
    
    # Find missing coins
    print("🔍 Checking for missing coins...")
    missing_coins = get_missing_coins(
        all_coins=all_coins,
        progress=progress,
        dropbox_sync=dropbox_sync,
        settings=settings,
        loader=loader,
        query_template=query_template,
    )
    
    print(f"✅ Found {len(missing_coins)} missing coins")
    if missing_coins:
        print(f"   Missing: {', '.join(missing_coins[:10])}{'...' if len(missing_coins) > 10 else ''}")
    print()
    
    if not missing_coins:
        print("✅ All coins already downloaded!")
        return
    
    # Download missing coins
    print("=" * 70)
    print(f"📥 DOWNLOADING {len(missing_coins)} MISSING COINS")
    print("=" * 70)
    print()
    
    success_count = 0
    failed_count = 0
    
    for idx, symbol in enumerate(missing_coins, 1):
        print(f"[{idx}/{len(missing_coins)}] 📥 {symbol}...")
        
        # Create query
        query = CandleQuery(
            symbol=symbol,
            timeframe=args.timeframe,
            start_at=start_at,
            end_at=end_at,
        )
        
        # Download with retry
        success, message = download_symbol_with_retry(
            symbol=symbol,
            loader=loader,
            query=query,
            dropbox_sync=dropbox_sync,
            settings=settings,
            max_retries=5,  # More retries for missing coins
            progress=progress,
        )
        
        if success:
            success_count += 1
            print(f"   ✅ {message}")
        else:
            failed_count += 1
            print(f"   ❌ {message}")
        
        # Delay between coins (except last)
        if idx < len(missing_coins):
            time.sleep(args.delay)
        
        # Periodic progress update
        if idx % 10 == 0:
            total_completed = len(progress.get("completed", []))
            print(f"\n📊 Progress: {total_completed} completed, {failed_count} failed, {idx}/{len(missing_coins)} processed\n")
    
    # Final summary
    print()
    print("=" * 70)
    print("📊 FINAL SUMMARY")
    print("=" * 70)
    print(f"   ✅ Success: {success_count}/{len(missing_coins)}")
    print(f"   ❌ Failed: {failed_count}/{len(missing_coins)}")
    print(f"   📦 Total completed: {len(progress.get('completed', []))}/{len(all_coins)}")
    print("=" * 70)
    print()
    
    if failed_count > 0:
        print("⚠️  Some coins failed. Check logs for details.")
        print(f"   Failed coins: {', '.join(progress.get('failed', []))}")
        print()
        print("💡 Tip: Run this script again to retry failed coins.")
    else:
        print("✅ All missing coins downloaded!")
    print()


if __name__ == "__main__":
    main()

