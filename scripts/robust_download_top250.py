#!/usr/bin/env python3
"""
Robust downloader for top 250 coins with resume capability and error handling.

Features:
- Resume from where it left off (saves progress)
- Retry logic for failed downloads
- Rate limit handling with exponential backoff
- Continues even if some coins fail
- Saves state to resume later
- Better error handling and logging

Usage:
    python scripts/robust_download_top250.py --dropbox-token YOUR_TOKEN
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta, timezone

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import polars as pl
import structlog
from src.cloud.training.config.settings import EngineSettings
from src.cloud.training.datasets.data_loader import CandleDataLoader, CandleQuery
from src.cloud.training.datasets.quality_checks import DataQualitySuite
from src.cloud.training.services.exchange import ExchangeClient
from src.cloud.training.integrations.dropbox_sync import DropboxSync

logger = structlog.get_logger(__name__)

# Progress file to track completed downloads
PROGRESS_FILE = Path("data/download_progress.json")


def load_progress() -> Dict[str, any]:
    """Load download progress from file."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning("failed_to_load_progress", error=str(e))
    return {
        "completed": [],
        "failed": [],
        "started_at": None,
        "last_updated": None,
    }


def save_progress(progress: Dict[str, any]) -> None:
    """Save download progress to file."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    progress["last_updated"] = datetime.now(timezone.utc).isoformat()
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, indent=2)
    except Exception as e:
        logger.warning("failed_to_save_progress", error=str(e))


def get_completed_symbols() -> Set[str]:
    """Get set of symbols that have been successfully downloaded and uploaded."""
    progress = load_progress()
    return set(progress.get("completed", []))


def mark_completed(symbol: str, progress: Dict[str, any]) -> None:
    """Mark a symbol as completed."""
    if symbol not in progress["completed"]:
        progress["completed"].append(symbol)
    if symbol in progress["failed"]:
        progress["failed"].remove(symbol)
    save_progress(progress)


def mark_failed(symbol: str, error: str, progress: Dict[str, any]) -> None:
    """Mark a symbol as failed."""
    if symbol not in progress["failed"]:
        progress["failed"].append(symbol)
    save_progress(progress)


def download_symbol_with_retry(
    symbol: str,
    loader: CandleDataLoader,
    query: CandleQuery,
    dropbox_sync: Optional[DropboxSync],
    settings: EngineSettings,
    max_retries: int = 3,
    progress: Optional[Dict[str, any]] = None,
) -> tuple[bool, str]:
    """Download a symbol with retry logic.
    
    Returns:
        (success: bool, message: str)
    """
    cache_path = loader._cache_path(query)
    
    # Check if already completed
    if progress and symbol in progress.get("completed", []):
        if cache_path.exists() and dropbox_sync:
            # Verify it's in Dropbox
            rel_path = cache_path.relative_to(Path("data/candles"))
            remote_path = f"/{settings.dropbox.app_folder}/data/candles/{rel_path.as_posix()}"
            try:
                dropbox_sync._dbx.files_get_metadata(remote_path)
                return (True, f"Already completed (skipped)")
            except Exception:
                pass  # Not in Dropbox, need to upload
    
    for attempt in range(max_retries):
        try:
            # Check cache first
            frame = None
            if cache_path.exists():
                try:
                    frame = pl.read_parquet(cache_path)
                    if not frame.is_empty():
                        logger.info("loaded_from_cache", symbol=symbol, rows=len(frame))
                except Exception as e:
                    logger.warning("cache_load_failed", symbol=symbol, error=str(e))
            
            # Download if not in cache
            if frame is None or frame.is_empty():
                # Rate limit handling
                if attempt > 0:
                    wait_time = min(2 ** attempt, 30)  # Exponential backoff, max 30s
                    logger.info("retrying_download", symbol=symbol, attempt=attempt+1, wait_seconds=wait_time)
                    time.sleep(wait_time)
                
                try:
                    frame = loader._download(query, skip_validation=False)
                except ValueError as e:
                    if "Coverage" in str(e):
                        # Try lenient validation
                        logger.warning("strict_validation_failed", symbol=symbol, trying_lenient=True)
                        frame = loader._download(query, skip_validation=True)
                    else:
                        raise
            
            if frame is None or frame.is_empty():
                return (False, "No data available from exchange")
            
            # Save to cache
            if not cache_path.parent.exists():
                cache_path.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(cache_path)
            logger.info("downloaded", symbol=symbol, rows=len(frame))
            
            # Upload to Dropbox
            if dropbox_sync:
                rel_path = cache_path.relative_to(Path("data/candles"))
                remote_path = f"/{settings.dropbox.app_folder}/data/candles/{rel_path.as_posix()}"
                
                # Ensure directory exists
                remote_dir = "/".join(remote_path.split("/")[:-1])
                try:
                    dropbox_sync._dbx.files_create_folder_v2(remote_dir)
                except Exception:
                    pass  # Directory might already exist
                
                # Check if already uploaded (same size)
                try:
                    import dropbox
                    existing = dropbox_sync._dbx.files_get_metadata(remote_path)
                    if isinstance(existing, dropbox.files.FileMetadata):
                        local_size = cache_path.stat().st_size
                        remote_size = existing.size
                        if local_size == remote_size:
                            logger.info("already_in_dropbox", symbol=symbol, size=local_size)
                            if progress:
                                mark_completed(symbol, progress)
                            return (True, "Already in Dropbox (same size)")
                except Exception:
                    pass  # File doesn't exist, will upload
                
                # Upload with retry
                upload_success = False
                for upload_attempt in range(3):
                    try:
                        success = dropbox_sync.upload_file(
                            local_path=str(cache_path),
                            remote_path=remote_path,
                            use_dated_folder=False,
                            overwrite=True,
                        )
                        if success:
                            upload_success = True
                            break
                    except Exception as e:
                        if upload_attempt < 2:
                            wait_time = min(2 ** upload_attempt, 10)
                            logger.warning("upload_retry", symbol=symbol, attempt=upload_attempt+1, wait_seconds=wait_time)
                            time.sleep(wait_time)
                        else:
                            raise
                
                if not upload_success:
                    return (False, "Failed to upload to Dropbox")
            
            # Mark as completed
            if progress:
                mark_completed(symbol, progress)
            
            return (True, f"Downloaded {len(frame)} rows")
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for rate limits
            if "rate limit" in error_msg or "429" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = min(60 * (attempt + 1), 300)  # Wait up to 5 minutes
                    logger.warning("rate_limited", symbol=symbol, wait_seconds=wait_time, attempt=attempt+1)
                    time.sleep(wait_time)
                    continue
                else:
                    return (False, f"Rate limited after {max_retries} attempts")
            
            # Check for network errors
            elif "network" in error_msg or "timeout" in error_msg or "connection" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = min(10 * (attempt + 1), 60)
                    logger.warning("network_error", symbol=symbol, wait_seconds=wait_time, attempt=attempt+1)
                    time.sleep(wait_time)
                    continue
                else:
                    return (False, f"Network error after {max_retries} attempts: {e}")
            
            # Other errors
            else:
                if attempt < max_retries - 1:
                    wait_time = min(5 * (attempt + 1), 30)
                    logger.warning("download_error", symbol=symbol, error=str(e), attempt=attempt+1, wait_seconds=wait_time)
                    time.sleep(wait_time)
                    continue
                else:
                    if progress:
                        mark_failed(symbol, str(e), progress)
                    return (False, f"Failed after {max_retries} attempts: {e}")
    
    return (False, f"Failed after {max_retries} attempts")


def get_top_coins_from_binance(top_n: int = 250, min_volume_usdt: float = 1_000_000) -> List[str]:
    """Get top N coins by 24h volume from Binance."""
    from scripts.simple_download_candles import get_top_coins_from_binance as _get_top_coins
    return _get_top_coins(top_n=top_n, min_volume_usdt=min_volume_usdt)


def main():
    parser = argparse.ArgumentParser(
        description="Robust downloader for top 250 coins with resume capability"
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous progress",
    )
    
    args = parser.parse_args()
    
    # Set Dropbox token
    os.environ["DROPBOX_ACCESS_TOKEN"] = args.dropbox_token
    
    # Load progress
    progress = load_progress()
    completed = set(progress.get("completed", []))
    
    print("=" * 70)
    print("🚀 ROBUST DOWNLOADER FOR TOP COINS")
    print("=" * 70)
    print(f"Top N: {args.top}")
    print(f"Days: {args.days}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Min Volume: ${args.min_volume:,.0f} USDT")
    print(f"Delay: {args.delay}s between coins")
    print(f"Resume: {args.resume}")
    if completed:
        print(f"Already completed: {len(completed)} coins")
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
    
    # Get top coins
    print(f"📊 Fetching top {args.top} coins from Binance...")
    symbols = get_top_coins_from_binance(top_n=args.top, min_volume_usdt=args.min_volume)
    print(f"✅ Found {len(symbols)} coins")
    print()
    
    # Filter out completed if resuming
    if args.resume and completed:
        remaining = [s for s in symbols if s not in completed]
        print(f"📋 Resuming: {len(remaining)}/{len(symbols)} coins remaining")
        print(f"   Already completed: {len(completed)} coins")
        symbols = remaining
    elif completed:
        print(f"📋 Progress: {len(completed)}/{len(symbols)} coins already completed")
        print(f"   Use --resume to skip completed coins")
        print()
    
    if not symbols:
        print("✅ All coins already downloaded!")
        return
    
    # Calculate date range
    end_at = datetime.now(tz=timezone.utc)
    start_at = end_at - timedelta(days=args.days)
    
    # Initialize progress
    if not progress.get("started_at"):
        progress["started_at"] = datetime.now(timezone.utc).isoformat()
    save_progress(progress)
    
    # Download each symbol
    print(f"📥 Downloading {len(symbols)} coins...")
    print(f"   Local cache: data/candles/")
    print(f"   Dropbox cache: /{settings.dropbox.app_folder}/data/candles/")
    print(f"   Progress file: {PROGRESS_FILE}")
    print()
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for idx, symbol in enumerate(symbols, 1):
        # Skip if already completed (unless forced)
        if symbol in completed and not args.resume:
            skipped_count += 1
            print(f"[{idx}/{len(symbols)}] ⏭️  {symbol} (already completed)")
            continue
        
        print(f"[{idx}/{len(symbols)}] 📥 {symbol}...")
        
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
            max_retries=3,
            progress=progress,
        )
        
        if success:
            success_count += 1
            print(f"   ✅ {message}")
        else:
            failed_count += 1
            print(f"   ❌ {message}")
        
        # Delay between coins (except last)
        if idx < len(symbols):
            time.sleep(args.delay)
        
        # Periodic progress update
        if idx % 10 == 0:
            total_completed = len(progress.get("completed", []))
            print(f"\n📊 Progress: {total_completed} completed, {failed_count} failed, {idx}/{len(symbols)} processed\n")
    
    # Final summary
    print()
    print("=" * 70)
    print("📊 FINAL SUMMARY")
    print("=" * 70)
    print(f"   ✅ Success: {success_count}/{len(symbols)}")
    print(f"   ❌ Failed: {failed_count}/{len(symbols)}")
    print(f"   ⏭️  Skipped: {skipped_count}/{len(symbols)}")
    print(f"   📦 Total completed: {len(progress.get('completed', []))}")
    print(f"   📝 Progress saved: {PROGRESS_FILE}")
    print("=" * 70)
    print()
    
    if failed_count > 0:
        print("⚠️  Some coins failed. You can run again with --resume to retry failed coins.")
        print(f"   Failed coins: {', '.join(progress.get('failed', []))}")
        print()
    
    print("✅ Download complete!")


if __name__ == "__main__":
    main()

