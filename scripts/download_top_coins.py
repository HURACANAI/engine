#!/usr/bin/env python3
"""
Download Candle Data for Top N Coins

Downloads historical candle data for top N coins by 24h volume from Binance.
Designed for scaling: download data for many coins, train on subset later.

Usage:
    python scripts/download_top_coins.py --top 250 --days 365 --timeframe 1h
    python scripts/download_top_coins.py --top 250 --days 1095 --timeframe 1h  # 3 years
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Set

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cloud.training.config.settings import EngineSettings
from src.cloud.training.datasets.data_loader import CandleDataLoader, CandleQuery
from src.cloud.training.datasets.quality_checks import DataQualitySuite
from src.cloud.training.services.exchange import ExchangeClient
from src.cloud.training.integrations.dropbox_sync import DropboxSync
import structlog

logger = structlog.get_logger(__name__)


def get_active_spot_markets(exchange: ExchangeClient) -> Set[str]:
    """Get set of active spot market symbols from exchange.markets."""
    if not exchange._client.markets:
        exchange._client.load_markets(reload=True)
    
    markets = exchange._client.markets or {}
    spot_symbols = set()
    
    for symbol, market in markets.items():
        if market.get("spot") and market.get("active"):
            spot_symbols.add(symbol)
    
    return spot_symbols


def get_top_coins_by_volume(
    exchange: ExchangeClient,
    top_n: int = 250,
    min_volume_usdt: float = 1_000_000,  # Minimum $1M 24h volume
) -> List[str]:
    """Get top N coins by 24h volume from Binance.
    
    Args:
        exchange: Exchange client
        top_n: Number of top coins to return
        min_volume_usdt: Minimum 24h volume in USDT
        
    Returns:
        List of symbol strings (e.g., ["BTC/USDT", "ETH/USDT", ...])
    """
    print(f"📊 Fetching top {top_n} coins from Binance by 24h volume...")
    
    # Get active spot markets
    spot_markets = get_active_spot_markets(exchange)
    logger.info("active_spot_markets", count=len(spot_markets))
    
    # Get all tickers
    tickers = exchange.fetch_tickers()
    
    # Filter USDT pairs (spot, not futures)
    # Exclude stablecoins and wrapped tokens
    excluded_bases = {
        "USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDT", "USDP",
        "WBTC", "WETH", "WBETH",  # Wrapped tokens
    }
    
    usdt_pairs = []
    for symbol, ticker in tickers.items():
        # Only consider symbols that are in spot markets
        if symbol not in spot_markets:
            continue
        
        # Only USDT pairs (not futures)
        if "/USDT" in symbol and ":USDT" not in symbol:
            # Extract base coin
            base = symbol.split("/")[0]
            # Skip stablecoins and wrapped tokens
            if base in excluded_bases:
                continue
            
            volume = ticker.get("quoteVolume", 0) or 0
            if volume >= min_volume_usdt:
                usdt_pairs.append((symbol, volume))
    
    # Sort by volume descending
    usdt_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N symbols
    top_coins = [symbol for symbol, volume in usdt_pairs[:top_n]]
    
    print(f"✅ Found {len(top_coins)} coins with volume >= ${min_volume_usdt:,.0f}")
    return top_coins


def download_and_upload_coins(
    symbols: List[str],
    days: int = 365,
    timeframe: str = "1h",
    exchange_id: str = "binance",
    dropbox_token: Optional[str] = None,
    upload_to_dropbox: bool = True,
    delay_seconds: float = 0.5,
) -> None:
    """Download and upload candle data for multiple coins.
    
    Args:
        symbols: List of symbols to download
        days: Number of days of history
        timeframe: Timeframe (e.g., "1h", "1d")
        exchange_id: Exchange ID
        dropbox_token: Dropbox access token
        upload_to_dropbox: Whether to upload to Dropbox
        delay_seconds: Delay between downloads (to avoid rate limits)
    """
    settings = EngineSettings.load(environment=os.getenv("HURACAN_ENV", "local"))
    
    # Initialize Dropbox (if enabled)
    dropbox_sync = None
    if upload_to_dropbox:
        token = dropbox_token or settings.dropbox.access_token
        if token:
            try:
                dropbox_sync = DropboxSync(
                    access_token=token,
                    app_folder=settings.dropbox.app_folder,
                    enabled=True,
                    create_dated_folder=False,
                )
                print("✅ Dropbox ready\n")
            except Exception as e:
                print(f"⚠️  Dropbox not available: {e}\n")
                upload_to_dropbox = False
        else:
            print("⚠️  No Dropbox token available, skipping upload\n")
            upload_to_dropbox = False
    
    # Initialize exchange
    credentials = settings.exchange.credentials.get(exchange_id, {})
    exchange = ExchangeClient(exchange_id, credentials=credentials, sandbox=settings.exchange.sandbox)
    
    # Initialize loader
    quality_suite = DataQualitySuite(coverage_threshold=0.5)  # Lenient threshold for bulk download
    loader = CandleDataLoader(
        exchange_client=exchange,
        quality_suite=quality_suite,
        cache_dir=Path("data/candles"),
    )
    
    end_at = datetime.now(tz=timezone.utc)
    start_at = end_at - timedelta(days=days)
    
    print(f"📊 Downloading {len(symbols)} coins")
    print(f"   Exchange: {exchange_id}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Days: {days}")
    print(f"   Start: {start_at.date()}")
    print(f"   End: {end_at.date()}")
    print(f"   Upload to Dropbox: {upload_to_dropbox}\n")
    
    downloaded_count = 0
    uploaded_count = 0
    skipped_count = 0
    failed_count = 0
    
    for idx, symbol in enumerate(symbols, 1):
        # Delay between downloads to avoid rate limits
        if idx > 1:
            time.sleep(delay_seconds)
        
        try:
            print(f"[{idx}/{len(symbols)}] 📥 {symbol}...")
            
            # Create query
            query = CandleQuery(
                symbol=symbol,
                timeframe=timeframe,
                start_at=start_at,
                end_at=end_at,
            )
            
            # Try to load from cache first
            cache_path = loader._cache_path(query)
            if cache_path.exists():
                print(f"   ⏭️  Already cached, skipping download")
                downloaded_count += 1
                
                # Upload to Dropbox if enabled
                if upload_to_dropbox and dropbox_sync:
                    try:
                        rel_path = cache_path.relative_to(Path("data/candles"))
                        remote_path = f"/{settings.dropbox.app_folder}/data/candles/{rel_path.as_posix()}"
                        
                        # Check if already in Dropbox
                        try:
                            import dropbox
                            existing = dropbox_sync._dbx.files_get_metadata(remote_path)
                            if isinstance(existing, dropbox.files.FileMetadata):
                                print(f"   ✅ Already in Dropbox")
                                uploaded_count += 1
                                continue
                        except:
                            pass
                        
                        # Upload to Dropbox
                        success = dropbox_sync.upload_file(
                            local_path=cache_path,
                            remote_path=remote_path,
                            use_dated_folder=False,
                            overwrite=True,
                        )
                        if success:
                            uploaded_count += 1
                            print(f"   ✅ Uploaded to Dropbox")
                    except Exception as e:
                        print(f"   ⚠️  Upload failed: {e}")
                
                continue
            
            # Download data
            try:
                frame = loader.load(query, use_cache=True)
                
                if frame.is_empty():
                    print(f"   ⚠️  No data available")
                    skipped_count += 1
                    continue
                
                downloaded_count += 1
                file_size = cache_path.stat().st_size if cache_path.exists() else 0
                print(f"   ✅ Downloaded {len(frame):,} rows ({file_size / 1024 / 1024:.2f} MB)")
                
                # Upload to Dropbox if enabled
                if upload_to_dropbox and dropbox_sync and cache_path.exists():
                    try:
                        rel_path = cache_path.relative_to(Path("data/candles"))
                        remote_path = f"/{settings.dropbox.app_folder}/data/candles/{rel_path.as_posix()}"
                        
                        success = dropbox_sync.upload_file(
                            local_path=cache_path,
                            remote_path=remote_path,
                            use_dated_folder=False,
                            overwrite=True,
                        )
                        if success:
                            uploaded_count += 1
                            print(f"   ✅ Uploaded to Dropbox")
                        else:
                            print(f"   ⚠️  Upload failed")
                    except Exception as e:
                        print(f"   ⚠️  Upload failed: {e}")
                
            except Exception as download_error:
                error_msg = str(download_error).lower()
                if "rate limit" in error_msg or "429" in error_msg:
                    print(f"   ⚠️  Rate limited - waiting 10 seconds...")
                    time.sleep(10)
                    skipped_count += 1
                elif "not found" in error_msg or "invalid symbol" in error_msg:
                    print(f"   ⚠️  Symbol not available: {symbol}")
                    skipped_count += 1
                else:
                    print(f"   ❌ Error: {download_error}")
                    failed_count += 1
                    
        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted by user")
            raise
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            failed_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Summary")
    print(f"{'='*60}")
    print(f"   ✅ Downloaded: {downloaded_count}/{len(symbols)}")
    print(f"   📤 Uploaded: {uploaded_count}/{downloaded_count}")
    print(f"   ⚠️  Skipped: {skipped_count}/{len(symbols)}")
    print(f"   ❌ Failed: {failed_count}/{len(symbols)}")
    print(f"{'='*60}\n")
    
    if downloaded_count > 0:
        print(f"✅ Successfully downloaded {downloaded_count} coin(s)")
        if uploaded_count > 0:
            print(f"✅ Successfully uploaded {uploaded_count} file(s) to Dropbox")
            print(f"   Location: /{settings.dropbox.app_folder}/data/candles/\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download candle data for top N coins by volume"
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
        default=365,
        help="Number of days of history (default: 365 = 1 year)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Timeframe (default: 1h)",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default="binance",
        help="Exchange ID (default: binance)",
    )
    parser.add_argument(
        "--dropbox-token",
        type=str,
        help="Dropbox access token",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Don't upload to Dropbox",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between downloads in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=1_000_000,
        help="Minimum 24h volume in USDT (default: 1,000,000)",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
    )
    
    print("=" * 70)
    print("🚀 DOWNLOAD TOP COINS")
    print("=" * 70)
    print(f"Top N: {args.top}")
    print(f"Days: {args.days}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Upload to Dropbox: {not args.no_upload}")
    print("=" * 70)
    print()
    
    # Initialize exchange
    settings = EngineSettings.load(environment=os.getenv("HURACAN_ENV", "local"))
    credentials = settings.exchange.credentials.get(args.exchange, {})
    exchange = ExchangeClient(
        args.exchange,
        credentials=credentials,
        sandbox=settings.exchange.sandbox,
    )
    
    # Get top coins
    top_coins = get_top_coins_by_volume(
        exchange,
        top_n=args.top,
        min_volume_usdt=args.min_volume,
    )
    
    if not top_coins:
        print("❌ No coins found")
        sys.exit(1)
    
    print(f"\n📋 Top {len(top_coins)} coins:")
    for i, symbol in enumerate(top_coins[:10], 1):
        print(f"   {i:3}. {symbol}")
    if len(top_coins) > 10:
        print(f"   ... and {len(top_coins) - 10} more")
    print()
    
    # Download and upload
    download_and_upload_coins(
        symbols=top_coins,
        days=args.days,
        timeframe=args.timeframe,
        exchange_id=args.exchange,
        dropbox_token=args.dropbox_token,
        upload_to_dropbox=not args.no_upload,
        delay_seconds=args.delay,
    )


if __name__ == "__main__":
    main()

