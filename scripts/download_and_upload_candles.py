#!/usr/bin/env python3
"""
Local script to download historical candle data and upload to Dropbox.

This script:
1. Downloads candle data for specified symbols from the exchange
2. Saves data locally to data/candles/
3. Uploads data to Dropbox (shared location: /Runpodhuracan/data/candles/)
4. Allows RunPod engine to restore from Dropbox instead of downloading from exchange

Usage:
    python scripts/download_and_upload_candles.py --symbols BTC/USDT ETH/USDT --days 150
    python scripts/download_and_upload_candles.py --all-symbols --days 150
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cloud.training.config.settings import EngineSettings
from src.cloud.training.datasets.data_loader import CandleDataLoader, CandleQuery
from src.cloud.training.datasets.quality_checks import DataQualitySuite
from src.cloud.training.integrations.dropbox_sync import DropboxSync
from src.cloud.training.services.exchange import ExchangeClient
import structlog

logger = structlog.get_logger(__name__)


def download_and_upload_candles(
    symbols: List[str],
    days: int = 150,
    exchange_id: str = "binance",
    timeframe: str = "1m",
    dropbox_token: Optional[str] = None,
    app_folder: str = "Runpodhuracan",
) -> None:
    """Download candle data for symbols and upload to Dropbox.
    
    Args:
        symbols: List of symbols to download (e.g., ["BTC/USDT", "ETH/USDT"])
        days: Number of days of historical data to download
        exchange_id: Exchange ID to download from
        timeframe: Timeframe for candles (1m, 5m, 1h, 1d, etc.)
        dropbox_token: Dropbox access token (if None, uses settings)
        app_folder: Dropbox app folder name
    """
    # Load settings (loads from config files)
    settings = EngineSettings.load(environment=os.getenv("HURACAN_ENV", "local"))
    
    # Initialize Dropbox
    dropbox_access_token = dropbox_token or settings.dropbox.access_token
    if not dropbox_access_token:
        logger.error("dropbox_token_missing", message="Dropbox token is required")
        print("❌ Error: Dropbox token is required")
        print("   Set DROPBOX_ACCESS_TOKEN environment variable or configure in settings")
        sys.exit(1)
    
    dropbox_sync = DropboxSync(
        access_token=dropbox_access_token,
        app_folder=app_folder,
        enabled=True,
        create_dated_folder=False,  # Don't create dated folder for shared data
    )
    
    # Initialize exchange client
    credentials = settings.exchange.credentials.get(exchange_id, {})
    exchange = ExchangeClient(exchange_id, credentials=credentials, sandbox=settings.exchange.sandbox)
    
    # Initialize data loader
    quality_suite = DataQualitySuite()
    loader = CandleDataLoader(
        exchange_client=exchange,
        quality_suite=quality_suite,
        cache_dir=Path("data/candles"),
    )
    
    # Calculate date range
    end_at = datetime.now(tz=timezone.utc)
    start_at = end_at - timedelta(days=days)
    
    print(f"\n📊 Downloading historical candle data")
    print(f"   Symbols: {', '.join(symbols)}")
    print(f"   Exchange: {exchange_id}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Days: {days}")
    print(f"   Start: {start_at.date()}")
    print(f"   End: {end_at.date()}\n")
    
    downloaded_count = 0
    uploaded_count = 0
    failed_count = 0
    
    for symbol in symbols:
        try:
            print(f"📥 Downloading {symbol}...")
            
            # Create query
            query = CandleQuery(
                symbol=symbol,
                timeframe=timeframe,
                start_at=start_at,
                end_at=end_at,
            )
            
            # Download data
            frame = loader.load(query, use_cache=True)
            
            if frame.is_empty():
                print(f"   ⚠️  No data available for {symbol}")
                failed_count += 1
                continue
            
            # Get cache path (same logic as CandleDataLoader._cache_path)
            symbol_safe = symbol.replace("/", "-")
            filename = f"{symbol_safe}_{timeframe}_{start_at:%Y%m%d}_{end_at:%Y%m%d}.parquet"
            cache_path = Path("data/candles") / filename
            
            # Ensure cache directory exists
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            if not cache_path.exists():
                print(f"   ❌ Cache file not found: {cache_path}")
                print(f"   ⚠️  Data was downloaded but cache file was not created")
                failed_count += 1
                continue
            
            downloaded_count += 1
            rows = len(frame)
            file_size = cache_path.stat().st_size
            print(f"   ✅ Downloaded {rows:,} rows ({file_size / 1024 / 1024:.2f} MB)")
            
            # Upload to Dropbox
            print(f"   📤 Uploading to Dropbox...")
            try:
                # Get relative path from data/candles/
                rel_path = cache_path.relative_to(Path("data/candles"))
                remote_path = f"/{app_folder}/data/candles/{rel_path.as_posix()}"
                
                # Upload to shared location (not dated folder)
                success = dropbox_sync.upload_file(
                    local_path=cache_path,
                    remote_path=remote_path,
                    use_dated_folder=False,  # Use shared location
                    overwrite=True,
                )
                
                if success:
                    uploaded_count += 1
                    print(f"   ✅ Uploaded to Dropbox: {remote_path}")
                else:
                    print(f"   ❌ Failed to upload to Dropbox")
                    failed_count += 1
                    
            except Exception as upload_error:
                print(f"   ❌ Upload failed: {upload_error}")
                failed_count += 1
                logger.warning("upload_failed", symbol=symbol, error=str(upload_error))
            
        except Exception as e:
            print(f"   ❌ Error downloading {symbol}: {e}")
            failed_count += 1
            logger.exception("download_failed", symbol=symbol, error=str(e))
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Summary")
    print(f"{'='*60}")
    print(f"   ✅ Downloaded: {downloaded_count}/{len(symbols)}")
    print(f"   📤 Uploaded: {uploaded_count}/{len(symbols)}")
    print(f"   ❌ Failed: {failed_count}/{len(symbols)}")
    print(f"{'='*60}\n")
    
    if uploaded_count > 0:
        print(f"✅ Successfully uploaded {uploaded_count} coin(s) to Dropbox")
        print(f"   Location: /{app_folder}/data/candles/")
        print(f"   RunPod engine will restore this data on startup\n")
    
    if failed_count > 0:
        print(f"⚠️  {failed_count} coin(s) failed to download/upload")
        sys.exit(1)


def get_all_symbols_from_universe() -> List[str]:
    """Get all symbols from universe selector."""
    try:
        from src.cloud.training.services.universe import UniverseSelector
        from src.cloud.training.config.settings import EngineSettings
        from src.cloud.training.services.exchange import ExchangeClient
        
        # Load settings (loads from config files)
        settings = EngineSettings.load(environment=os.getenv("HURACAN_ENV", "local"))
        
        exchange = ExchangeClient(
            exchange_id="binance",
            credentials=settings.exchange.credentials.get("binance", {}),
            sandbox=settings.exchange.sandbox,
        )
        universe_selector = UniverseSelector(exchange=exchange, settings=settings)
        universe = universe_selector.select()
        rows = list(universe.iter_rows(named=True))
        symbols = [row["symbol"] for row in rows]
        logger.info("universe_symbols_loaded", count=len(symbols), symbols=symbols[:10])
        return symbols
    except Exception as e:
        logger.warning("failed_to_get_universe_symbols", error=str(e))
        # Fallback to common symbols if universe selector fails
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT", "MATIC/USDT", "DOT/USDT", "AVAX/USDT"]


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download historical candle data and upload to Dropbox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download specific symbols
  python scripts/download_and_upload_candles.py --symbols BTC/USDT ETH/USDT --days 150
  
  # Download all symbols from universe
  python scripts/download_and_upload_candles.py --all-symbols --days 150
  
  # Use custom Dropbox token
  python scripts/download_and_upload_candles.py --symbols BTC/USDT --days 150 --dropbox-token YOUR_TOKEN
        """,
    )
    
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols to download (e.g., BTC/USDT ETH/USDT)",
    )
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Download all symbols from universe",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=150,
        help="Number of days of historical data (default: 150)",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default="binance",
        help="Exchange ID (default: binance)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1m",
        help="Timeframe for candles (default: 1m)",
    )
    parser.add_argument(
        "--dropbox-token",
        type=str,
        help="Dropbox access token (default: from settings or environment)",
    )
    parser.add_argument(
        "--app-folder",
        type=str,
        default="Runpodhuracan",
        help="Dropbox app folder name (default: Runpodhuracan)",
    )
    
    args = parser.parse_args()
    
    # Determine symbols
    if args.all_symbols:
        symbols = get_all_symbols_from_universe()
        if not symbols:
            print("❌ Error: Failed to get symbols from universe")
            print("   Use --symbols instead to specify symbols manually")
            sys.exit(1)
        print(f"📋 Found {len(symbols)} symbols in universe\n")
    elif args.symbols:
        symbols = args.symbols
    else:
        print("❌ Error: Must specify --symbols or --all-symbols")
        parser.print_help()
        sys.exit(1)
    
    # Run download and upload
    download_and_upload_candles(
        symbols=symbols,
        days=args.days,
        exchange_id=args.exchange,
        timeframe=args.timeframe,
        dropbox_token=args.dropbox_token,
        app_folder=args.app_folder,
    )


if __name__ == "__main__":
    main()

