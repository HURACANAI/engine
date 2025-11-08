#!/usr/bin/env python3
"""
Fast parallel download of candle data.

Downloads multiple coins in parallel while respecting rate limits.
Much faster than sequential downloads.

Usage:
    python scripts/fast_download_candles.py --symbols BTC/USDT ETH/USDT --days 150
    python scripts/fast_download_candles.py --all-symbols --days 150
"""

import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.download_and_upload_candles import download_and_upload_candles, get_all_symbols_from_universe
import structlog  # type: ignore[reportMissingImports]

logger = structlog.get_logger(__name__)


def download_symbol(symbol: str, days: int, timeframe: str, exchange_id: str, dropbox_token: Optional[str] = None) -> tuple[str, bool, str]:
    """Download a single symbol. Returns (symbol, success, message)."""
    try:
        from datetime import datetime, timedelta, timezone
        from pathlib import Path
        from src.cloud.training.config.settings import EngineSettings
        from src.cloud.training.datasets.data_loader import CandleDataLoader, CandleQuery
        from src.cloud.training.datasets.quality_checks import DataQualitySuite
        from src.cloud.training.services.exchange import ExchangeClient
        from src.cloud.training.integrations.dropbox_sync import DropboxSync
        
        # Load settings
        settings = EngineSettings.load(environment=os.getenv("HURACAN_ENV", "local"))
        
        # Initialize exchange
        credentials = settings.exchange.credentials.get(exchange_id, {})
        exchange = ExchangeClient(exchange_id, credentials=credentials, sandbox=settings.exchange.sandbox)
        
        # Initialize loader
        quality_suite = DataQualitySuite()
        loader = CandleDataLoader(
            exchange_client=exchange,
            quality_suite=quality_suite,
            cache_dir=Path("data/candles"),
        )
        
        # Calculate date range
        end_at = datetime.now(tz=timezone.utc)
        start_at = end_at - timedelta(days=days)
        
        # Normalize symbol
        normalized_symbol = symbol.split(":")[0] if ":" in symbol else symbol
        
        # Create query
        query = CandleQuery(
            symbol=normalized_symbol,
            timeframe=timeframe,
            start_at=start_at,
            end_at=end_at,
        )
        
        # Download
        frame = loader.load(query, use_cache=True)
        
        if frame.is_empty():
            return (symbol, False, "No data available")
        
        # Get cache path (should be in coin folder now)
        cache_path = loader._cache_path(query)
        if not cache_path.exists():
            return (symbol, False, f"Cache file not found: {cache_path}")
        
        # Upload to Dropbox if token provided
        if dropbox_token or settings.dropbox.access_token:
            try:
                dropbox_sync = DropboxSync(
                    access_token=dropbox_token or settings.dropbox.access_token,
                    app_folder=settings.dropbox.app_folder,
                    enabled=True,
                    create_dated_folder=False,
                )
                rel_path = cache_path.relative_to(Path("data/candles"))
                remote_path = f"/{settings.dropbox.app_folder}/data/candles/{rel_path.as_posix()}"
                dropbox_sync.upload_file(
                    local_path=cache_path,
                    remote_path=remote_path,
                    use_dated_folder=False,
                    overwrite=True,
                )
            except Exception as upload_error:
                return (symbol, True, f"Downloaded but upload failed: {upload_error}")
        
        rows = len(frame)
        file_size = cache_path.stat().st_size
        return (symbol, True, f"Downloaded {rows:,} rows ({file_size / 1024 / 1024:.2f} MB)")
        
    except Exception as e:
        return (symbol, False, str(e))


def fast_download(
    symbols: List[str],
    days: int = 150,
    exchange_id: str = "binance",
    timeframe: str = "1m",
    max_workers: int = 2,  # Download 2 coins in parallel (reduced to avoid rate limits)
    dropbox_token: Optional[str] = None,
) -> None:
    """Download multiple symbols in parallel for faster execution."""
    print(f"\n🚀 Fast Parallel Download")
    print(f"   Symbols: {len(symbols)}")
    print(f"   Parallel workers: {max_workers}")
    print(f"   Days: {days}")
    print(f"   Timeframe: {timeframe}\n")
    
    success_count = 0
    failed_count = 0
    
    # Download in parallel batches
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_symbol, symbol, days, timeframe, exchange_id, dropbox_token): symbol
            for symbol in symbols
        }
        
        for future in as_completed(futures):
            symbol, success, message = future.result()
            if success:
                success_count += 1
                print(f"✅ {symbol} - {message}")
            else:
                failed_count += 1
                print(f"❌ {symbol} - {message}")
    
    print(f"\n{'='*60}")
    print(f"📊 Summary")
    print(f"{'='*60}")
    print(f"   ✅ Success: {success_count}/{len(symbols)}")
    print(f"   ❌ Failed: {failed_count}/{len(symbols)}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Fast parallel download of candle data")
    parser.add_argument("--symbols", nargs="+", help="Symbols to download")
    parser.add_argument("--all-symbols", action="store_true", help="Download all symbols from universe")
    parser.add_argument("--days", type=int, default=150, help="Number of days")
    parser.add_argument("--timeframe", type=str, default="1m", help="Timeframe")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange ID")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    parser.add_argument("--dropbox-token", type=str, help="Dropbox token")
    
    args = parser.parse_args()
    
    if args.all_symbols:
        symbols = get_all_symbols_from_universe()
    elif args.symbols:
        symbols = args.symbols
    else:
        print("❌ Error: Must specify --symbols or --all-symbols")
        sys.exit(1)
    
    fast_download(
        symbols=symbols,
        days=args.days,
        exchange_id=args.exchange,
        timeframe=args.timeframe,
        max_workers=args.workers,
        dropbox_token=args.dropbox_token,
    )


if __name__ == "__main__":
    main()

