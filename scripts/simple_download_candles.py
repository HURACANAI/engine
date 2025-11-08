#!/usr/bin/env python3
"""
Simple, fast, error-free candle download with market validation.

Downloads coins sequentially from Binance with:
- Real, active spot market validation
- Symbol normalization (Binance IDs from exchange.markets)
- Age checks (onboardDate validation)
- Adaptive window (150 -> 60 -> 30 days)
- Ticker alias mapping (ASTR not ASTER, XPLA not XPL)
- Fail-safe: log and continue on low coverage

Usage:
    python scripts/simple_download_candles.py --symbols BTC/USDT ETH/USDT --days 150
    python scripts/simple_download_candles.py --all-top20 --days 150
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cloud.training.config.settings import EngineSettings
from src.cloud.training.datasets.data_loader import CandleDataLoader, CandleQuery
from src.cloud.training.datasets.quality_checks import DataQualitySuite
from src.cloud.training.services.exchange import ExchangeClient
from src.cloud.training.integrations.dropbox_sync import DropboxSync
import structlog  # type: ignore[reportMissingImports]
import polars as pl  # type: ignore[reportMissingImports]

logger = structlog.get_logger(__name__)

# Ticker alias map - map common names to Binance IDs
TICKER_ALIASES: Dict[str, str] = {
    "ASTER/USDT": "ASTR/USDT",
    "XPL/USDT": "XPLA/USDT",
    # Add more as needed
}

MIN_COVERAGE = 0.95
ADAPTIVE_DAYS = [150, 60, 30]  # Try 150 days first, then 60, then 30


def get_active_spot_markets(exchange: ExchangeClient) -> set:
    """Get set of active spot market symbols from exchange.markets."""
    # Ensure markets are loaded
    if not exchange._client.markets:
        exchange._client.load_markets(reload=True)
    
    markets = exchange._client.markets or {}
    spot_symbols = set()
    
    for symbol, market in markets.items():
        # Check if it's a spot market and active
        if market.get("spot") and market.get("active"):
            spot_symbols.add(symbol)
    
    return spot_symbols


def normalize_symbols(symbols: List[str], spot_markets: set, aliases: Dict[str, str]) -> List[str]:
    """Normalize symbols using alias map and filter to valid spot markets."""
    normalized = []
    for symbol in symbols:
        # Apply alias if exists
        mapped_symbol = aliases.get(symbol, symbol)
        # Only include if it's a valid spot market
        if mapped_symbol in spot_markets:
            normalized.append(mapped_symbol)
        else:
            logger.warning("symbol_not_spot_market", original=symbol, mapped=mapped_symbol)
    
    return normalized


def has_enough_age(market_info: dict, start_ms: int) -> bool:
    """Check if market has enough age (onboardDate <= start_time)."""
    info = market_info.get("info", {})
    onboard_date = info.get("onboardDate")
    
    if onboard_date is None:
        # No onboard date available - assume it's old enough
        return True
    
    try:
        # Binance onboardDate is in milliseconds
        onboard_ms = int(onboard_date)
        return onboard_ms <= start_ms
    except (ValueError, TypeError):
        # If we can't parse, assume it's fine
        return True


def filter_by_age(symbols: List[str], exchange: ExchangeClient, start_at: datetime) -> List[str]:
    """Filter symbols to only those with enough trading history."""
    start_ms = int(start_at.timestamp() * 1000)
    filtered = []
    
    for symbol in symbols:
        try:
            market = exchange._client.market(symbol)
            if has_enough_age(market, start_ms):
                filtered.append(symbol)
            else:
                onboard_date = market.get("info", {}).get("onboardDate")
                logger.info(
                    "symbol_too_new",
                    symbol=symbol,
                    onboard_date=onboard_date,
                    start_date=start_at.isoformat(),
                )
        except Exception as e:
            logger.warning("age_check_failed", symbol=symbol, error=str(e))
            # If we can't check age, include it (fail-safe)
            filtered.append(symbol)
    
    return filtered


def get_top20_coins_from_binance() -> List[str]:
    """Get top 20 coins by 24h volume from Binance, validated against spot markets."""
    settings = EngineSettings.load(environment=os.getenv("HURACAN_ENV", "local"))
    exchange = ExchangeClient("binance", credentials={}, sandbox=settings.exchange.sandbox)
    
    # Get active spot markets first
    spot_markets = get_active_spot_markets(exchange)
    logger.info("active_spot_markets", count=len(spot_markets))
    
    # Get all tickers
    tickers = exchange.fetch_tickers()
    
    # Filter USDT pairs (spot, not futures)
    # Exclude stablecoins (USDC, FDUSD, TUSD, BUSD, DAI, etc.)
    excluded_bases = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDT", "USDP"}
    
    usdt_pairs = []
    for symbol, ticker in tickers.items():
        # Only consider symbols that are in spot markets
        if symbol not in spot_markets:
            continue
        
        if "/USDT" in symbol and ":USDT" not in symbol:
            # Extract base coin
            base = symbol.split("/")[0]
            # Skip stablecoins
            if base in excluded_bases:
                continue
            volume = ticker.get("quoteVolume", 0) or 0
            if volume > 0:
                usdt_pairs.append((symbol, volume))
    
    # Sort by volume descending
    usdt_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Return top 20 symbols (actual cryptocurrencies, not stablecoins)
    top20 = [symbol for symbol, volume in usdt_pairs[:20]]
    
    # Apply alias mapping and validate
    top20 = normalize_symbols(top20, spot_markets, TICKER_ALIASES)
    
    return top20


def fetch_with_adaptive_days(
    symbol: str,
    loader: CandleDataLoader,
    timeframe: str,
    end_at: datetime,
    days_sequence: List[int],
    min_coverage: float,
) -> Tuple[Optional[pl.DataFrame], Optional[int], Optional[float]]:
    """Try downloading with adaptive day windows. Returns (frame, used_days, coverage) or (None, None, None)."""
    for days in days_sequence:
        try:
            start_at = end_at - timedelta(days=days)
            query = CandleQuery(
                symbol=symbol,
                timeframe=timeframe,
                start_at=start_at,
                end_at=end_at,
            )
            
            # Calculate expected rows
            delta = query.end_at - query.start_at
            expected_rows = int(delta.total_seconds() // 60) + 1
            
            # Try to load (will download if not cached)
            frame = loader.load(query, use_cache=True)
            
            if frame.is_empty():
                logger.debug("no_data", symbol=symbol, days=days)
                continue
            
            # Calculate coverage
            coverage = len(frame) / expected_rows if expected_rows else 0.0
            
            if coverage >= min_coverage:
                logger.info(
                    "adaptive_download_success",
                    symbol=symbol,
                    days=days,
                    coverage=coverage,
                    rows=len(frame),
                )
                return frame, days, coverage
            else:
                logger.debug(
                    "low_coverage_try_next",
                    symbol=symbol,
                    days=days,
                    coverage=coverage,
                    min_coverage=min_coverage,
                )
        except Exception as e:
            logger.debug("adaptive_download_failed", symbol=symbol, days=days, error=str(e))
            continue
    
    # All attempts failed
    return None, None, None


def download_and_upload(
    symbols: List[str],
    days: int = 150,
    exchange_id: str = "binance",
    timeframe: str = "1m",
    dropbox_token: Optional[str] = None,
    use_adaptive: bool = True,
) -> None:
    """Download with market validation, age checks, and adaptive windows."""
    settings = EngineSettings.load(environment=os.getenv("HURACAN_ENV", "local"))
    
    # Initialize Dropbox (optional)
    dropbox_sync = None
    if dropbox_token or settings.dropbox.access_token:
        try:
            dropbox_sync = DropboxSync(
                access_token=dropbox_token or settings.dropbox.access_token,
                app_folder=settings.dropbox.app_folder,
                enabled=True,
                create_dated_folder=False,
            )
            print("✅ Dropbox ready\n")
        except Exception as e:
            print(f"⚠️  Dropbox not available: {e}\n")
    
    # Initialize exchange (Binance only)
    credentials = settings.exchange.credentials.get(exchange_id, {})
    exchange = ExchangeClient(exchange_id, credentials=credentials, sandbox=settings.exchange.sandbox)
    
    # Get active spot markets
    spot_markets = get_active_spot_markets(exchange)
    print(f"📊 Found {len(spot_markets)} active spot markets\n")
    
    # Normalize symbols (apply aliases, filter to spot markets)
    symbols = normalize_symbols(symbols, spot_markets, TICKER_ALIASES)
    print(f"📋 After normalization: {len(symbols)} valid symbols\n")
    
    # Filter by age if using fixed days
    end_at = datetime.now(tz=timezone.utc)
    start_at = end_at - timedelta(days=days)
    
    if not use_adaptive:
        symbols = filter_by_age(symbols, exchange, start_at)
        print(f"📋 After age filtering: {len(symbols)} symbols\n")
    
    # Initialize loader
    quality_suite = DataQualitySuite(coverage_threshold=MIN_COVERAGE)
    loader = CandleDataLoader(
        exchange_client=exchange,
        quality_suite=quality_suite,
        cache_dir=Path("data/candles"),
        fallback_exchanges=[],
    )
    
    print(f"📊 Downloading {len(symbols)} coins")
    print(f"   Exchange: {exchange_id}")
    print(f"   Timeframe: {timeframe}")
    if use_adaptive:
        print(f"   Adaptive days: {ADAPTIVE_DAYS}")
    else:
        print(f"   Days: {days}")
    print(f"   End: {end_at.date()}\n")
    
    downloaded_count = 0
    uploaded_count = 0
    skipped_count = 0
    
    for idx, symbol in enumerate(symbols):
        # Small delay between coins
        if idx > 0:
            time.sleep(0.5)
        
        try:
            print(f"[{idx+1}/{len(symbols)}] 📥 {symbol}...")
            
            # Use adaptive window if enabled
            if use_adaptive:
                frame, used_days, coverage = fetch_with_adaptive_days(
                    symbol,
                    loader,
                    timeframe,
                    end_at,
                    ADAPTIVE_DAYS,
                    MIN_COVERAGE,
                )
                
                if frame is None or used_days is None:
                    logger.warning(
                        "symbol_skipped_low_coverage",
                        symbol=symbol,
                        message="Not enough data after trying all adaptive windows",
                    )
                    print(f"   ⚠️  Skipped: Not enough data (tried {ADAPTIVE_DAYS})")
                    skipped_count += 1
                    continue
                
                # Recreate query with the days that worked
                assert used_days is not None  # Type guard
                start_at = end_at - timedelta(days=used_days)
                query = CandleQuery(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_at=start_at,
                    end_at=end_at,
                )
                print(f"   ✅ {len(frame):,} rows ({used_days} days, {coverage:.1%} coverage)")
            else:
                # Fixed window
                query = CandleQuery(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_at=start_at,
                    end_at=end_at,
                )
                
                frame = loader.load(query, use_cache=True)
                
                if frame.is_empty():
                    print(f"   ⚠️  No data available")
                    skipped_count += 1
                    continue
                
                print(f"   ✅ {len(frame):,} rows")
            
            # Get cache path
            cache_path = loader._cache_path(query)
            if not cache_path.exists():
                print(f"   ⚠️  Cache file not found")
                skipped_count += 1
                continue
            
            downloaded_count += 1
            file_size = cache_path.stat().st_size
            print(f"      ({file_size / 1024 / 1024:.2f} MB)")
            
            # Upload to Dropbox
            if dropbox_sync:
                try:
                    rel_path = cache_path.relative_to(Path("data/candles"))
                    remote_path = f"/{settings.dropbox.app_folder}/data/candles/{rel_path.as_posix()}"
                    
                    # Check if file already exists
                    try:
                        import dropbox  # type: ignore[reportMissingImports]
                        existing = dropbox_sync._dbx.files_get_metadata(remote_path)
                        if isinstance(existing, dropbox.files.FileMetadata):  # type: ignore[misc]
                            print(f"   ⏭️  Already in Dropbox (skipping)")
                            uploaded_count += 1
                            continue
                    except:
                        pass
                    
                    success = dropbox_sync.upload_file(
                        local_path=cache_path,
                        remote_path=remote_path,
                        use_dated_folder=False,
                        overwrite=True,
                    )
                    if success:
                        uploaded_count += 1
                        print(f"   📤 Uploaded to Dropbox: {rel_path.as_posix()}")
                except Exception as e:
                    print(f"   ⚠️  Upload failed: {e}")
            
        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted by user")
            raise
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "429" in error_msg:
                print(f"   ⚠️  Rate limited - waiting 10 seconds...")
                time.sleep(10)
                skipped_count += 1
            elif "not found" in error_msg or "invalid symbol" in error_msg:
                logger.warning("symbol_not_found", symbol=symbol, error=str(e))
                print(f"   ⚠️  Symbol not available: {symbol}")
                skipped_count += 1
            else:
                logger.warning("download_error", symbol=symbol, error=str(e))
                print(f"   ⚠️  Error: {e}")
                skipped_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Summary")
    print(f"{'='*60}")
    print(f"   ✅ Downloaded: {downloaded_count}/{len(symbols)}")
    print(f"   📤 Uploaded: {uploaded_count}/{downloaded_count}")
    print(f"   ⚠️  Skipped: {skipped_count}/{len(symbols)}")
    print(f"{'='*60}\n")
    
    if downloaded_count > 0:
        print(f"✅ Successfully downloaded {downloaded_count} coin(s)")
        if uploaded_count > 0:
            print(f"✅ Successfully uploaded {uploaded_count} file(s) to Dropbox")
            print(f"   Location: /{settings.dropbox.app_folder}/data/candles/\n")


def main():
    parser = argparse.ArgumentParser(description="Simple, fast candle download with market validation")
    parser.add_argument("--symbols", nargs="+", help="Symbols to download")
    parser.add_argument("--all-top20", action="store_true", help="Download top 20 coins")
    parser.add_argument("--days", type=int, default=150, help="Number of days (used if --no-adaptive)")
    parser.add_argument("--no-adaptive", action="store_true", help="Disable adaptive window (use fixed days)")
    parser.add_argument("--timeframe", type=str, default="1m", help="Timeframe")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange ID")
    parser.add_argument("--dropbox-token", type=str, help="Dropbox token")
    
    args = parser.parse_args()
    
    if args.all_top20:
        print("📊 Fetching top 20 coins from Binance by 24h volume...")
        symbols = get_top20_coins_from_binance()
        print(f"✅ Found {len(symbols)} coins:")
        for i, symbol in enumerate(symbols, 1):
            print(f"   {i:2}. {symbol}")
        print()
    elif args.symbols:
        symbols = args.symbols
    else:
        print("❌ Error: Must specify --symbols or --all-top20")
        sys.exit(1)
    
    download_and_upload(
        symbols=symbols,
        days=args.days,
        exchange_id=args.exchange,
        timeframe=args.timeframe,
        dropbox_token=args.dropbox_token,
        use_adaptive=not args.no_adaptive,
    )


if __name__ == "__main__":
    main()
