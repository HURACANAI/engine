"""Data loading interfaces for fetching candle and market data for the Engine."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional, Callable

import polars as pl  # type: ignore[reportMissingImports]
import structlog  # type: ignore[reportMissingImports]

from ..services.exchange import ExchangeClient
from .quality_checks import DataQualitySuite

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class CandleQuery:
    symbol: str
    timeframe: str = "1m"
    start_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc) - timedelta(days=180))
    end_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


class CandleDataLoader:
    """Retrieves and caches historical OHLCV data through the exchange adapter.
    
    Supports multi-exchange fallback to avoid rate limits:
    - Tries primary exchange (usually Binance) first
    - Falls back to other exchanges if rate limited
    - Data is the same across exchanges for the same symbol
    """

    def __init__(
        self,
        exchange_client: ExchangeClient,
        cache_dir: Optional[Path] = None,
        quality_suite: Optional[DataQualitySuite] = None,
        fallback_exchanges: Optional[list[str]] = None,
        exchange_credentials: Optional[dict] = None,
        on_data_saved: Optional[Callable[[Path], None]] = None,
    ) -> None:
        """Initialize CandleDataLoader.
        
        Args:
            exchange_client: Exchange client for downloading data
            cache_dir: Directory to cache downloaded data
            quality_suite: Data quality validation suite
            fallback_exchanges: List of fallback exchange IDs
            exchange_credentials: Credentials for fallback exchanges
            on_data_saved: Optional callback function called after data is saved to cache.
                          Receives the cache_path as argument. Useful for triggering Dropbox sync.
        """
        self._exchange = exchange_client
        self._cache_dir = Path(cache_dir or Path.cwd() / "data" / "candles")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._quality = quality_suite or DataQualitySuite()
        
        # Multi-exchange fallback support
        # DISABLED by default - Binance is fast and reliable, parallel exchanges cause errors
        # Set fallback_exchanges=[] to disable, or pass specific exchanges if needed
        self._fallback_exchanges = fallback_exchanges if fallback_exchanges is not None else []
        self._exchange_credentials = exchange_credentials or {}
        self._fallback_clients: dict[str, ExchangeClient] = {}
        
        # Optional callback for when data is saved (e.g., trigger Dropbox sync)
        self._on_data_saved = on_data_saved

    def load(self, query: CandleQuery, use_cache: bool = True) -> pl.DataFrame:
        """Load data from cache (Dropbox/local) or download from exchange.

        ENHANCED: Always fetches the most recent data up to NOW, then tops up any missing gaps.

        Steps:
        1. Check if data exists in cache (Dropbox/local)
        2. If cached, check date range and determine missing data
        3. Download missing data from exchange (top-up both historical gaps AND recent data)
        4. Merge cached data with new data
        5. Update cache and upload to Dropbox

        Key improvement: Always downloads data up to the current time (NOW) to ensure
        the bot has the most recent market data for training.
        """
        cache_path = self._cache_path(query)
        cached_data = None
        cache_source = None

        # CRITICAL: Always update query.end_at to NOW to ensure we get the most recent data
        # This ensures the bot never trains on stale data
        now = datetime.now(tz=timezone.utc)
        if query.end_at < now:
            # Update query to fetch data up to NOW
            query = CandleQuery(
                symbol=query.symbol,
                timeframe=query.timeframe,
                start_at=query.start_at,
                end_at=now,  # Always fetch up to NOW
            )
            print(f"⏰ [{query.symbol}] Updated end time to NOW: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")

        # Step 1: Check if data exists in cache
        if use_cache and cache_path.exists():
            try:
                print(f"📦 [{query.symbol}] Checking local cache: {cache_path.name}")
                cached_data = pl.read_parquet(cache_path)
                cache_source = "local"
                
                # CRITICAL: Check if cached data has corrupted timestamps (1970 dates)
                if "ts" in cached_data.columns and len(cached_data) > 0:
                    ts_min = cached_data["ts"].min()
                    ts_max = cached_data["ts"].max()
                    
                    # Check if dates are reasonable (after 2020)
                    if ts_min.year < 2020 or ts_max.year < 2020:
                        print(f"⚠️  [{query.symbol}] Cache has corrupted timestamps (1970 dates) - re-downloading")
                        logger.warning(
                            "cached_data_has_corrupted_timestamps",
                            cache_path=str(cache_path),
                            ts_min=ts_min.isoformat(),
                            ts_max=ts_max.isoformat(),
                            message="Cached data has corrupted timestamps (1970 dates) - re-downloading to fix",
                        )
                        # Delete corrupted cache and re-download
                        try:
                            cache_path.unlink()
                            logger.info("corrupted_cache_deleted", cache_path=str(cache_path))
                            cached_data = None
                            cache_source = None
                        except Exception as del_error:
                            logger.warning("cache_delete_failed", error=str(del_error))
                    else:
                        # Cached data is valid - check date range
                        cache_start = ts_min
                        cache_end = ts_max
                        requested_start = query.start_at
                        requested_end = query.end_at
                        
                        print(f"✅ [{query.symbol}] Found cached data: {cache_start.date()} to {cache_end.date()} ({len(cached_data):,} rows)")
                        print(f"📅 [{query.symbol}] Requested range: {requested_start.date()} to {requested_end.date()}")
                        
                        # Check if we need to download missing data
                        needs_topup = False
                        topup_start = requested_start
                        topup_end = requested_end
                        
                        if cache_start > requested_start:
                            needs_topup = True
                            topup_end = cache_start - timedelta(minutes=1)
                            print(f"⬇️  [{query.symbol}] Need to download earlier data: {requested_start.date()} to {topup_end.date()}")
                        
                        if cache_end < requested_end:
                            needs_topup = True
                            topup_start = cache_end + timedelta(minutes=1)
                            print(f"⬇️  [{query.symbol}] Need to download newer data: {topup_start.date()} to {requested_end.date()}")
                            # Show how many hours/days of new data we're fetching
                            time_diff = requested_end - cache_end
                            if time_diff.days > 0:
                                print(f"📊 [{query.symbol}] Fetching {time_diff.days} days and {time_diff.seconds // 3600} hours of new data")
                        
                        if not needs_topup:
                            # Cache has all requested data
                            print(f"✅ [{query.symbol}] Cache has all requested data - using cached data")
                            validated = self._quality.validate(cached_data, query=query)
                            return validated
                        
                        # We need to top-up from exchange
                        print(f"🔄 [{query.symbol}] Top-up needed: Downloading missing data from {self._exchange.exchange_id}")
                        
                        # Download missing data
                        topup_query = CandleQuery(
                            symbol=query.symbol,
                            timeframe=query.timeframe,
                            start_at=topup_start,
                            end_at=topup_end,
                        )
                        topup_data = self._download(topup_query, skip_validation=True)
                        
                        if len(topup_data) > 0:
                            print(f"✅ [{query.symbol}] Downloaded {len(topup_data):,} new rows from {self._exchange.exchange_id}")
                            
                            # Merge cached data with new data
                            if cache_start > requested_start:
                                # New data comes before cached data
                                merged = pl.concat([topup_data, cached_data])
                            else:
                                # New data comes after cached data
                                merged = pl.concat([cached_data, topup_data])
                            
                            # Remove duplicates and sort
                            merged = merged.unique(subset=["ts"], keep="first").sort("ts")
                            
                            print(f"✅ [{query.symbol}] Merged data: {len(merged):,} total rows ({len(cached_data):,} cached + {len(topup_data):,} new)")
                            
                            # Update cache
                            try:
                                merged.write_parquet(cache_path)
                                import os
                                os.utime(cache_path, None)
                                print(f"💾 [{query.symbol}] Updated cache: {cache_path.name}")
                                
                                # Trigger Dropbox upload
                                if self._on_data_saved:
                                    try:
                                        self._on_data_saved(cache_path)
                                        print(f"☁️  [{query.symbol}] Uploaded to Dropbox: {cache_path.name}")
                                    except Exception as callback_error:
                                        print(f"⚠️  [{query.symbol}] Dropbox upload failed: {callback_error}")
                                        logger.warning("on_data_saved_callback_failed", error=str(callback_error))
                                
                                return self._quality.validate(merged, query=query)
                            except Exception as e:
                                print(f"❌ [{query.symbol}] Failed to update cache: {e}")
                                logger.warning("cache_write_failed", path=str(cache_path), error=str(e))
                        else:
                            # Top-up failed, use cached data
                            print(f"⚠️  [{query.symbol}] Top-up failed - using cached data only")
                            return self._quality.validate(cached_data, query=query)
                else:
                    # No ts column or empty frame
                    print(f"⚠️  [{query.symbol}] Cache has no timestamp column - re-downloading")
                    cached_data = None
                    cache_source = None
            except Exception as e:
                # If cache read fails, fall back to download
                print(f"⚠️  [{query.symbol}] Cache read failed: {e} - downloading from exchange")
                logger.warning("cache_read_failed", path=str(cache_path), error=str(e))
                cached_data = None
                cache_source = None
        
        # Step 2: No cache or cache invalid - download from exchange
        if cached_data is None:
            print(f"⬇️  [{query.symbol}] Downloading from {self._exchange.exchange_id}: {query.start_at.date()} to {query.end_at.date()}")
            frame = self._download(query)
            
            if len(frame) > 0:
                print(f"✅ [{query.symbol}] Downloaded {len(frame):,} rows from {self._exchange.exchange_id}")
                
                # Save to cache
                try:
                    frame.write_parquet(cache_path)
                    import os
                    os.utime(cache_path, None)
                    print(f"💾 [{query.symbol}] Saved to cache: {cache_path.name}")
                    
                    # Trigger Dropbox upload
                    if self._on_data_saved:
                        try:
                            self._on_data_saved(cache_path)
                            print(f"☁️  [{query.symbol}] Uploaded to Dropbox: {cache_path.name}")
                        except Exception as callback_error:
                            print(f"⚠️  [{query.symbol}] Dropbox upload failed: {callback_error}")
                            logger.warning("on_data_saved_callback_failed", error=str(callback_error))
                except Exception as e:
                    print(f"❌ [{query.symbol}] Failed to save cache: {e}")
                    logger.warning("cache_write_failed", path=str(cache_path), error=str(e))
            else:
                print(f"⚠️  [{query.symbol}] No data downloaded from {self._exchange.exchange_id}")
        
        return frame

    def _download(self, query: CandleQuery, skip_validation: bool = False) -> pl.DataFrame:
        """Download data from primary exchange only (fastest and most reliable).
        
        Simplified approach: Use only the primary exchange (usually Binance).
        This avoids parallel exchange errors and is still very fast.
        """
        # Use only primary exchange - fastest and most reliable
        # Binance has excellent API limits and data quality
        logger.info(
            "download_start",
            symbol=query.symbol,
            exchange=self._exchange.exchange_id,
        )
        
        try:
            result = self._download_from_exchange(
                self._exchange,
                query,
                skip_validation,
            )
            if result is not None and len(result) > 0:
                logger.info(
                    "download_success",
                    symbol=query.symbol,
                    exchange=self._exchange.exchange_id,
                    rows=len(result),
                )
                return result
        except Exception as e:
            logger.warning(
                "download_failed",
                symbol=query.symbol,
                exchange=self._exchange.exchange_id,
                error=str(e),
            )
            raise
        
        raise RuntimeError(f"Failed to download {query.symbol} from {self._exchange.exchange_id}")
    
    def _download_from_exchange(
        self,
        exchange_client: ExchangeClient,
        query: CandleQuery,
        skip_validation: bool = False,
    ) -> pl.DataFrame:
        """Download data from a specific exchange."""
        rows = []
        since_ms = int(query.start_at.timestamp() * 1_000)
        end_ms = int(query.end_at.timestamp() * 1_000)
        # Use maximum batch size for faster downloads
        # Binance supports up to 1000 candles per request, which is already max
        # Add small delay between batches to respect rate limits
        import time
        batch_count = 0
        start_ms = since_ms  # Track start for progress calculation
        
        while since_ms < end_ms:
            batch = exchange_client.fetch_ohlcv(query.symbol, query.timeframe, since=since_ms, limit=1_000)
            if not batch or len(batch) == 0:
                break
            rows.extend(batch)
            batch_count += 1
            
            # Progress update every 10 batches
            if batch_count % 10 == 0:
                progress_pct = min(100, int((since_ms - start_ms) / (end_ms - start_ms) * 100)) if (end_ms - start_ms) > 0 else 0
                print(f"  ⬇️  [{query.symbol}] Downloading... {batch_count} batches, {len(rows):,} rows ({progress_pct}%)", end='\r')
            
            # Safely get last timestamp
            last_ts = batch[-1][0] if len(batch) > 0 else None
            if last_ts is None or last_ts == since_ms:
                break
            since_ms = last_ts + 60_000
            # Small delay between batches to avoid rate limits (50ms)
            # Binance allows 40 requests/second, so 50ms = ~20 requests/second is safe
            time.sleep(0.05)
        
        print(f"  ✅ [{query.symbol}] Download complete: {batch_count} batches, {len(rows):,} rows")
        if not rows:
            # Return empty DataFrame with correct schema
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Int64,
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Float64,
                    "ts": pl.Datetime(time_unit="ms", time_zone="UTC"),
                }
            )
        
        # Fix polars warning by explicitly specifying row orientation
        # Exchange returns timestamps in milliseconds since Unix epoch
        frame = pl.DataFrame(
            rows,
            schema=[
                ("timestamp", pl.Int64),
                ("open", pl.Float64),
                ("high", pl.Float64),
                ("low", pl.Float64),
                ("close", pl.Float64),
                ("volume", pl.Float64),
            ],
            orient="row",  # Explicitly specify row orientation to avoid warning
        ).with_columns(
            # CRITICAL FIX: Exchange timestamps are in milliseconds, so use time_unit="ms" directly
            # Do NOT divide by 1000 - that was causing 1970 dates (interpreting seconds as milliseconds)
            pl.col("timestamp").cast(pl.Datetime(time_unit="ms", time_zone="UTC")).alias("ts")
        )
        frame = frame.sort("ts")
        
        # Validate timestamp conversion - dates should be reasonable (not 1970)
        if len(frame) > 0:
            ts_min = frame["ts"].min()
            ts_max = frame["ts"].max()
            # Check if dates are reasonable (after 2020, before 2030)
            if ts_min.year < 2020 or ts_max.year > 2030:
                logger.error(
                    "timestamp_conversion_error",
                    ts_min=ts_min.isoformat(),
                    ts_max=ts_max.isoformat(),
                    message=f"Timestamp conversion produced invalid dates. Min: {ts_min}, Max: {ts_max}",
                )
                # Try alternative conversion: maybe timestamps are in seconds?
                # But first log the actual timestamp values for debugging
                sample_timestamps = frame.select("timestamp").head(5)["timestamp"].to_list()
                logger.debug(
                    "timestamp_debug",
                    sample_raw_timestamps=sample_timestamps,
                    message="Sample raw timestamp values from exchange",
                )

        # Only validate if not skipping and quality suite is available
        if not skip_validation and self._quality is not None:
            return self._quality.validate(frame, query=query)
        return frame
    
    def _get_fallback_client(self, exchange_id: str) -> ExchangeClient:
        """Get or create a fallback exchange client."""
        if exchange_id not in self._fallback_clients:
            credentials = self._exchange_credentials.get(exchange_id, {})
            self._fallback_clients[exchange_id] = ExchangeClient(
                exchange_id=exchange_id,
                credentials=credentials,
                sandbox=False,
                load_markets=True,
            )
        return self._fallback_clients[exchange_id]

    def _cache_path(self, query: CandleQuery) -> Path:
        """Generate cache path with coin-specific folder structure.

        ENHANCED: First checks for existing cache files for this symbol/timeframe combo,
        and reuses them if found. This allows the cache to grow organically with top-ups
        rather than creating new files with different date ranges.

        Structure: data/candles/{COIN}/{SYMBOL}_{TIMEFRAME}_{START}_{END}.parquet
        Example: data/candles/BTC/BTC-USDT_1m_20250611_20251108.parquet
        """
        # Normalize symbol (remove futures suffix if present)
        normalized_symbol = query.symbol.split(":")[0] if ":" in query.symbol else query.symbol
        symbol_safe = normalized_symbol.replace("/", "-")

        # Extract base coin (e.g., "BTC" from "BTC/USDT")
        base_coin = normalized_symbol.split("/")[0]

        # Create coin-specific folder
        coin_dir = self._cache_dir / base_coin
        coin_dir.mkdir(parents=True, exist_ok=True)

        # ENHANCED: Check if there's already a cache file for this symbol/timeframe
        # Look for existing files with pattern: {SYMBOL}_{TIMEFRAME}_*.parquet
        existing_pattern = f"{symbol_safe}_{query.timeframe}_*.parquet"
        existing_files = list(coin_dir.glob(existing_pattern))

        if existing_files:
            # Use the most recently modified cache file
            # This allows us to keep updating the same file rather than creating new ones
            most_recent = max(existing_files, key=lambda p: p.stat().st_mtime)
            print(f"♻️  [{query.symbol}] Found existing cache file: {most_recent.name}")
            return most_recent

        # No existing cache - generate new filename with current date range
        filename = f"{symbol_safe}_{query.timeframe}_{query.start_at:%Y%m%d}_{query.end_at:%Y%m%d}.parquet"
        return coin_dir / filename


class MarketMetadataLoader:
    """Gathers fee, spread, and liquidity metadata for downstream cost modeling."""

    def __init__(self, exchange_client: ExchangeClient) -> None:
        self._exchange = exchange_client

    def fee_schedule(self, symbols: Iterable[str]) -> pl.DataFrame:
        markets = self._exchange.fetch_markets()
        records = []
        for symbol in symbols:
            market = markets.get(symbol)
            if not market:
                continue
            records.append(
                {
                    "symbol": symbol,
                    "maker_fee_bps": market.maker * 10_000,
                    "taker_fee_bps": market.taker * 10_000,
                    "active": market.active,
                }
            )
        return pl.DataFrame(records)

    def liquidity_snapshot(self, symbols: Iterable[str]) -> pl.DataFrame:
        tickers = self._exchange.fetch_tickers(symbols)
        records = []
        for symbol, info in tickers.items():
            bid = info.get("bid") or 0.0
            ask = info.get("ask") or 0.0
            spread = ((ask - bid) / ask) * 10_000 if ask else 0.0
            records.append(
                {
                    "symbol": symbol,
                    "bid": bid,
                    "ask": ask,
                    "spread_bps": spread,
                    "base_volume": info.get("baseVolume") or 0.0,
                    "quote_volume": info.get("quoteVolume") or 0.0,
                }
            )
        return pl.DataFrame(records)
