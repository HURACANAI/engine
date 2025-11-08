"""Data loading interfaces for fetching candle and market data for the Engine."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional, Callable

import polars as pl
import structlog

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
        # Default fallback exchanges (major exchanges with good API limits)
        self._fallback_exchanges = fallback_exchanges or ["coinbasepro", "kraken", "okx", "bybit"]
        self._exchange_credentials = exchange_credentials or {}
        self._fallback_clients: dict[str, ExchangeClient] = {}
        
        # Optional callback for when data is saved (e.g., trigger Dropbox sync)
        self._on_data_saved = on_data_saved

    def load(self, query: CandleQuery, use_cache: bool = True) -> pl.DataFrame:
        cache_path = self._cache_path(query)
        if use_cache and cache_path.exists():
            try:
                frame = pl.read_parquet(cache_path)
                return self._quality.validate(frame, query=query)
            except Exception as e:
                # If cache read fails, fall back to download
                logger.warning("cache_read_failed", path=str(cache_path), error=str(e))
        
        frame = self._download(query)
        if len(frame) > 0:  # Only write if we got data
            try:
                frame.write_parquet(cache_path)
                
                # Update file modification time to ensure it's detected as "recent" by sync
                import os
                os.utime(cache_path, None)  # Update to current time
                
                logger.info(
                    "coin_data_cached",
                    symbol=query.symbol,
                    cache_path=str(cache_path),
                    rows=len(frame),
                    message="Coin data saved - will be synced to Dropbox within 5 minutes",
                )
                
                # Trigger callback if provided (e.g., sync to Dropbox)
                if self._on_data_saved:
                    try:
                        self._on_data_saved(cache_path)
                    except Exception as callback_error:
                        # Callback errors are non-fatal - don't break data loading
                        logger.warning(
                            "on_data_saved_callback_failed",
                            cache_path=str(cache_path),
                            error=str(callback_error),
                        )
            except Exception as e:
                logger.warning("cache_write_failed", path=str(cache_path), error=str(e))
        return frame

    def _download(self, query: CandleQuery, skip_validation: bool = False) -> pl.DataFrame:
        """Download data with parallel multi-exchange support.
        
        Downloads from multiple exchanges simultaneously to speed up the process.
        Since different exchanges are different APIs, we can download in parallel
        without hitting rate limits. Uses the first successful result.
        """
        # Prepare all exchanges to try (primary + fallbacks)
        exchanges_to_try = [
            (self._exchange.exchange_id, self._exchange),
        ]
        
        # Add fallback exchanges
        for fallback_id in self._fallback_exchanges:
            try:
                fallback_client = self._get_fallback_client(fallback_id)
                exchanges_to_try.append((fallback_id, fallback_client))
            except Exception as e:
                logger.warning(
                    "fallback_exchange_init_failed",
                    exchange=fallback_id,
                    error=str(e),
                )
                continue
        
        # Download from all exchanges in parallel
        # Different exchanges = different APIs = no rate limit conflicts
        logger.info(
            "parallel_download_start",
            symbol=query.symbol,
            num_exchanges=len(exchanges_to_try),
            exchanges=[ex_id for ex_id, _ in exchanges_to_try],
        )
        
        with ThreadPoolExecutor(max_workers=len(exchanges_to_try)) as executor:
            # Submit all download tasks
            futures: dict[Future[pl.DataFrame], tuple[str, ExchangeClient]] = {}
            for exchange_id, exchange_client in exchanges_to_try:
                future = executor.submit(
                    self._download_from_exchange,
                    exchange_client,
                    query,
                    skip_validation,
                )
                futures[future] = (exchange_id, exchange_client)
            
            # Use first successful result
            for future in as_completed(futures):
                exchange_id, exchange_client = futures[future]
                try:
                    result = future.result()
                    if result is not None and len(result) > 0:
                        logger.info(
                            "parallel_download_success",
                            symbol=query.symbol,
                            exchange=exchange_id,
                            rows=len(result),
                            message="Using data from this exchange",
                        )
                        # Cancel remaining tasks (optional - they'll finish but we won't use them)
                        return result
                except Exception as e:
                    error_str = str(e).lower()
                    logger.warning(
                        "parallel_download_failed",
                        symbol=query.symbol,
                        exchange=exchange_id,
                        error=str(e),
                    )
                    continue
        
        # If all exchanges failed, raise error
        logger.error(
            "all_exchanges_failed_parallel",
            symbol=query.symbol,
            exchanges=[ex_id for ex_id, _ in exchanges_to_try],
        )
        raise RuntimeError(f"Failed to download {query.symbol} from all exchanges")
    
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
        while since_ms < end_ms:
            batch = exchange_client.fetch_ohlcv(query.symbol, query.timeframe, since=since_ms, limit=1_000)
            if not batch or len(batch) == 0:
                break
            rows.extend(batch)
            # Safely get last timestamp
            last_ts = batch[-1][0] if len(batch) > 0 else None
            if last_ts is None or last_ts == since_ms:
                break
            since_ms = last_ts + 60_000
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
        ).with_columns(
            (pl.col("timestamp") / 1000).cast(pl.Datetime(time_unit="ms", time_zone="UTC")).alias("ts")
        )
        frame = frame.sort("ts")

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
        symbol_safe = query.symbol.replace("/", "-")
        filename = f"{symbol_safe}_{query.timeframe}_{query.start_at:%Y%m%d}_{query.end_at:%Y%m%d}.parquet"
        return self._cache_dir / filename


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
