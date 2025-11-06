"""Data loading interfaces for fetching candle and market data for the Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

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
    ) -> None:
        self._exchange = exchange_client
        self._cache_dir = Path(cache_dir or Path.cwd() / "data" / "candles")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._quality = quality_suite or DataQualitySuite()
        
        # Multi-exchange fallback support
        # Default fallback exchanges (major exchanges with good API limits)
        self._fallback_exchanges = fallback_exchanges or ["coinbasepro", "kraken", "okx", "bybit"]
        self._exchange_credentials = exchange_credentials or {}
        self._fallback_clients: dict[str, ExchangeClient] = {}

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
            except Exception as e:
                logger.warning("cache_write_failed", path=str(cache_path), error=str(e))
        return frame

    def _download(self, query: CandleQuery, skip_validation: bool = False) -> pl.DataFrame:
        """Download data with multi-exchange fallback."""
        # Try primary exchange first
        try:
            return self._download_from_exchange(self._exchange, query, skip_validation)
        except Exception as e:
            error_str = str(e).lower()
            # Check if it's a rate limit error
            if any(keyword in error_str for keyword in ["429", "rate limit", "too many requests", "ddos"]):
                logger.warning(
                    "rate_limit_on_primary",
                    exchange=self._exchange.exchange_id,
                    symbol=query.symbol,
                    error=str(e),
                    message="Trying fallback exchanges",
                )
                # Try fallback exchanges
                for fallback_id in self._fallback_exchanges:
                    try:
                        fallback_client = self._get_fallback_client(fallback_id)
                        logger.info(
                            "trying_fallback_exchange",
                            exchange=fallback_id,
                            symbol=query.symbol,
                        )
                        return self._download_from_exchange(fallback_client, query, skip_validation)
                    except Exception as fallback_error:
                        logger.warning(
                            "fallback_exchange_failed",
                            exchange=fallback_id,
                            symbol=query.symbol,
                            error=str(fallback_error),
                        )
                        continue
                # If all fallbacks failed, raise original error
                logger.error(
                    "all_exchanges_failed",
                    symbol=query.symbol,
                    primary_exchange=self._exchange.exchange_id,
                    fallback_exchanges=self._fallback_exchanges,
                    error=str(e),
                )
            raise
    
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
