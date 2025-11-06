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
    """Retrieves and caches historical OHLCV data through the exchange adapter."""

    def __init__(
        self,
        exchange_client: ExchangeClient,
        cache_dir: Optional[Path] = None,
        quality_suite: Optional[DataQualitySuite] = None,
    ) -> None:
        self._exchange = exchange_client
        self._cache_dir = Path(cache_dir or Path.cwd() / "data" / "candles")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._quality = quality_suite or DataQualitySuite()

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
        rows = []
        since_ms = int(query.start_at.timestamp() * 1_000)
        end_ms = int(query.end_at.timestamp() * 1_000)
        while since_ms < end_ms:
            batch = self._exchange.fetch_ohlcv(query.symbol, query.timeframe, since=since_ms, limit=1_000)
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
