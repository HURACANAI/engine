"""Universe selection service for liquidity-based coin filtering."""

from __future__ import annotations

from typing import List

import polars as pl

from ..config.settings import UniverseSettings
from ..datasets.data_loader import MarketMetadataLoader
from .exchange import ExchangeClient


class UniverseSelector:
    """Constructs the daily coin universe prioritising Binance while remaining exchange-agnostic."""

    def __init__(
        self,
        exchange_client: ExchangeClient,
        metadata_loader: MarketMetadataLoader,
        settings: UniverseSettings,
    ) -> None:
        self._exchange = exchange_client
        self._metadata_loader = metadata_loader
        self._settings = settings

    def select(self) -> pl.DataFrame:
        candidates = self._candidate_symbols()
        liquidity = self._metadata_loader.liquidity_snapshot(candidates)
        fees = self._metadata_loader.fee_schedule(candidates)
        merged = liquidity.join(fees, on="symbol", how="inner")
        filtered = merged.filter(
            (pl.col("spread_bps") <= self._settings.max_spread_bps)
            & (pl.col("quote_volume") >= self._settings.liquidity_threshold_adv_gbp)
        )
        ranked = filtered.sort(pl.col("quote_volume"), descending=True).head(self._settings.target_size)
        return ranked.with_columns(pl.Series("rank", range(1, len(ranked) + 1)))

    def _candidate_symbols(self) -> List[str]:
        markets = self._exchange.fetch_markets()
        preferred_quotes = {"USDT", "USDC", "BUSD", "GBP", "EUR"}
        symbols: List[str] = []
        for market in markets.values():
            if not market.active:
                continue
            if not market.quote or market.quote not in preferred_quotes:
                continue
            base = (market.base or "").upper()
            if base in {"USDT", "USDC", "BUSD", "DAI"}:
                continue
            symbols.append(market.symbol)
        return symbols
