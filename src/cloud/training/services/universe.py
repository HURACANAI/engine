"""Universe selection service for liquidity-based coin filtering."""

from __future__ import annotations

from typing import List

import polars as pl  # type: ignore[reportMissingImports]

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
        """Get candidate symbols, preferring USD-pegged stablecoins (USDC, USDT, USD).
        
        Priority order:
        1. USDC (USD Coin - most transparent USD peg)
        2. USDT (Tether - most liquid)
        3. USD (direct USD pairs if available)
        4. BUSD, GBP, EUR (other stable/quoted currencies)
        """
        markets = self._exchange.fetch_markets()
        # Prefer USD-pegged stablecoins, with USDC first (most transparent)
        preferred_quotes = {"USDC", "USDT", "USD", "BUSD", "GBP", "EUR"}
        symbols: List[str] = []
        
        # Track symbols by base coin to prefer better quote currencies
        symbols_by_base: dict[str, str] = {}
        
        for market in markets.values():
            if not market.active:
                continue
            if not market.quote or market.quote not in preferred_quotes:
                continue
            base = (market.base or "").upper()
            if base in {"USDT", "USDC", "BUSD", "DAI", "USD"}:
                continue
            
            symbol = market.symbol
            quote = market.quote
            
            # Prefer USDC > USDT > USD > others for same base coin
            if base not in symbols_by_base:
                symbols_by_base[base] = symbol
            else:
                existing_symbol = symbols_by_base[base]
                existing_quote = existing_symbol.split("/")[-1] if "/" in existing_symbol else ""
                
                # Priority: USDC > USDT > USD > others
                quote_priority = {"USDC": 1, "USDT": 2, "USD": 3}.get(quote, 4)
                existing_priority = {"USDC": 1, "USDT": 2, "USD": 3}.get(existing_quote, 4)
                
                if quote_priority < existing_priority:
                    symbols_by_base[base] = symbol
        
        return list(symbols_by_base.values())
