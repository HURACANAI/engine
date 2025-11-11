"""
Dynamic Coin Selection System

Selects coins based on daily liquidity ranking, spread filtering, and volume
thresholds. Supports 400+ coins with dynamic daily updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set

import polars as pl
import structlog

logger = structlog.get_logger(__name__)


class RankingMethod(str, Enum):
    """Coin ranking methods."""
    LIQUIDITY_SCORE = "liquidity_score"  # Combined liquidity metric
    VOLUME = "volume"  # Pure volume ranking
    SPREAD = "spread"  # Lowest spread first


@dataclass
class CoinSelectionConfig:
    """Configuration for coin selection."""
    min_daily_volume_usd: float = 1_000_000.0
    max_spread_bps: float = 8.0
    min_age_days: int = 30
    ranking_method: RankingMethod = RankingMethod.LIQUIDITY_SCORE
    update_frequency: str = "daily"  # "daily", "hourly", "realtime"
    max_coins: Optional[int] = None  # None = no limit, configurable throttle


@dataclass
class CoinMetrics:
    """Metrics for a single coin."""
    symbol: str
    daily_volume_usd: float
    spread_bps: float
    age_days: int
    liquidity_score: float
    rank: int
    last_updated: datetime


class DynamicCoinSelector:
    """
    Dynamic coin selection based on liquidity ranking.
    
    Features:
    - Fetches all available Binance coins (400+)
    - Ranks by liquidity metrics
    - Filters by spread, volume, age
    - Daily ranking updates
    - Configurable limits (no hard limits)
    """
    
    def __init__(
        self,
        config: CoinSelectionConfig,
        exchange_client: Any,  # ExchangeClient type
        metadata_loader: Any,  # MarketMetadataLoader type
    ) -> None:
        """
        Initialize coin selector.
        
        Args:
            config: Coin selection configuration
            exchange_client: Exchange client for fetching markets
            metadata_loader: Metadata loader for liquidity data
        """
        self.config = config
        self.exchange_client = exchange_client
        self.metadata_loader = metadata_loader
        
        # Cache for coin metrics
        self.coin_metrics_cache: Dict[str, CoinMetrics] = {}
        self.last_update: Optional[datetime] = None
        
        logger.info(
            "coin_selector_initialized",
            min_volume_usd=config.min_daily_volume_usd,
            max_spread_bps=config.max_spread_bps,
            ranking_method=config.ranking_method.value,
        )
    
    def select_coins(
        self,
        force_refresh: bool = False,
    ) -> List[str]:
        """
        Select coins based on current ranking.
        
        Args:
            force_refresh: Force refresh of metrics even if recently updated
        
        Returns:
            List of selected coin symbols
        """
        # Check if update is needed
        if force_refresh or self._should_update():
            self._update_coin_metrics()
        
        # Filter and rank coins
        filtered_coins = self._filter_coins()
        ranked_coins = self._rank_coins(filtered_coins)
        
        # Apply max_coins limit if configured
        if self.config.max_coins is not None:
            ranked_coins = ranked_coins[:self.config.max_coins]
        
        symbols = [coin.symbol for coin in ranked_coins]
        
        logger.info(
            "coins_selected",
            num_selected=len(symbols),
            total_available=len(self.coin_metrics_cache),
            ranking_method=self.config.ranking_method.value,
        )
        
        return symbols
    
    def _should_update(self) -> bool:
        """Check if coin metrics should be updated."""
        if self.last_update is None:
            return True
        
        if self.config.update_frequency == "daily":
            # Update once per day
            return (datetime.now(timezone.utc) - self.last_update).days >= 1
        elif self.config.update_frequency == "hourly":
            # Update once per hour
            return (datetime.now(timezone.utc) - self.last_update).hours >= 1
        else:
            # Realtime - always update
            return True
    
    def _update_coin_metrics(self) -> None:
        """Update coin metrics from exchange and metadata."""
        logger.info("updating_coin_metrics")
        
        # Fetch all markets
        markets = self.exchange_client.fetch_markets()
        
        # Get liquidity snapshot
        symbols = [m.symbol for m in markets.values() if m.active]
        liquidity_data = self.metadata_loader.liquidity_snapshot(symbols)
        fee_data = self.metadata_loader.fee_schedule(symbols)
        
        # Merge data
        merged = liquidity_data.join(fee_data, on="symbol", how="inner")
        
        # Calculate metrics for each coin
        coin_metrics = []
        
        for row in merged.iter_rows(named=True):
            symbol = row["symbol"]
            daily_volume = row.get("quote_volume", 0.0)
            spread_bps = row.get("spread_bps", 999.0)
            
            # Calculate age (days since listing)
            # This would come from exchange metadata
            age_days = row.get("age_days", 365)  # Default to 1 year if unknown
            
            # Calculate liquidity score
            # Higher volume and lower spread = higher score
            volume_score = min(1.0, daily_volume / (10 * self.config.min_daily_volume_usd))
            spread_score = max(0.0, 1.0 - (spread_bps / (2 * self.config.max_spread_bps)))
            liquidity_score = (volume_score * 0.7 + spread_score * 0.3)
            
            metrics = CoinMetrics(
                symbol=symbol,
                daily_volume_usd=daily_volume,
                spread_bps=spread_bps,
                age_days=age_days,
                liquidity_score=liquidity_score,
                rank=0,  # Will be set during ranking
                last_updated=datetime.now(timezone.utc),
            )
            
            coin_metrics.append(metrics)
        
        # Update cache
        self.coin_metrics_cache = {m.symbol: m for m in coin_metrics}
        self.last_update = datetime.now(timezone.utc)
        
        logger.info(
            "coin_metrics_updated",
            num_coins=len(self.coin_metrics_cache),
        )
    
    def _filter_coins(self) -> List[CoinMetrics]:
        """Filter coins based on criteria."""
        filtered = []
        
        for metrics in self.coin_metrics_cache.values():
            # Volume filter
            if metrics.daily_volume_usd < self.config.min_daily_volume_usd:
                continue
            
            # Spread filter
            if metrics.spread_bps > self.config.max_spread_bps:
                continue
            
            # Age filter
            if metrics.age_days < self.config.min_age_days:
                continue
            
            filtered.append(metrics)
        
        logger.debug(
            "coins_filtered",
            before_filter=len(self.coin_metrics_cache),
            after_filter=len(filtered),
        )
        
        return filtered
    
    def _rank_coins(self, coins: List[CoinMetrics]) -> List[CoinMetrics]:
        """Rank coins based on ranking method."""
        if self.config.ranking_method == RankingMethod.LIQUIDITY_SCORE:
            # Sort by liquidity score (descending)
            ranked = sorted(coins, key=lambda c: c.liquidity_score, reverse=True)
        elif self.config.ranking_method == RankingMethod.VOLUME:
            # Sort by volume (descending)
            ranked = sorted(coins, key=lambda c: c.daily_volume_usd, reverse=True)
        elif self.config.ranking_method == RankingMethod.SPREAD:
            # Sort by spread (ascending - lower is better)
            ranked = sorted(coins, key=lambda c: c.spread_bps)
        else:
            # Default to liquidity score
            ranked = sorted(coins, key=lambda c: c.liquidity_score, reverse=True)
        
        # Assign ranks
        for rank, coin in enumerate(ranked, start=1):
            coin.rank = rank
        
        return ranked
    
    def get_coin_metrics(self, symbol: str) -> Optional[CoinMetrics]:
        """Get metrics for a specific coin."""
        return self.coin_metrics_cache.get(symbol)
    
    def get_all_metrics(self) -> Dict[str, CoinMetrics]:
        """Get all coin metrics."""
        return self.coin_metrics_cache.copy()
    
    def get_top_coins(self, n: int) -> List[str]:
        """Get top N coins by current ranking."""
        selected = self.select_coins()
        return selected[:n]

