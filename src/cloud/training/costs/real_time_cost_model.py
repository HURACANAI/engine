"""
Real-Time Cost Model for Scalable Architecture

Tracks real-time spread, fee, and funding data per symbol.
Calculates edge-after-cost and ranks coins by cost efficiency.

Author: Huracan Engine Team
Date: 2025-01-27
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class CostSource(Enum):
    """Cost data source."""
    EXCHANGE_WEBSOCKET = "exchange_websocket"
    EXCHANGE_API = "exchange_api"
    CACHED = "cached"


@dataclass
class CostData:
    """Cost data for a symbol."""
    symbol: str
    spread_bps: float
    maker_fee_bps: float
    taker_fee_bps: float
    funding_rate_bps: float  # Per 8 hours, annualized
    last_update: float = field(default_factory=time.time)
    source: CostSource = CostSource.CACHED


@dataclass
class CostRanking:
    """Cost ranking for a symbol."""
    symbol: str
    total_cost_bps: float
    edge_after_cost_bps: float
    rank: int
    cost_efficiency: float  # Higher is better


class SpreadTracker:
    """Tracks real-time spreads from orderbook data."""
    
    def __init__(self):
        """Initialize spread tracker."""
        self.spreads: Dict[str, float] = {}  # symbol -> spread_bps
        self.last_update: Dict[str, float] = {}
        logger.info("spread_tracker_initialized")
    
    def update_spread(self, symbol: str, bid: float, ask: float) -> None:
        """
        Update spread for a symbol.
        
        Args:
            symbol: Trading symbol
            bid: Bid price
            ask: Ask price
        """
        if bid > 0 and ask > 0:
            mid_price = (bid + ask) / 2.0
            spread = ask - bid
            spread_bps = (spread / mid_price) * 10000.0  # Convert to basis points
            
            self.spreads[symbol] = spread_bps
            self.last_update[symbol] = time.time()
            
            logger.debug(
                "spread_updated",
                symbol=symbol,
                spread_bps=spread_bps,
            )
    
    def get_spread(self, symbol: str) -> float:
        """Get spread for a symbol."""
        return self.spreads.get(symbol, 0.0)
    
    def is_stale(self, symbol: str, max_age_seconds: int = 300) -> bool:
        """Check if spread data is stale."""
        if symbol not in self.last_update:
            return True
        
        age = time.time() - self.last_update[symbol]
        return age > max_age_seconds


class FeeTracker:
    """Tracks maker and taker fees from exchange APIs."""
    
    def __init__(self, default_maker_bps: float = 2.0, default_taker_bps: float = 5.0):
        """
        Initialize fee tracker.
        
        Args:
            default_maker_bps: Default maker fee in basis points
            default_taker_bps: Default taker fee in basis points
        """
        self.maker_fees: Dict[str, float] = {}  # symbol -> maker_fee_bps
        self.taker_fees: Dict[str, float] = {}  # symbol -> taker_fee_bps
        self.default_maker_bps = default_maker_bps
        self.default_taker_bps = default_taker_bps
        self.last_update: Dict[str, float] = {}
        logger.info(
            "fee_tracker_initialized",
            default_maker=default_maker_bps,
            default_taker=default_taker_bps,
        )
    
    def update_fees(self, symbol: str, maker_bps: float, taker_bps: float) -> None:
        """
        Update fees for a symbol.
        
        Args:
            symbol: Trading symbol
            maker_bps: Maker fee in basis points
            taker_bps: Taker fee in basis points
        """
        self.maker_fees[symbol] = maker_bps
        self.taker_fees[symbol] = taker_bps
        self.last_update[symbol] = time.time()
        
        logger.debug(
            "fees_updated",
            symbol=symbol,
            maker_bps=maker_bps,
            taker_bps=taker_bps,
        )
    
    def get_maker_fee(self, symbol: str) -> float:
        """Get maker fee for a symbol."""
        return self.maker_fees.get(symbol, self.default_maker_bps)
    
    def get_taker_fee(self, symbol: str) -> float:
        """Get taker fee for a symbol."""
        return self.taker_fees.get(symbol, self.default_taker_bps)
    
    def is_stale(self, symbol: str, max_age_seconds: int = 3600) -> bool:
        """Check if fee data is stale."""
        if symbol not in self.last_update:
            return True
        
        age = time.time() - self.last_update[symbol]
        return age > max_age_seconds


class FundingTracker:
    """Tracks funding rates from exchange APIs."""
    
    def __init__(self):
        """Initialize funding tracker."""
        self.funding_rates: Dict[str, float] = {}  # symbol -> funding_rate_bps (per 8h, annualized)
        self.last_update: Dict[str, float] = {}
        logger.info("funding_tracker_initialized")
    
    def update_funding_rate(self, symbol: str, funding_rate_bps: float) -> None:
        """
        Update funding rate for a symbol.
        
        Args:
            symbol: Trading symbol
            funding_rate_bps: Funding rate in basis points (per 8 hours, annualized)
        """
        self.funding_rates[symbol] = funding_rate_bps
        self.last_update[symbol] = time.time()
        
        logger.debug(
            "funding_rate_updated",
            symbol=symbol,
            funding_rate_bps=funding_rate_bps,
        )
    
    def get_funding_rate(self, symbol: str) -> float:
        """Get funding rate for a symbol."""
        return self.funding_rates.get(symbol, 0.0)
    
    def is_stale(self, symbol: str, max_age_seconds: int = 3600) -> bool:
        """Check if funding rate data is stale."""
        if symbol not in self.last_update:
            return True
        
        age = time.time() - self.last_update[symbol]
        return age > max_age_seconds


class RealTimeCostModel:
    """
    Real-time cost model for spread, fee, and funding tracking.
    
    Features:
    - Real-time spread tracking from orderbook
    - Fee tracking from exchange APIs
    - Funding rate tracking from exchange APIs
    - Edge-after-cost calculation
    - Cost efficiency ranking
    - Skip coins failing threshold
    """
    
    def __init__(
        self,
        update_interval_seconds: int = 60,
        min_edge_after_cost_bps: float = 5.0,
        default_maker_fee_bps: float = 2.0,
        default_taker_fee_bps: float = 5.0,
    ):
        """
        Initialize real-time cost model.
        
        Args:
            update_interval_seconds: Update interval for cost data
            min_edge_after_cost_bps: Minimum edge after cost in basis points
            default_maker_fee_bps: Default maker fee in basis points
            default_taker_fee_bps: Default taker fee in basis points
        """
        self.update_interval = update_interval_seconds
        self.min_edge_after_cost_bps = min_edge_after_cost_bps
        
        # Trackers
        self.spread_tracker = SpreadTracker()
        self.fee_tracker = FeeTracker(
            default_maker_bps=default_maker_fee_bps,
            default_taker_bps=default_taker_fee_bps,
        )
        self.funding_tracker = FundingTracker()
        
        # Cost data cache
        self.cost_data: Dict[str, CostData] = {}
        
        # Update task
        self.update_task: Optional[asyncio.Task] = None
        self.running = False
        
        logger.info(
            "real_time_cost_model_initialized",
            update_interval=update_interval_seconds,
            min_edge=min_edge_after_cost_bps,
        )
    
    async def start(self) -> None:
        """Start cost model updates."""
        if self.running:
            return
        
        self.running = True
        self.update_task = asyncio.create_task(self._update_loop())
        logger.info("cost_model_started")
    
    async def stop(self) -> None:
        """Stop cost model updates."""
        self.running = False
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        logger.info("cost_model_stopped")
    
    async def _update_loop(self) -> None:
        """Update cost data in a loop."""
        while self.running:
            try:
                await self.update_costs()
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("cost_update_error", error=str(e))
                await asyncio.sleep(self.update_interval)
    
    async def update_costs(self) -> None:
        """Update cost data from all sources."""
        # This would fetch from exchange APIs/WebSockets
        # For now, we'll just update the cache with existing data
        
        # Update cost data cache
        for symbol in set(list(self.spread_tracker.spreads.keys()) +
                         list(self.fee_tracker.maker_fees.keys()) +
                         list(self.funding_tracker.funding_rates.keys())):
            cost_data = CostData(
                symbol=symbol,
                spread_bps=self.spread_tracker.get_spread(symbol),
                maker_fee_bps=self.fee_tracker.get_maker_fee(symbol),
                taker_fee_bps=self.fee_tracker.get_taker_fee(symbol),
                funding_rate_bps=self.funding_tracker.get_funding_rate(symbol),
                last_update=time.time(),
                source=CostSource.EXCHANGE_API,
            )
            self.cost_data[symbol] = cost_data
        
        logger.debug("costs_updated", symbols=len(self.cost_data))
    
    def update_spread(self, symbol: str, bid: float, ask: float) -> None:
        """Update spread for a symbol."""
        self.spread_tracker.update_spread(symbol, bid, ask)
    
    def update_fees(self, symbol: str, maker_bps: float, taker_bps: float) -> None:
        """Update fees for a symbol."""
        self.fee_tracker.update_fees(symbol, maker_bps, taker_bps)
    
    def update_funding_rate(self, symbol: str, funding_rate_bps: float) -> None:
        """Update funding rate for a symbol."""
        self.funding_tracker.update_funding_rate(symbol, funding_rate_bps)
    
    def calculate_total_cost(
        self,
        symbol: str,
        use_maker: bool = True,
        include_funding: bool = True,
    ) -> float:
        """
        Calculate total cost for a symbol.
        
        Args:
            symbol: Trading symbol
            use_maker: Use maker fee (True) or taker fee (False)
            include_funding: Include funding rate in cost
        
        Returns:
            Total cost in basis points
        """
        spread = self.spread_tracker.get_spread(symbol)
        fee = (
            self.fee_tracker.get_maker_fee(symbol) if use_maker
            else self.fee_tracker.get_taker_fee(symbol)
        )
        funding = (
            self.funding_tracker.get_funding_rate(symbol) if include_funding
            else 0.0
        )
        
        return spread + fee + funding
    
    def calculate_edge_after_cost(
        self,
        symbol: str,
        edge_bps: float,
        use_maker: bool = True,
        include_funding: bool = True,
    ) -> float:
        """
        Calculate edge after cost for a symbol.
        
        Args:
            symbol: Trading symbol
            edge_bps: Edge in basis points (before cost)
            use_maker: Use maker fee (True) or taker fee (False)
            include_funding: Include funding rate in cost
        
        Returns:
            Edge after cost in basis points
        """
        total_cost = self.calculate_total_cost(symbol, use_maker, include_funding)
        return edge_bps - total_cost
    
    def should_trade(
        self,
        symbol: str,
        edge_bps: float,
        use_maker: bool = True,
        include_funding: bool = True,
    ) -> bool:
        """
        Check if a symbol should be traded based on edge-after-cost.
        
        Args:
            symbol: Trading symbol
            edge_bps: Edge in basis points (before cost)
            use_maker: Use maker fee (True) or taker fee (False)
            include_funding: Include funding rate in cost
        
        Returns:
            True if edge-after-cost >= minimum threshold
        """
        edge_after_cost = self.calculate_edge_after_cost(
            symbol, edge_bps, use_maker, include_funding
        )
        return edge_after_cost >= self.min_edge_after_cost_bps
    
    def rank_symbols_by_cost_efficiency(
        self,
        symbols: List[str],
        edges_bps: Dict[str, float],
        use_maker: bool = True,
        include_funding: bool = True,
    ) -> List[CostRanking]:
        """
        Rank symbols by cost efficiency.
        
        Args:
            symbols: List of symbols to rank
            edges_bps: Edge in basis points per symbol
            use_maker: Use maker fee (True) or taker fee (False)
            include_funding: Include funding rate in cost
        
        Returns:
            List of CostRanking sorted by cost efficiency (highest first)
        """
        rankings = []
        
        for symbol in symbols:
            edge_bps = edges_bps.get(symbol, 0.0)
            total_cost = self.calculate_total_cost(symbol, use_maker, include_funding)
            edge_after_cost = self.calculate_edge_after_cost(
                symbol, edge_bps, use_maker, include_funding
            )
            
            # Cost efficiency = edge_after_cost / total_cost (higher is better)
            if total_cost > 0:
                cost_efficiency = edge_after_cost / total_cost
            else:
                cost_efficiency = float('inf') if edge_after_cost > 0 else 0.0
            
            rankings.append(CostRanking(
                symbol=symbol,
                total_cost_bps=total_cost,
                edge_after_cost_bps=edge_after_cost,
                rank=0,  # Will be set after sorting
                cost_efficiency=cost_efficiency,
            ))
        
        # Sort by cost efficiency (highest first)
        rankings.sort(key=lambda x: x.cost_efficiency, reverse=True)
        
        # Set ranks
        for i, ranking in enumerate(rankings):
            ranking.rank = i + 1
        
        return rankings
    
    def get_cost_data(self, symbol: str) -> Optional[CostData]:
        """Get cost data for a symbol."""
        return self.cost_data.get(symbol)
    
    def get_all_cost_data(self) -> Dict[str, CostData]:
        """Get all cost data."""
        return self.cost_data.copy()

