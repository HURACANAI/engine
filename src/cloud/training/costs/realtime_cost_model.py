"""
Real-Time Cost Model

Calculates real-time trading costs including fees, spreads, slippage,
and funding costs per venue. Updates costs dynamically via WebSocket
or API polling.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class CostSource(str, Enum):
    """Cost data source."""
    WEBSOCKET = "websocket"  # Real-time via WebSocket
    API = "api"  # Polling via API
    STATIC = "static"  # Static values


@dataclass
class VenueConfig:
    """Venue (exchange) configuration."""
    name: str
    maker_fee_bps: float
    taker_fee_bps: float
    funding_rate_bps_per_8h: float  # For perpetuals
    avg_latency_ms: float
    min_spread_bps: float


@dataclass
class CostBreakdown:
    """Detailed cost breakdown."""
    entry_fee_bps: float
    exit_fee_bps: float
    spread_cost_bps: float
    slippage_bps: float
    funding_cost_bps: float
    latency_penalty_bps: float
    total_cost_bps: float
    net_edge_bps: float  # Expected edge minus costs
    passes_safety_margin: bool


@dataclass
class RealTimeCostConfig:
    """Configuration for real-time cost model."""
    update_interval_seconds: int = 60
    spread_source: CostSource = CostSource.WEBSOCKET
    fee_source: CostSource = CostSource.API
    funding_source: CostSource = CostSource.API
    min_edge_after_cost_bps: float = 3.0
    slippage_participation_rate: float = 0.1  # 10% of volume
    slippage_impact_factor: float = 0.5


class RealTimeCostModel:
    """
    Real-time cost model with dynamic updates.
    
    Features:
    - Venue-specific fee schedules
    - Real-time spread updates
    - Slippage modeling
    - Funding cost tracking
    - Edge-after-cost calculation
    """
    
    def __init__(
        self,
        config: RealTimeCostConfig,
        venues: Optional[Dict[str, VenueConfig]] = None,
    ) -> None:
        """
        Initialize real-time cost model.
        
        Args:
            config: Cost model configuration
            venues: Venue configurations (defaults to Binance/Coinbase)
        """
        self.config = config
        
        # Venue configurations
        self.venues = venues or self._default_venues()
        
        # Real-time cost data (symbol -> cost data)
        self.spread_data: Dict[str, float] = {}  # symbol -> spread_bps
        self.funding_data: Dict[str, float] = {}  # symbol -> funding_bps_per_8h
        
        # Update task
        self.update_task: Optional[asyncio.Task] = None
        self.running = False
        
        logger.info(
            "realtime_cost_model_initialized",
            update_interval_seconds=config.update_interval_seconds,
            spread_source=config.spread_source.value,
            num_venues=len(self.venues),
        )
    
    def _default_venues(self) -> Dict[str, VenueConfig]:
        """Get default venue configurations."""
        return {
            "binance": VenueConfig(
                name="binance",
                maker_fee_bps=2.0,
                taker_fee_bps=4.0,
                funding_rate_bps_per_8h=1.0,
                avg_latency_ms=50.0,
                min_spread_bps=1.0,
            ),
            "coinbase": VenueConfig(
                name="coinbase",
                maker_fee_bps=5.0,
                taker_fee_bps=5.0,
                funding_rate_bps_per_8h=1.0,
                avg_latency_ms=80.0,
                min_spread_bps=2.0,
            ),
        }
    
    async def start(self) -> None:
        """Start real-time cost updates."""
        if self.running:
            return
        
        self.running = True
        
        if self.config.spread_source == CostSource.WEBSOCKET:
            self.update_task = asyncio.create_task(self._websocket_update_loop())
        else:
            self.update_task = asyncio.create_task(self._polling_update_loop())
        
        logger.info("realtime_cost_model_started")
    
    async def stop(self) -> None:
        """Stop real-time cost updates."""
        self.running = False
        
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        logger.info("realtime_cost_model_stopped")
    
    async def _websocket_update_loop(self) -> None:
        """WebSocket update loop for real-time spreads."""
        # This would connect to exchange WebSocket and update spreads
        # For now, this is a placeholder
        while self.running:
            try:
                # Update spreads from WebSocket
                # await self._update_spreads_from_websocket()
                await asyncio.sleep(self.config.update_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("websocket_update_error", error=str(e))
                await asyncio.sleep(self.config.update_interval_seconds)
    
    async def _polling_update_loop(self) -> None:
        """Polling update loop for costs."""
        while self.running:
            try:
                # Update costs from API
                await self._update_costs_from_api()
                await asyncio.sleep(self.config.update_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("polling_update_error", error=str(e))
                await asyncio.sleep(self.config.update_interval_seconds)
    
    async def _update_costs_from_api(self) -> None:
        """Update costs from exchange API."""
        # This would fetch spreads, funding rates, etc. from exchange API
        # For now, this is a placeholder
        pass
    
    def calculate_costs(
        self,
        symbol: str,
        venue: str,
        side: str,  # "buy" or "sell"
        size_usd: float,
        expected_edge_bps: float,
        order_type: str = "taker",  # "maker" or "taker"
        holding_hours: float = 1.0,  # For funding cost calculation
    ) -> CostBreakdown:
        """
        Calculate total trading costs.
        
        Args:
            symbol: Trading symbol
            venue: Exchange venue name
            side: Trade side ("buy" or "sell")
            size_usd: Trade size in USD
            expected_edge_bps: Expected edge in basis points
            order_type: Order type ("maker" or "taker")
            holding_hours: Expected holding time in hours (for funding)
        
        Returns:
            Cost breakdown
        """
        # Get venue config
        venue_config = self.venues.get(venue)
        if venue_config is None:
            logger.warning("unknown_venue", venue=venue)
            venue_config = self.venues["binance"]  # Default
        
        # Entry fee
        entry_fee_bps = venue_config.taker_fee_bps if order_type == "taker" else venue_config.maker_fee_bps
        
        # Exit fee (same as entry for simplicity)
        exit_fee_bps = entry_fee_bps
        
        # Spread cost
        spread_bps = self.spread_data.get(symbol, venue_config.min_spread_bps)
        spread_cost_bps = spread_bps / 2.0  # Half spread on entry, half on exit
        
        # Slippage
        slippage_bps = self._calculate_slippage(size_usd, spread_bps)
        
        # Funding cost (for perpetuals)
        funding_bps_per_8h = self.funding_data.get(symbol, venue_config.funding_rate_bps_per_8h)
        funding_cost_bps = funding_bps_per_8h * (holding_hours / 8.0)
        
        # Latency penalty (minimal for most cases)
        latency_penalty_bps = 0.0  # Usually negligible
        
        # Total cost
        total_cost_bps = (
            entry_fee_bps +
            exit_fee_bps +
            spread_cost_bps +
            slippage_bps +
            funding_cost_bps +
            latency_penalty_bps
        )
        
        # Net edge
        net_edge_bps = expected_edge_bps - total_cost_bps
        
        # Safety margin check
        passes_safety_margin = net_edge_bps >= self.config.min_edge_after_cost_bps
        
        breakdown = CostBreakdown(
            entry_fee_bps=entry_fee_bps,
            exit_fee_bps=exit_fee_bps,
            spread_cost_bps=spread_cost_bps,
            slippage_bps=slippage_bps,
            funding_cost_bps=funding_cost_bps,
            latency_penalty_bps=latency_penalty_bps,
            total_cost_bps=total_cost_bps,
            net_edge_bps=net_edge_bps,
            passes_safety_margin=passes_safety_margin,
        )
        
        logger.debug(
            "costs_calculated",
            symbol=symbol,
            venue=venue,
            total_cost_bps=total_cost_bps,
            net_edge_bps=net_edge_bps,
            passes_safety_margin=passes_safety_margin,
        )
        
        return breakdown
    
    def _calculate_slippage(
        self,
        size_usd: float,
        spread_bps: float,
    ) -> float:
        """
        Calculate slippage based on order size and spread.
        
        Args:
            size_usd: Order size in USD
            spread_bps: Current spread in basis points
        
        Returns:
            Slippage in basis points
        """
        # Simple slippage model: increases with participation rate
        # More sophisticated models would use order book depth
        
        participation_rate = self.config.slippage_participation_rate
        impact_factor = self.config.slippage_impact_factor
        
        # Base slippage from spread
        base_slippage = spread_bps / 2.0
        
        # Additional slippage from market impact
        # This is a simplified model - real implementation would use order book data
        market_impact = base_slippage * participation_rate * impact_factor
        
        total_slippage = base_slippage + market_impact
        
        return total_slippage
    
    def update_spread(self, symbol: str, spread_bps: float) -> None:
        """Update spread for a symbol."""
        self.spread_data[symbol] = spread_bps
    
    def update_funding_rate(self, symbol: str, funding_bps_per_8h: float) -> None:
        """Update funding rate for a symbol."""
        self.funding_data[symbol] = funding_bps_per_8h
    
    def get_spread(self, symbol: str) -> float:
        """Get current spread for a symbol."""
        return self.spread_data.get(symbol, 5.0)  # Default 5 bps
    
    def get_funding_rate(self, symbol: str) -> float:
        """Get current funding rate for a symbol."""
        return self.funding_data.get(symbol, 1.0)  # Default 1 bps per 8h

