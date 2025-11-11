"""
Full Cost Model - Realistic trading costs.

Includes:
- Slippage (spread + participation rate)
- Fees (per venue and product)
- Funding and borrow costs (perps)
- Latency penalty (cross-exchange)
- Safety margin check
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


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
class VenueConfig:
    """Exchange/venue configuration."""
    name: str
    maker_fee_bps: float
    taker_fee_bps: float
    funding_rate_bps: float  # Per 8 hours for perps
    avg_latency_ms: float
    min_spread_bps: float


class FullCostModel:
    """
    Full cost model for realistic trading.
    
    Calculates:
    - Slippage as function of spread and participation rate
    - Fees per venue and product
    - Funding and borrow costs on perps
    - Latency penalty for cross-exchange logic
    - Safety margin check
    """
    
    def __init__(
        self,
        safety_margin_bps: float = 3.0,  # Minimum edge above costs
        default_participation_rate: float = 0.1,  # 10% of volume
    ) -> None:
        """
        Initialize full cost model.
        
        Args:
            safety_margin_bps: Minimum edge above costs in bps (default: 3.0)
            default_participation_rate: Default participation rate (default: 0.1)
        """
        self.safety_margin_bps = safety_margin_bps
        self.default_participation_rate = default_participation_rate
        
        # Default venue configurations
        self.venues: Dict[str, VenueConfig] = {
            "binance": VenueConfig(
                name="binance",
                maker_fee_bps=2.0,
                taker_fee_bps=4.0,
                funding_rate_bps=1.0,  # ~0.01% per 8h
                avg_latency_ms=50.0,
                min_spread_bps=1.0
            ),
            "coinbase": VenueConfig(
                name="coinbase",
                maker_fee_bps=5.0,
                taker_fee_bps=5.0,
                funding_rate_bps=1.0,
                avg_latency_ms=80.0,
                min_spread_bps=2.0
            ),
        }
        
        logger.info(
            "full_cost_model_initialized",
            safety_margin_bps=safety_margin_bps,
            default_participation_rate=default_participation_rate
        )
    
    def calculate_slippage(
        self,
        spread_bps: float,
        participation_rate: float,
        order_size_usd: float,
        avg_volume_usd: float
    ) -> float:
        """
        Calculate slippage as function of spread and participation rate.
        
        Slippage = spread_cost + market_impact
        Market impact ∝ sqrt(participation_rate)
        
        Args:
            spread_bps: Bid-ask spread in bps
            participation_rate: Order size / average volume
            order_size_usd: Order size in USD
            avg_volume_usd: Average volume in USD
        
        Returns:
            Slippage in bps
        """
        # Spread cost (half spread for market orders)
        spread_cost = spread_bps / 2.0
        
        # Market impact (square root model)
        # Impact increases with participation rate
        impact_factor = np.sqrt(participation_rate)
        market_impact = impact_factor * 5.0  # Base impact of 5 bps at 100% participation
        
        # Cap market impact
        market_impact = min(market_impact, 20.0)  # Max 20 bps
        
        total_slippage = spread_cost + market_impact
        
        logger.debug(
            "slippage_calculated",
            spread_bps=spread_bps,
            participation_rate=participation_rate,
            spread_cost=spread_cost,
            market_impact=market_impact,
            total_slippage=total_slippage
        )
        
        return total_slippage
    
    def calculate_fees(
        self,
        venue: str,
        is_maker: bool,
        entry_size_usd: float,
        exit_size_usd: float
    ) -> Tuple[float, float]:
        """
        Calculate fees per venue and product.
        
        Args:
            venue: Venue name
            is_maker: Whether using maker orders
            entry_size_usd: Entry order size
            exit_size_usd: Exit order size
        
        Returns:
            (entry_fee_bps, exit_fee_bps)
        """
        if venue not in self.venues:
            # Default fees
            fee_bps = 2.0 if is_maker else 4.0
            return fee_bps, fee_bps
        
        venue_config = self.venues[venue]
        fee_bps = venue_config.maker_fee_bps if is_maker else venue_config.taker_fee_bps
        
        return fee_bps, fee_bps
    
    def calculate_funding_cost(
        self,
        venue: str,
        position_size_usd: float,
        funding_rate_bps: Optional[float] = None,
        hours_held: float = 8.0
    ) -> float:
        """
        Calculate funding and borrow costs on perps.
        
        Args:
            venue: Venue name
            position_size_usd: Position size in USD
            funding_rate_bps: Funding rate in bps (uses venue default if None)
            hours_held: Hours position will be held (default: 8h)
        
        Returns:
            Funding cost in bps
        """
        if venue not in self.venues:
            funding_rate_bps = funding_rate_bps or 1.0
        else:
            funding_rate_bps = funding_rate_bps or self.venues[venue].funding_rate_bps
        
        # Funding is paid every 8 hours
        funding_periods = hours_held / 8.0
        total_funding_bps = funding_rate_bps * funding_periods
        
        return total_funding_bps
    
    def calculate_latency_penalty(
        self,
        venue: str,
        is_cross_exchange: bool = False
    ) -> float:
        """
        Calculate latency penalty for cross-exchange logic.
        
        Args:
            venue: Venue name
            is_cross_exchange: Whether using cross-exchange arbitrage
        
        Returns:
            Latency penalty in bps
        """
        if not is_cross_exchange:
            return 0.0
        
        if venue not in self.venues:
            avg_latency_ms = 100.0  # Default
        else:
            avg_latency_ms = self.venues[venue].avg_latency_ms
        
        # Latency penalty: ~0.1 bps per 10ms of latency
        penalty_bps = (avg_latency_ms / 10.0) * 0.1
        
        return penalty_bps
    
    def calculate_total_cost(
        self,
        venue: str,
        spread_bps: float,
        expected_edge_bps: float,
        order_size_usd: float,
        avg_volume_usd: float,
        is_maker: bool = False,
        is_perp: bool = False,
        hours_held: float = 8.0,
        is_cross_exchange: bool = False,
        participation_rate: Optional[float] = None
    ) -> CostBreakdown:
        """
        Calculate full cost breakdown.
        
        Args:
            venue: Venue name
            spread_bps: Current spread in bps
            expected_edge_bps: Expected edge from model in bps
            order_size_usd: Order size in USD
            avg_volume_usd: Average volume in USD
            is_maker: Whether using maker orders
            is_perp: Whether trading perpetuals
            hours_held: Expected holding time in hours
            is_cross_exchange: Whether cross-exchange trade
            participation_rate: Participation rate (uses default if None)
        
        Returns:
            CostBreakdown with all costs and safety check
        """
        if participation_rate is None:
            participation_rate = self.default_participation_rate
        
        # Calculate all cost components
        entry_fee, exit_fee = self.calculate_fees(venue, is_maker, order_size_usd, order_size_usd)
        slippage = self.calculate_slippage(spread_bps, participation_rate, order_size_usd, avg_volume_usd)
        
        funding_cost = 0.0
        if is_perp:
            funding_cost = self.calculate_funding_cost(venue, order_size_usd, hours_held=hours_held)
        
        latency_penalty = self.calculate_latency_penalty(venue, is_cross_exchange)
        
        # Total costs
        total_cost = entry_fee + exit_fee + slippage + funding_cost + latency_penalty
        
        # Net edge after costs
        net_edge = expected_edge_bps - total_cost
        
        # Safety margin check
        passes_safety = net_edge >= self.safety_margin_bps
        
        breakdown = CostBreakdown(
            entry_fee_bps=entry_fee,
            exit_fee_bps=exit_fee,
            spread_cost_bps=spread_bps / 2.0,
            slippage_bps=slippage,
            funding_cost_bps=funding_cost,
            latency_penalty_bps=latency_penalty,
            total_cost_bps=total_cost,
            net_edge_bps=net_edge,
            passes_safety_margin=passes_safety
        )
        
        logger.info(
            "cost_calculation_complete",
            venue=venue,
            expected_edge_bps=expected_edge_bps,
            total_cost_bps=total_cost,
            net_edge_bps=net_edge,
            passes_safety=passes_safety
        )
        
        return breakdown
    
    def should_reject_trade(
        self,
        expected_edge_bps: float,
        cost_breakdown: CostBreakdown
    ) -> bool:
        """
        Reject trades that don't clear net edge after costs by safety margin.
        
        Args:
            expected_edge_bps: Expected edge
            cost_breakdown: Cost breakdown
        
        Returns:
            True if trade should be rejected
        """
        return not cost_breakdown.passes_safety_margin

