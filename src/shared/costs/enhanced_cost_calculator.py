"""
Enhanced Cost Calculator

Handles holding context: funding costs, overnight risk, multi-day costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import structlog

from ..engines.enhanced_engine_interface import TradingHorizon

logger = structlog.get_logger(__name__)


@dataclass
class EnhancedCostModel:
    """Enhanced cost model with holding context."""
    symbol: str
    taker_fee_bps: float  # Taker fee in basis points
    maker_fee_bps: float  # Maker fee in basis points
    median_spread_bps: float  # Median spread in basis points
    slippage_bps_per_sigma: float  # Slippage in basis points per sigma
    min_notional: float  # Minimum notional for trading
    step_size: float  # Step size for order sizing
    last_updated_utc: datetime
    # Funding and overnight costs
    funding_rate_bps_per_8h: float = 1.0  # Funding rate per 8 hours in basis points
    borrow_rate_bps_per_day: float = 0.0  # Borrow rate per day (for margin)
    overnight_risk_multiplier: float = 1.2  # Multiplier for overnight risk
    # Liquidity over time
    liquidity_decay_factor: float = 0.95  # Liquidity decay factor per day
    spread_widening_bps_per_day: float = 0.5  # Spread widening per day
    slippage_calibration_date: Optional[datetime] = None
    
    def calculate_funding_cost(
        self,
        holding_hours: float,
        position_size_usd: float,
    ) -> float:
        """Calculate funding cost over holding period.
        
        Args:
            holding_hours: Holding period in hours
            position_size_usd: Position size in USD
            
        Returns:
            Funding cost in basis points
        """
        # Funding is paid every 8 hours
        funding_periods = holding_hours / 8.0
        total_funding_bps = self.funding_rate_bps_per_8h * funding_periods
        
        # Apply overnight risk multiplier for positions held > 24 hours
        if holding_hours > 24.0:
            overnight_multiplier = 1.0 + (self.overnight_risk_multiplier - 1.0) * min((holding_hours - 24.0) / 24.0, 1.0)
            total_funding_bps *= overnight_multiplier
        
        return total_funding_bps
    
    def calculate_borrow_cost(
        self,
        holding_days: float,
        position_size_usd: float,
    ) -> float:
        """Calculate borrow cost over holding period.
        
        Args:
            holding_days: Holding period in days
            position_size_usd: Position size in USD
            
        Returns:
            Borrow cost in basis points
        """
        return self.borrow_rate_bps_per_day * holding_days
    
    def calculate_liquidity_cost(
        self,
        holding_days: float,
        order_size_sigma: float = 1.0,
    ) -> float:
        """Calculate liquidity cost over holding period.
        
        Args:
            holding_days: Holding period in days
            order_size_sigma: Order size in standard deviations
            
        Returns:
            Liquidity cost in basis points (spread + slippage with decay)
        """
        # Base spread and slippage
        base_spread_bps = self.median_spread_bps
        base_slippage_bps = self.slippage_bps_per_sigma * order_size_sigma
        
        # Apply liquidity decay over time
        liquidity_factor = self.liquidity_decay_factor ** holding_days
        spread_widening = self.spread_widening_bps_per_day * holding_days
        
        # Adjusted costs
        adjusted_spread_bps = (base_spread_bps / liquidity_factor) + spread_widening
        adjusted_slippage_bps = base_slippage_bps / liquidity_factor
        
        return adjusted_spread_bps + adjusted_slippage_bps
    
    def total_cost_bps(
        self,
        order_type: str = "taker",
        order_size_sigma: float = 1.0,
        holding_hours: float = 0.0,
        include_funding: bool = True,
        include_borrow: bool = False,
        include_liquidity_decay: bool = True,
    ) -> float:
        """Calculate total cost in basis points with holding context.
        
        Args:
            order_type: Order type ("taker" or "maker")
            order_size_sigma: Order size in standard deviations
            holding_hours: Holding period in hours
            include_funding: Include funding costs
            include_borrow: Include borrow costs
            include_liquidity_decay: Include liquidity decay
            
        Returns:
            Total cost in basis points
        """
        # Entry/exit fees
        fee_bps = self.taker_fee_bps if order_type == "taker" else self.maker_fee_bps
        total_fees_bps = fee_bps * 2.0  # Entry + exit
        
        # Spread and slippage
        if include_liquidity_decay and holding_hours > 0.0:
            holding_days = holding_hours / 24.0
            liquidity_cost_bps = self.calculate_liquidity_cost(holding_days, order_size_sigma)
        else:
            spread_bps = self.median_spread_bps
            slippage_bps = self.slippage_bps_per_sigma * order_size_sigma
            liquidity_cost_bps = spread_bps + slippage_bps
        
        # Funding cost
        funding_cost_bps = 0.0
        if include_funding and holding_hours > 0.0:
            funding_cost_bps = self.calculate_funding_cost(holding_hours, 1.0)  # Size doesn't affect bps
        
        # Borrow cost
        borrow_cost_bps = 0.0
        if include_borrow and holding_hours > 0.0:
            holding_days = holding_hours / 24.0
            borrow_cost_bps = self.calculate_borrow_cost(holding_days, 1.0)  # Size doesn't affect bps
        
        total_cost_bps = total_fees_bps + liquidity_cost_bps + funding_cost_bps + borrow_cost_bps
        
        return total_cost_bps


@dataclass
class CostBreakdown:
    """Cost breakdown with holding context."""
    entry_fee_bps: float
    exit_fee_bps: float
    spread_bps: float
    slippage_bps: float
    funding_cost_bps: float
    borrow_cost_bps: float
    liquidity_decay_bps: float
    overnight_risk_bps: float
    total_cost_bps: float
    net_edge_bps: float
    holding_hours: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedCostCalculator:
    """Enhanced cost calculator with holding context."""
    
    def __init__(self):
        """Initialize enhanced cost calculator."""
        self.cost_models: Dict[str, EnhancedCostModel] = {}
        logger.info("enhanced_cost_calculator_initialized")
    
    def get_costs(
        self,
        symbol: str,
        timestamp: datetime,
        order_type: str = "taker",
        order_size_sigma: float = 1.0,
        holding_hours: float = 0.0,
        horizon_type: TradingHorizon = TradingHorizon.SCALP,
        include_funding: bool = True,
        include_borrow: bool = False,
    ) -> CostBreakdown:
        """Get costs for a symbol with holding context.
        
        Args:
            symbol: Trading symbol
            timestamp: Timestamp for cost calculation
            order_type: Order type ("taker" or "maker")
            order_size_sigma: Order size in standard deviations
            holding_hours: Holding period in hours
            horizon_type: Trading horizon type
            include_funding: Include funding costs
            include_borrow: Include borrow costs
            
        Returns:
            Cost breakdown with all components
        """
        cost_model = self.cost_models.get(symbol)
        
        if not cost_model:
            # Return default costs if model not found
            logger.warning("cost_model_not_found", symbol=symbol, message="Using default costs")
            cost_model = EnhancedCostModel(
                symbol=symbol,
                taker_fee_bps=4.0,
                maker_fee_bps=2.0,
                median_spread_bps=5.0,
                slippage_bps_per_sigma=2.0,
                min_notional=10.0,
                step_size=0.01,
                last_updated_utc=timestamp,
            )
        
        # Determine if we should include funding/borrow based on horizon
        if horizon_type in [TradingHorizon.SWING, TradingHorizon.POSITION, TradingHorizon.CORE]:
            include_funding = True
            include_borrow = True if horizon_type == TradingHorizon.CORE else False
        
        # Calculate cost components
        fee_bps = cost_model.taker_fee_bps if order_type == "taker" else cost_model.maker_fee_bps
        entry_fee_bps = fee_bps
        exit_fee_bps = fee_bps
        
        # Spread and slippage (with liquidity decay for longer holds)
        if holding_hours > 0.0:
            holding_days = holding_hours / 24.0
            liquidity_cost_bps = cost_model.calculate_liquidity_cost(holding_days, order_size_sigma)
            base_spread_bps = cost_model.median_spread_bps
            base_slippage_bps = cost_model.slippage_bps_per_sigma * order_size_sigma
            spread_bps = base_spread_bps
            slippage_bps = base_slippage_bps
            liquidity_decay_bps = liquidity_cost_bps - (base_spread_bps + base_slippage_bps)
        else:
            spread_bps = cost_model.median_spread_bps
            slippage_bps = cost_model.slippage_bps_per_sigma * order_size_sigma
            liquidity_decay_bps = 0.0
        
        # Funding cost
        funding_cost_bps = 0.0
        if include_funding and holding_hours > 0.0:
            funding_cost_bps = cost_model.calculate_funding_cost(holding_hours, 1.0)
        
        # Borrow cost
        borrow_cost_bps = 0.0
        if include_borrow and holding_hours > 0.0:
            holding_days = holding_hours / 24.0
            borrow_cost_bps = cost_model.calculate_borrow_cost(holding_days, 1.0)
        
        # Overnight risk (for positions held > 24 hours)
        overnight_risk_bps = 0.0
        if holding_hours > 24.0:
            overnight_multiplier = cost_model.overnight_risk_multiplier - 1.0
            overnight_risk_bps = (funding_cost_bps + borrow_cost_bps) * overnight_multiplier * min((holding_hours - 24.0) / 24.0, 1.0)
        
        # Total cost
        total_cost_bps = (
            entry_fee_bps +
            exit_fee_bps +
            spread_bps +
            slippage_bps +
            funding_cost_bps +
            borrow_cost_bps +
            liquidity_decay_bps +
            overnight_risk_bps
        )
        
        return CostBreakdown(
            entry_fee_bps=entry_fee_bps,
            exit_fee_bps=exit_fee_bps,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
            funding_cost_bps=funding_cost_bps,
            borrow_cost_bps=borrow_cost_bps,
            liquidity_decay_bps=liquidity_decay_bps,
            overnight_risk_bps=overnight_risk_bps,
            total_cost_bps=total_cost_bps,
            net_edge_bps=0.0,  # Will be calculated separately
            holding_hours=holding_hours,
            metadata={
                "horizon_type": horizon_type.value,
                "order_type": order_type,
                "order_size_sigma": order_size_sigma,
            },
        )
    
    def calculate_net_edge(
        self,
        symbol: str,
        edge_bps_before_costs: float,
        timestamp: datetime,
        order_type: str = "taker",
        order_size_sigma: float = 1.0,
        holding_hours: float = 0.0,
        horizon_type: TradingHorizon = TradingHorizon.SCALP,
        include_funding: bool = True,
        include_borrow: bool = False,
    ) -> float:
        """Calculate net edge after costs with holding context.
        
        Args:
            symbol: Trading symbol
            edge_bps_before_costs: Edge before costs in basis points
            timestamp: Timestamp for cost calculation
            order_type: Order type ("taker" or "maker")
            order_size_sigma: Order size in standard deviations
            holding_hours: Holding period in hours
            horizon_type: Trading horizon type
            include_funding: Include funding costs
            include_borrow: Include borrow costs
            
        Returns:
            Net edge after costs in basis points
        """
        cost_breakdown = self.get_costs(
            symbol=symbol,
            timestamp=timestamp,
            order_type=order_type,
            order_size_sigma=order_size_sigma,
            holding_hours=holding_hours,
            horizon_type=horizon_type,
            include_funding=include_funding,
            include_borrow=include_borrow,
        )
        
        net_edge = edge_bps_before_costs - cost_breakdown.total_cost_bps
        
        logger.debug(
            "net_edge_calculated_with_holding",
            symbol=symbol,
            edge_before_costs=edge_bps_before_costs,
            total_cost_bps=cost_breakdown.total_cost_bps,
            net_edge=net_edge,
            holding_hours=holding_hours,
            horizon_type=horizon_type.value,
        )
        
        return net_edge
    
    def register_cost_model(self, cost_model: EnhancedCostModel) -> None:
        """Register a cost model for a symbol.
        
        Args:
            cost_model: Enhanced cost model instance
        """
        self.cost_models[cost_model.symbol] = cost_model
        logger.info("enhanced_cost_model_registered", symbol=cost_model.symbol)
    
    def should_trade(
        self,
        symbol: str,
        edge_bps_before_costs: float,
        timestamp: datetime,
        net_edge_floor_bps: float = 3.0,
        order_type: str = "taker",
        order_size_sigma: float = 1.0,
        holding_hours: float = 0.0,
        horizon_type: TradingHorizon = TradingHorizon.SCALP,
        include_funding: bool = True,
        include_borrow: bool = False,
    ) -> bool:
        """Check if should trade based on net edge floor with holding context.
        
        Args:
            symbol: Trading symbol
            edge_bps_before_costs: Edge before costs in basis points
            timestamp: Timestamp for cost calculation
            net_edge_floor_bps: Minimum net edge required to trade
            order_type: Order type ("taker" or "maker")
            order_size_sigma: Order size in standard deviations
            holding_hours: Holding period in hours
            horizon_type: Trading horizon type
            include_funding: Include funding costs
            include_borrow: Include borrow costs
            
        Returns:
            True if should trade, False otherwise
        """
        net_edge = self.calculate_net_edge(
            symbol=symbol,
            edge_bps_before_costs=edge_bps_before_costs,
            timestamp=timestamp,
            order_type=order_type,
            order_size_sigma=order_size_sigma,
            holding_hours=holding_hours,
            horizon_type=horizon_type,
            include_funding=include_funding,
            include_borrow=include_borrow,
        )
        
        should_trade = net_edge >= net_edge_floor_bps
        
        if not should_trade:
            logger.debug(
                "trade_skipped_low_edge_with_holding",
                symbol=symbol,
                net_edge=net_edge,
                net_edge_floor=net_edge_floor_bps,
                holding_hours=holding_hours,
                horizon_type=horizon_type.value,
            )
        
        return should_trade

