"""
Cost Calculator

Costs in the loop. Fees, spread, slippage per symbol, per bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CostModel:
    """Cost model for a symbol."""
    symbol: str
    taker_fee_bps: float  # Taker fee in basis points
    maker_fee_bps: float  # Maker fee in basis points
    median_spread_bps: float  # Median spread in basis points
    slippage_bps_per_sigma: float  # Slippage in basis points per sigma
    min_notional: float  # Minimum notional for trading
    step_size: float  # Step size for order sizing
    last_updated_utc: datetime
    slippage_calibration_date: Optional[datetime] = None
    
    def total_cost_bps(self, order_type: str = "taker", order_size_sigma: float = 1.0) -> float:
        """Calculate total cost in basis points.
        
        Args:
            order_type: Order type ("taker" or "maker")
            order_size_sigma: Order size in standard deviations (for slippage)
            
        Returns:
            Total cost in basis points
        """
        fee_bps = self.taker_fee_bps if order_type == "taker" else self.maker_fee_bps
        spread_bps = self.median_spread_bps
        slippage_bps = self.slippage_bps_per_sigma * order_size_sigma
        
        return fee_bps + spread_bps + slippage_bps


class CostCalculator:
    """Cost calculator for per-symbol, per-bar costs."""
    
    def __init__(self):
        """Initialize cost calculator."""
        self.cost_models: Dict[str, CostModel] = {}
        logger.info("cost_calculator_initialized")
    
    def get_costs(
        self,
        symbol: str,
        timestamp: datetime,
        order_type: str = "taker",
        order_size_sigma: float = 1.0,
    ) -> Dict[str, float]:
        """Get costs for a symbol at a given timestamp.
        
        Args:
            symbol: Trading symbol
            timestamp: Timestamp for cost calculation
            order_type: Order type ("taker" or "maker")
            order_size_sigma: Order size in standard deviations
            
        Returns:
            Dictionary with cost components
        """
        cost_model = self.cost_models.get(symbol)
        
        if not cost_model:
            # Return default costs if model not found
            logger.warning("cost_model_not_found", symbol=symbol, message="Using default costs")
            return {
                "fees_bps": 4.0,  # Default taker fee
                "spread_bps": 5.0,  # Default spread
                "slippage_bps": 2.0,  # Default slippage
                "total_cost_bps": 11.0,  # Total
            }
        
        fee_bps = cost_model.taker_fee_bps if order_type == "taker" else cost_model.maker_fee_bps
        spread_bps = cost_model.median_spread_bps
        slippage_bps = cost_model.slippage_bps_per_sigma * order_size_sigma
        total_cost_bps = cost_model.total_cost_bps(order_type, order_size_sigma)
        
        return {
            "fees_bps": fee_bps,
            "spread_bps": spread_bps,
            "slippage_bps": slippage_bps,
            "total_cost_bps": total_cost_bps,
            "min_notional": cost_model.min_notional,
            "step_size": cost_model.step_size,
        }
    
    def register_cost_model(self, cost_model: CostModel) -> None:
        """Register a cost model for a symbol.
        
        Args:
            cost_model: Cost model instance
        """
        self.cost_models[cost_model.symbol] = cost_model
        logger.info("cost_model_registered", symbol=cost_model.symbol)
    
    def calculate_net_edge(
        self,
        symbol: str,
        edge_bps_before_costs: float,
        timestamp: datetime,
        order_type: str = "taker",
        order_size_sigma: float = 1.0,
    ) -> float:
        """Calculate net edge after costs.
        
        Args:
            symbol: Trading symbol
            edge_bps_before_costs: Edge before costs in basis points
            timestamp: Timestamp for cost calculation
            order_type: Order type ("taker" or "maker")
            order_size_sigma: Order size in standard deviations
            
        Returns:
            Net edge after costs in basis points
        """
        costs = self.get_costs(symbol, timestamp, order_type, order_size_sigma)
        total_cost_bps = costs["total_cost_bps"]
        
        net_edge = edge_bps_before_costs - total_cost_bps
        
        logger.debug("net_edge_calculated", 
                    symbol=symbol,
                    edge_before_costs=edge_bps_before_costs,
                    total_cost_bps=total_cost_bps,
                    net_edge=net_edge)
        
        return net_edge
    
    def should_trade(
        self,
        symbol: str,
        edge_bps_before_costs: float,
        timestamp: datetime,
        net_edge_floor_bps: float = 3.0,
        order_type: str = "taker",
        order_size_sigma: float = 1.0,
    ) -> bool:
        """Check if should trade based on net edge floor.
        
        Args:
            symbol: Trading symbol
            edge_bps_before_costs: Edge before costs in basis points
            timestamp: Timestamp for cost calculation
            net_edge_floor_bps: Minimum net edge required to trade
            order_type: Order type ("taker" or "maker")
            order_size_sigma: Order size in standard deviations
            
        Returns:
            True if should trade, False otherwise
        """
        net_edge = self.calculate_net_edge(symbol, edge_bps_before_costs, timestamp, order_type, order_size_sigma)
        
        should_trade = net_edge >= net_edge_floor_bps
        
        if not should_trade:
            logger.debug("trade_skipped_low_edge", 
                        symbol=symbol,
                        net_edge=net_edge,
                        net_edge_floor=net_edge_floor_bps)
        
        return should_trade

