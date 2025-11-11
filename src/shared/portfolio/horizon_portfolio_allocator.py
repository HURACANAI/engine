"""
Horizon-Based Portfolio Allocator

Manages portfolio allocation across different horizon buckets (scalps vs swings vs core).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from ..engines.enhanced_engine_interface import TradingHorizon

logger = structlog.get_logger(__name__)


@dataclass
class HorizonAllocation:
    """Allocation for a specific horizon."""
    horizon: TradingHorizon
    max_allocation_pct: float  # Maximum allocation as % of total portfolio
    current_allocation_pct: float  # Current allocation as % of total portfolio
    target_allocation_pct: float  # Target allocation as % of total portfolio
    max_positions: int  # Maximum number of positions
    current_positions: int  # Current number of positions
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioAllocation:
    """Portfolio allocation across horizons."""
    total_portfolio_value: float
    allocations: Dict[TradingHorizon, HorizonAllocation]
    last_updated: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_available_capacity(self, horizon: TradingHorizon) -> float:
        """Get available capacity for a horizon.
        
        Args:
            horizon: Trading horizon
            
        Returns:
            Available capacity in USD
        """
        if horizon not in self.allocations:
            return 0.0
        
        allocation = self.allocations[horizon]
        available_pct = allocation.max_allocation_pct - allocation.current_allocation_pct
        return (available_pct / 100.0) * self.total_portfolio_value
    
    def get_max_position_size(self, horizon: TradingHorizon) -> float:
        """Get maximum position size for a horizon.
        
        Args:
            horizon: Trading horizon
            
        Returns:
            Maximum position size in USD
        """
        if horizon not in self.allocations:
            return 0.0
        
        allocation = self.allocations[horizon]
        max_allocation_usd = (allocation.max_allocation_pct / 100.0) * self.total_portfolio_value
        
        # Divide by max positions to get max position size
        if allocation.max_positions > 0:
            return max_allocation_usd / allocation.max_positions
        else:
            return max_allocation_usd


@dataclass
class HorizonPortfolioConfig:
    """Configuration for horizon-based portfolio allocation."""
    # Allocation limits by horizon
    scalp_max_allocation_pct: float = 20.0  # Max 20% for scalps
    swing_max_allocation_pct: float = 40.0  # Max 40% for swings
    position_max_allocation_pct: float = 30.0  # Max 30% for positions
    core_max_allocation_pct: float = 50.0  # Max 50% for core (can overlap)
    # Position limits
    scalp_max_positions: int = 5
    swing_max_positions: int = 3
    position_max_positions: int = 2
    core_max_positions: int = 3
    # Rebalancing
    rebalance_threshold_pct: float = 5.0  # Rebalance if allocation drifts by 5%
    rebalance_frequency_hours: float = 24.0  # Rebalance every 24 hours


class HorizonPortfolioAllocator:
    """Manages portfolio allocation across horizons."""
    
    def __init__(self, config: HorizonPortfolioConfig):
        """Initialize horizon portfolio allocator.
        
        Args:
            config: Horizon portfolio configuration
        """
        self.config = config
        self.allocation: Optional[PortfolioAllocation] = None
        logger.info("horizon_portfolio_allocator_initialized")
    
    def initialize(
        self,
        total_portfolio_value: float,
        current_allocations: Optional[Dict[TradingHorizon, float]] = None,
    ) -> PortfolioAllocation:
        """Initialize portfolio allocation.
        
        Args:
            total_portfolio_value: Total portfolio value in USD
            current_allocations: Current allocations by horizon (optional)
            
        Returns:
            Portfolio allocation
        """
        allocations = {}
        
        # Create allocations for each horizon
        for horizon in TradingHorizon:
            max_allocation_pct = self._get_max_allocation_pct(horizon)
            max_positions = self._get_max_positions(horizon)
            current_allocation_pct = current_allocations.get(horizon, 0.0) if current_allocations else 0.0
            
            allocations[horizon] = HorizonAllocation(
                horizon=horizon,
                max_allocation_pct=max_allocation_pct,
                current_allocation_pct=current_allocation_pct,
                target_allocation_pct=max_allocation_pct * 0.8,  # Target 80% of max
                max_positions=max_positions,
                current_positions=0,
            )
        
        self.allocation = PortfolioAllocation(
            total_portfolio_value=total_portfolio_value,
            allocations=allocations,
            last_updated=datetime.now(),
        )
        
        logger.info(
            "portfolio_allocation_initialized",
            total_portfolio_value=total_portfolio_value,
            allocations={h.value: a.max_allocation_pct for h, a in allocations.items()},
        )
        
        return self.allocation
    
    def _get_max_allocation_pct(self, horizon: TradingHorizon) -> float:
        """Get max allocation percentage for a horizon.
        
        Args:
            horizon: Trading horizon
            
        Returns:
            Max allocation percentage
        """
        mapping = {
            TradingHorizon.SCALP: self.config.scalp_max_allocation_pct,
            TradingHorizon.SWING: self.config.swing_max_allocation_pct,
            TradingHorizon.POSITION: self.config.position_max_allocation_pct,
            TradingHorizon.CORE: self.config.core_max_allocation_pct,
        }
        return mapping.get(horizon, 10.0)
    
    def _get_max_positions(self, horizon: TradingHorizon) -> int:
        """Get max positions for a horizon.
        
        Args:
            horizon: Trading horizon
            
        Returns:
            Max positions
        """
        mapping = {
            TradingHorizon.SCALP: self.config.scalp_max_positions,
            TradingHorizon.SWING: self.config.swing_max_positions,
            TradingHorizon.POSITION: self.config.position_max_positions,
            TradingHorizon.CORE: self.config.core_max_positions,
        }
        return mapping.get(horizon, 1)
    
    def can_open_position(
        self,
        horizon: TradingHorizon,
        position_size_usd: float,
    ) -> Tuple[bool, str]:
        """Check if a position can be opened.
        
        Args:
            horizon: Trading horizon
            position_size_usd: Position size in USD
            
        Returns:
            Tuple of (can_open, reason)
        """
        if not self.allocation:
            return False, "Portfolio allocation not initialized"
        
        if horizon not in self.allocation.allocations:
            return False, f"Horizon {horizon.value} not supported"
        
        allocation = self.allocation.allocations[horizon]
        
        # Check max positions
        if allocation.current_positions >= allocation.max_positions:
            return False, f"Max positions ({allocation.max_positions}) reached for {horizon.value}"
        
        # Check allocation capacity
        current_allocation_usd = (allocation.current_allocation_pct / 100.0) * self.allocation.total_portfolio_value
        new_allocation_usd = current_allocation_usd + position_size_usd
        max_allocation_usd = (allocation.max_allocation_pct / 100.0) * self.allocation.total_portfolio_value
        
        if new_allocation_usd > max_allocation_usd:
            available_capacity = max_allocation_usd - current_allocation_usd
            return False, f"Insufficient allocation capacity. Available: ${available_capacity:.2f}, Requested: ${position_size_usd:.2f}"
        
        return True, "OK"
    
    def allocate_position(
        self,
        horizon: TradingHorizon,
        position_size_usd: float,
    ) -> bool:
        """Allocate a position.
        
        Args:
            horizon: Trading horizon
            position_size_usd: Position size in USD
            
        Returns:
            True if allocated, False otherwise
        """
        if not self.allocation:
            logger.error("portfolio_allocation_not_initialized")
            return False
        
        can_open, reason = self.can_open_position(horizon, position_size_usd)
        if not can_open:
            logger.warning("position_allocation_failed", horizon=horizon.value, reason=reason)
            return False
        
        allocation = self.allocation.allocations[horizon]
        
        # Update allocation
        allocation_pct = (position_size_usd / self.allocation.total_portfolio_value) * 100.0
        allocation.current_allocation_pct += allocation_pct
        allocation.current_positions += 1
        
        self.allocation.last_updated = datetime.now()
        
        logger.info(
            "position_allocated",
            horizon=horizon.value,
            position_size_usd=position_size_usd,
            current_allocation_pct=allocation.current_allocation_pct,
            current_positions=allocation.current_positions,
        )
        
        return True
    
    def deallocate_position(
        self,
        horizon: TradingHorizon,
        position_size_usd: float,
    ) -> bool:
        """Deallocate a position.
        
        Args:
            horizon: Trading horizon
            position_size_usd: Position size in USD
            
        Returns:
            True if deallocated, False otherwise
        """
        if not self.allocation:
            logger.error("portfolio_allocation_not_initialized")
            return False
        
        if horizon not in self.allocation.allocations:
            logger.error("horizon_not_found", horizon=horizon.value)
            return False
        
        allocation = self.allocation.allocations[horizon]
        
        # Update allocation
        allocation_pct = (position_size_usd / self.allocation.total_portfolio_value) * 100.0
        allocation.current_allocation_pct = max(0.0, allocation.current_allocation_pct - allocation_pct)
        allocation.current_positions = max(0, allocation.current_positions - 1)
        
        self.allocation.last_updated = datetime.now()
        
        logger.info(
            "position_deallocated",
            horizon=horizon.value,
            position_size_usd=position_size_usd,
            current_allocation_pct=allocation.current_allocation_pct,
            current_positions=allocation.current_positions,
        )
        
        return True
    
    def update_portfolio_value(self, total_portfolio_value: float) -> None:
        """Update total portfolio value.
        
        Args:
            total_portfolio_value: Total portfolio value in USD
        """
        if not self.allocation:
            logger.error("portfolio_allocation_not_initialized")
            return
        
        self.allocation.total_portfolio_value = total_portfolio_value
        self.allocation.last_updated = datetime.now()
        
        logger.info("portfolio_value_updated", total_portfolio_value=total_portfolio_value)
    
    def get_allocation(self, horizon: TradingHorizon) -> Optional[HorizonAllocation]:
        """Get allocation for a horizon.
        
        Args:
            horizon: Trading horizon
            
        Returns:
            Horizon allocation, or None if not found
        """
        if not self.allocation:
            return None
        
        return self.allocation.allocations.get(horizon)
    
    def get_all_allocations(self) -> Dict[TradingHorizon, HorizonAllocation]:
        """Get all allocations.
        
        Returns:
            Dictionary of horizon allocations
        """
        if not self.allocation:
            return {}
        
        return self.allocation.allocations.copy()
    
    def get_available_capacity(self, horizon: TradingHorizon) -> float:
        """Get available capacity for a horizon.
        
        Args:
            horizon: Trading horizon
            
        Returns:
            Available capacity in USD
        """
        if not self.allocation:
            return 0.0
        
        return self.allocation.get_available_capacity(horizon)
    
    def get_max_position_size(self, horizon: TradingHorizon) -> float:
        """Get maximum position size for a horizon.
        
        Args:
            horizon: Trading horizon
            
        Returns:
            Maximum position size in USD
        """
        if not self.allocation:
            return 0.0
        
        return self.allocation.get_max_position_size(horizon)

