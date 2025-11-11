"""
Unit Tests for Horizon Portfolio Allocator

Tests for horizon-based portfolio allocation.
"""

from __future__ import annotations

import pytest

from src.shared.portfolio.horizon_portfolio_allocator import (
    HorizonPortfolioAllocator,
    HorizonPortfolioConfig,
    HorizonAllocation,
    PortfolioAllocation,
)
from src.shared.engines.enhanced_engine_interface import TradingHorizon


def test_horizon_portfolio_config_initialization():
    """Test horizon portfolio config initialization."""
    config = HorizonPortfolioConfig(
        scalp_max_allocation_pct=20.0,
        swing_max_allocation_pct=40.0,
        position_max_allocation_pct=30.0,
        core_max_allocation_pct=50.0,
        scalp_max_positions=5,
        swing_max_positions=3,
        position_max_positions=2,
        core_max_positions=3,
        rebalance_threshold_pct=5.0,
        rebalance_frequency_hours=24.0,
    )
    
    assert config.scalp_max_allocation_pct == 20.0
    assert config.swing_max_allocation_pct == 40.0
    assert config.position_max_allocation_pct == 30.0
    assert config.core_max_allocation_pct == 50.0
    assert config.scalp_max_positions == 5
    assert config.swing_max_positions == 3
    assert config.position_max_positions == 2
    assert config.core_max_positions == 3


def test_horizon_portfolio_allocator_initialization():
    """Test horizon portfolio allocator initialization."""
    config = HorizonPortfolioConfig()
    allocator = HorizonPortfolioAllocator(config)
    
    assert allocator.config == config
    assert allocator.allocation is None


def test_horizon_portfolio_allocator_initialize():
    """Test portfolio allocation initialization."""
    config = HorizonPortfolioConfig()
    allocator = HorizonPortfolioAllocator(config)
    
    allocation = allocator.initialize(
        total_portfolio_value=10000.0,
        current_allocations={
            TradingHorizon.SCALP: 10.0,
            TradingHorizon.SWING: 20.0,
        },
    )
    
    assert allocation.total_portfolio_value == 10000.0
    assert len(allocation.allocations) == 4  # SCALP, SWING, POSITION, CORE
    assert allocation.allocations[TradingHorizon.SCALP].current_allocation_pct == 10.0
    assert allocation.allocations[TradingHorizon.SWING].current_allocation_pct == 20.0


def test_horizon_portfolio_allocator_can_open_position():
    """Test can open position check."""
    config = HorizonPortfolioConfig(
        swing_max_allocation_pct=40.0,
        swing_max_positions=3,
    )
    allocator = HorizonPortfolioAllocator(config)
    
    allocator.initialize(total_portfolio_value=10000.0)
    
    # Can open position within limits
    can_open, reason = allocator.can_open_position(
        horizon=TradingHorizon.SWING,
        position_size_usd=1000.0,  # 10% of portfolio
    )
    
    assert can_open is True
    assert reason == "OK"
    
    # Cannot open position exceeding allocation
    can_open, reason = allocator.can_open_position(
        horizon=TradingHorizon.SWING,
        position_size_usd=5000.0,  # 50% of portfolio (exceeds 40% limit)
    )
    
    assert can_open is False
    assert "Insufficient allocation capacity" in reason


def test_horizon_portfolio_allocator_can_open_position_max_positions():
    """Test can open position check with max positions."""
    config = HorizonPortfolioConfig(
        swing_max_allocation_pct=40.0,
        swing_max_positions=2,
    )
    allocator = HorizonPortfolioAllocator(config)
    
    allocator.initialize(total_portfolio_value=10000.0)
    
    # Open 2 positions
    allocator.allocate_position(TradingHorizon.SWING, 1000.0)
    allocator.allocate_position(TradingHorizon.SWING, 1000.0)
    
    # Cannot open third position
    can_open, reason = allocator.can_open_position(
        horizon=TradingHorizon.SWING,
        position_size_usd=1000.0,
    )
    
    assert can_open is False
    assert "Max positions" in reason


def test_horizon_portfolio_allocator_allocate_position():
    """Test allocating a position."""
    config = HorizonPortfolioConfig()
    allocator = HorizonPortfolioAllocator(config)
    
    allocator.initialize(total_portfolio_value=10000.0)
    
    # Allocate position
    success = allocator.allocate_position(
        horizon=TradingHorizon.SWING,
        position_size_usd=1000.0,
    )
    
    assert success is True
    
    allocation = allocator.get_allocation(TradingHorizon.SWING)
    assert allocation is not None
    assert allocation.current_allocation_pct == 10.0  # 1000 / 10000 * 100
    assert allocation.current_positions == 1


def test_horizon_portfolio_allocator_deallocate_position():
    """Test deallocating a position."""
    config = HorizonPortfolioConfig()
    allocator = HorizonPortfolioAllocator(config)
    
    allocator.initialize(total_portfolio_value=10000.0)
    
    # Allocate position
    allocator.allocate_position(TradingHorizon.SWING, 1000.0)
    
    # Deallocate position
    success = allocator.deallocate_position(
        horizon=TradingHorizon.SWING,
        position_size_usd=1000.0,
    )
    
    assert success is True
    
    allocation = allocator.get_allocation(TradingHorizon.SWING)
    assert allocation.current_allocation_pct == 0.0
    assert allocation.current_positions == 0


def test_horizon_portfolio_allocator_get_available_capacity():
    """Test getting available capacity."""
    config = HorizonPortfolioConfig(
        swing_max_allocation_pct=40.0,
    )
    allocator = HorizonPortfolioAllocator(config)
    
    allocator.initialize(total_portfolio_value=10000.0)
    
    # Allocate some positions
    allocator.allocate_position(TradingHorizon.SWING, 1000.0)  # 10%
    
    # Available capacity should be 30% (40% - 10%)
    available_capacity = allocator.get_available_capacity(TradingHorizon.SWING)
    assert available_capacity == 3000.0  # 30% of 10000


def test_horizon_portfolio_allocator_get_max_position_size():
    """Test getting max position size."""
    config = HorizonPortfolioConfig(
        swing_max_allocation_pct=40.0,
        swing_max_positions=3,
    )
    allocator = HorizonPortfolioAllocator(config)
    
    allocator.initialize(total_portfolio_value=10000.0)
    
    # Max allocation is 40% = 4000, divided by 3 positions = 1333.33 per position
    max_position_size = allocator.get_max_position_size(TradingHorizon.SWING)
    assert max_position_size == pytest.approx(4000.0 / 3, rel=0.01)


def test_horizon_portfolio_allocator_update_portfolio_value():
    """Test updating portfolio value."""
    config = HorizonPortfolioConfig()
    allocator = HorizonPortfolioAllocator(config)
    
    allocator.initialize(total_portfolio_value=10000.0)
    
    # Allocate position
    allocator.allocate_position(TradingHorizon.SWING, 1000.0)
    
    # Update portfolio value
    allocator.update_portfolio_value(20000.0)
    
    assert allocator.allocation.total_portfolio_value == 20000.0
    
    # Allocation percentage should remain the same (10%)
    allocation = allocator.get_allocation(TradingHorizon.SWING)
    assert allocation.current_allocation_pct == 5.0  # 1000 / 20000 * 100 = 5%


def test_horizon_portfolio_allocator_multiple_horizons():
    """Test allocation across multiple horizons."""
    config = HorizonPortfolioConfig(
        scalp_max_allocation_pct=20.0,
        swing_max_allocation_pct=40.0,
        position_max_allocation_pct=30.0,
        core_max_allocation_pct=50.0,
    )
    allocator = HorizonPortfolioAllocator(config)
    
    allocator.initialize(total_portfolio_value=10000.0)
    
    # Allocate across different horizons
    allocator.allocate_position(TradingHorizon.SCALP, 500.0)  # 5%
    allocator.allocate_position(TradingHorizon.SWING, 1000.0)  # 10%
    allocator.allocate_position(TradingHorizon.POSITION, 1500.0)  # 15%
    allocator.allocate_position(TradingHorizon.CORE, 2000.0)  # 20%
    
    # Check allocations
    scalp_allocation = allocator.get_allocation(TradingHorizon.SCALP)
    assert scalp_allocation.current_allocation_pct == 5.0
    
    swing_allocation = allocator.get_allocation(TradingHorizon.SWING)
    assert swing_allocation.current_allocation_pct == 10.0
    
    position_allocation = allocator.get_allocation(TradingHorizon.POSITION)
    assert position_allocation.current_allocation_pct == 15.0
    
    core_allocation = allocator.get_allocation(TradingHorizon.CORE)
    assert core_allocation.current_allocation_pct == 20.0


def test_horizon_portfolio_allocator_get_all_allocations():
    """Test getting all allocations."""
    config = HorizonPortfolioConfig()
    allocator = HorizonPortfolioAllocator(config)
    
    allocator.initialize(total_portfolio_value=10000.0)
    
    all_allocations = allocator.get_all_allocations()
    
    assert len(all_allocations) == 4
    assert TradingHorizon.SCALP in all_allocations
    assert TradingHorizon.SWING in all_allocations
    assert TradingHorizon.POSITION in all_allocations
    assert TradingHorizon.CORE in all_allocations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

