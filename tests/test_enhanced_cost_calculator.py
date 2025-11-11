"""
Unit Tests for Enhanced Cost Calculator

Tests for enhanced cost calculator with holding context.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from src.shared.costs.enhanced_cost_calculator import (
    EnhancedCostCalculator,
    EnhancedCostModel,
    CostBreakdown,
    TradingHorizon,
)


def test_enhanced_cost_model_initialization():
    """Test enhanced cost model initialization."""
    cost_model = EnhancedCostModel(
        symbol="BTCUSDT",
        taker_fee_bps=4.0,
        maker_fee_bps=2.0,
        median_spread_bps=5.0,
        slippage_bps_per_sigma=2.0,
        min_notional=10.0,
        step_size=0.01,
        last_updated_utc=datetime.now(timezone.utc),
        funding_rate_bps_per_8h=1.0,
        borrow_rate_bps_per_day=0.0,
        overnight_risk_multiplier=1.2,
        liquidity_decay_factor=0.95,
        spread_widening_bps_per_day=0.5,
    )
    
    assert cost_model.symbol == "BTCUSDT"
    assert cost_model.taker_fee_bps == 4.0
    assert cost_model.maker_fee_bps == 2.0
    assert cost_model.median_spread_bps == 5.0
    assert cost_model.funding_rate_bps_per_8h == 1.0
    assert cost_model.overnight_risk_multiplier == 1.2


def test_enhanced_cost_model_calculate_funding_cost():
    """Test funding cost calculation."""
    cost_model = EnhancedCostModel(
        symbol="BTCUSDT",
        taker_fee_bps=4.0,
        maker_fee_bps=2.0,
        median_spread_bps=5.0,
        slippage_bps_per_sigma=2.0,
        min_notional=10.0,
        step_size=0.01,
        last_updated_utc=datetime.now(timezone.utc),
        funding_rate_bps_per_8h=1.0,
    )
    
    # 8 hours = 1 period
    funding_cost_8h = cost_model.calculate_funding_cost(8.0, 1000.0)
    assert funding_cost_8h == 1.0
    
    # 24 hours = 3 periods
    funding_cost_24h = cost_model.calculate_funding_cost(24.0, 1000.0)
    assert funding_cost_24h == 3.0
    
    # 48 hours = 6 periods, with overnight risk multiplier
    funding_cost_48h = cost_model.calculate_funding_cost(48.0, 1000.0)
    assert funding_cost_48h > 6.0  # Should be higher due to overnight risk


def test_enhanced_cost_model_calculate_borrow_cost():
    """Test borrow cost calculation."""
    cost_model = EnhancedCostModel(
        symbol="BTCUSDT",
        taker_fee_bps=4.0,
        maker_fee_bps=2.0,
        median_spread_bps=5.0,
        slippage_bps_per_sigma=2.0,
        min_notional=10.0,
        step_size=0.01,
        last_updated_utc=datetime.now(timezone.utc),
        borrow_rate_bps_per_day=5.0,
    )
    
    # 1 day
    borrow_cost_1d = cost_model.calculate_borrow_cost(1.0, 1000.0)
    assert borrow_cost_1d == 5.0
    
    # 7 days
    borrow_cost_7d = cost_model.calculate_borrow_cost(7.0, 1000.0)
    assert borrow_cost_7d == 35.0


def test_enhanced_cost_model_calculate_liquidity_cost():
    """Test liquidity cost calculation."""
    cost_model = EnhancedCostModel(
        symbol="BTCUSDT",
        taker_fee_bps=4.0,
        maker_fee_bps=2.0,
        median_spread_bps=5.0,
        slippage_bps_per_sigma=2.0,
        min_notional=10.0,
        step_size=0.01,
        last_updated_utc=datetime.now(timezone.utc),
        liquidity_decay_factor=0.95,
        spread_widening_bps_per_day=0.5,
    )
    
    # No holding time
    liquidity_cost_0d = cost_model.calculate_liquidity_cost(0.0, 1.0)
    assert liquidity_cost_0d == 7.0  # 5.0 spread + 2.0 slippage
    
    # 1 day holding
    liquidity_cost_1d = cost_model.calculate_liquidity_cost(1.0, 1.0)
    assert liquidity_cost_1d > 7.0  # Should be higher due to decay and widening


def test_enhanced_cost_model_total_cost_bps():
    """Test total cost calculation."""
    cost_model = EnhancedCostModel(
        symbol="BTCUSDT",
        taker_fee_bps=4.0,
        maker_fee_bps=2.0,
        median_spread_bps=5.0,
        slippage_bps_per_sigma=2.0,
        min_notional=10.0,
        step_size=0.01,
        last_updated_utc=datetime.now(timezone.utc),
        funding_rate_bps_per_8h=1.0,
    )
    
    # Taker order, no holding
    total_cost_taker = cost_model.total_cost_bps(
        order_type="taker",
        order_size_sigma=1.0,
        holding_hours=0.0,
        include_funding=False,
    )
    assert total_cost_taker == 15.0  # 4.0 * 2 (entry+exit) + 5.0 (spread) + 2.0 (slippage)
    
    # Maker order, no holding
    total_cost_maker = cost_model.total_cost_bps(
        order_type="maker",
        order_size_sigma=1.0,
        holding_hours=0.0,
        include_funding=False,
    )
    assert total_cost_maker == 9.0  # 2.0 * 2 (entry+exit) + 5.0 (spread) + 2.0 (slippage)
    
    # Taker order, 24 hours holding with funding
    total_cost_24h = cost_model.total_cost_bps(
        order_type="taker",
        order_size_sigma=1.0,
        holding_hours=24.0,
        include_funding=True,
    )
    assert total_cost_24h > 15.0  # Should include funding cost


def test_enhanced_cost_calculator_initialization():
    """Test enhanced cost calculator initialization."""
    calculator = EnhancedCostCalculator()
    assert calculator.cost_models == {}


def test_enhanced_cost_calculator_register_cost_model():
    """Test cost model registration."""
    calculator = EnhancedCostCalculator()
    
    cost_model = EnhancedCostModel(
        symbol="BTCUSDT",
        taker_fee_bps=4.0,
        maker_fee_bps=2.0,
        median_spread_bps=5.0,
        slippage_bps_per_sigma=2.0,
        min_notional=10.0,
        step_size=0.01,
        last_updated_utc=datetime.now(timezone.utc),
    )
    
    calculator.register_cost_model(cost_model)
    assert "BTCUSDT" in calculator.cost_models
    assert calculator.cost_models["BTCUSDT"] == cost_model


def test_enhanced_cost_calculator_get_costs():
    """Test getting costs with holding context."""
    calculator = EnhancedCostCalculator()
    
    cost_model = EnhancedCostModel(
        symbol="BTCUSDT",
        taker_fee_bps=4.0,
        maker_fee_bps=2.0,
        median_spread_bps=5.0,
        slippage_bps_per_sigma=2.0,
        min_notional=10.0,
        step_size=0.01,
        last_updated_utc=datetime.now(timezone.utc),
        funding_rate_bps_per_8h=1.0,
    )
    
    calculator.register_cost_model(cost_model)
    
    # Scalp trade (no holding)
    cost_breakdown_scalp = calculator.get_costs(
        symbol="BTCUSDT",
        timestamp=datetime.now(timezone.utc),
        order_type="taker",
        order_size_sigma=1.0,
        holding_hours=0.0,
        horizon_type=TradingHorizon.SCALP,
        include_funding=False,
    )
    
    assert cost_breakdown_scalp.entry_fee_bps == 4.0
    assert cost_breakdown_scalp.exit_fee_bps == 4.0
    assert cost_breakdown_scalp.spread_bps == 5.0
    assert cost_breakdown_scalp.slippage_bps == 2.0
    assert cost_breakdown_scalp.funding_cost_bps == 0.0
    assert cost_breakdown_scalp.total_cost_bps > 0.0
    
    # Swing trade (24 hours holding)
    cost_breakdown_swing = calculator.get_costs(
        symbol="BTCUSDT",
        timestamp=datetime.now(timezone.utc),
        order_type="maker",
        order_size_sigma=1.0,
        holding_hours=24.0,
        horizon_type=TradingHorizon.SWING,
        include_funding=True,
    )
    
    assert cost_breakdown_swing.funding_cost_bps > 0.0
    assert cost_breakdown_swing.total_cost_bps > cost_breakdown_scalp.total_cost_bps


def test_enhanced_cost_calculator_calculate_net_edge():
    """Test net edge calculation with holding context."""
    calculator = EnhancedCostCalculator()
    
    cost_model = EnhancedCostModel(
        symbol="BTCUSDT",
        taker_fee_bps=4.0,
        maker_fee_bps=2.0,
        median_spread_bps=5.0,
        slippage_bps_per_sigma=2.0,
        min_notional=10.0,
        step_size=0.01,
        last_updated_utc=datetime.now(timezone.utc),
        funding_rate_bps_per_8h=1.0,
    )
    
    calculator.register_cost_model(cost_model)
    
    # Scalp trade
    net_edge_scalp = calculator.calculate_net_edge(
        symbol="BTCUSDT",
        edge_bps_before_costs=100.0,
        timestamp=datetime.now(timezone.utc),
        order_type="taker",
        order_size_sigma=1.0,
        holding_hours=0.0,
        horizon_type=TradingHorizon.SCALP,
        include_funding=False,
    )
    
    assert net_edge_scalp < 100.0
    assert net_edge_scalp > 0.0
    
    # Swing trade with funding
    net_edge_swing = calculator.calculate_net_edge(
        symbol="BTCUSDT",
        edge_bps_before_costs=100.0,
        timestamp=datetime.now(timezone.utc),
        order_type="maker",
        order_size_sigma=1.0,
        holding_hours=24.0,
        horizon_type=TradingHorizon.SWING,
        include_funding=True,
    )
    
    assert net_edge_swing < net_edge_scalp  # Should be lower due to funding cost


def test_enhanced_cost_calculator_should_trade():
    """Test should trade check with holding context."""
    calculator = EnhancedCostCalculator()
    
    cost_model = EnhancedCostModel(
        symbol="BTCUSDT",
        taker_fee_bps=4.0,
        maker_fee_bps=2.0,
        median_spread_bps=5.0,
        slippage_bps_per_sigma=2.0,
        min_notional=10.0,
        step_size=0.01,
        last_updated_utc=datetime.now(timezone.utc),
    )
    
    calculator.register_cost_model(cost_model)
    
    # High edge, should trade
    should_trade_high = calculator.should_trade(
        symbol="BTCUSDT",
        edge_bps_before_costs=100.0,
        timestamp=datetime.now(timezone.utc),
        net_edge_floor_bps=3.0,
        order_type="taker",
        order_size_sigma=1.0,
        holding_hours=0.0,
        horizon_type=TradingHorizon.SCALP,
    )
    assert should_trade_high is True
    
    # Low edge, should not trade
    should_trade_low = calculator.should_trade(
        symbol="BTCUSDT",
        edge_bps_before_costs=5.0,
        timestamp=datetime.now(timezone.utc),
        net_edge_floor_bps=10.0,
        order_type="taker",
        order_size_sigma=1.0,
        holding_hours=0.0,
        horizon_type=TradingHorizon.SCALP,
    )
    assert should_trade_low is False


def test_enhanced_cost_calculator_default_costs():
    """Test default costs when model not found."""
    calculator = EnhancedCostCalculator()
    
    cost_breakdown = calculator.get_costs(
        symbol="UNKNOWN",
        timestamp=datetime.now(timezone.utc),
        order_type="taker",
        order_size_sigma=1.0,
        holding_hours=0.0,
        horizon_type=TradingHorizon.SCALP,
    )
    
    assert cost_breakdown.total_cost_bps > 0.0
    assert cost_breakdown.entry_fee_bps == 4.0
    assert cost_breakdown.exit_fee_bps == 4.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

