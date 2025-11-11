"""
Unit Tests for Swing Position Manager

Tests for swing position manager with stop-loss, take-profit, and holding logic.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta

from src.shared.trading.swing_position_manager import (
    SwingPositionManager,
    SwingPositionConfig,
    SwingPosition,
    StopLossLevel,
    TakeProfitLevel,
    ExitReason,
)
from src.shared.engines.enhanced_engine_interface import TradingHorizon, Direction


def test_swing_position_config_initialization():
    """Test swing position config initialization."""
    config = SwingPositionConfig(
        default_stop_loss_bps=200.0,
        use_trailing_stop=True,
        trailing_stop_distance_bps=100.0,
        take_profit_levels=[(200.0, 0.30), (400.0, 0.40)],
        max_holding_hours=48.0,
        max_funding_cost_bps=500.0,
        exit_on_panic=True,
    )
    
    assert config.default_stop_loss_bps == 200.0
    assert config.use_trailing_stop is True
    assert config.trailing_stop_distance_bps == 100.0
    assert len(config.take_profit_levels) == 2
    assert config.max_holding_hours == 48.0
    assert config.max_funding_cost_bps == 500.0
    assert config.exit_on_panic is True


def test_swing_position_manager_initialization():
    """Test swing position manager initialization."""
    config = SwingPositionConfig()
    manager = SwingPositionManager(config)
    
    assert manager.config == config
    assert manager.positions == {}


def test_swing_position_manager_open_position():
    """Test opening a swing position."""
    config = SwingPositionConfig()
    manager = SwingPositionManager(config)
    
    position = manager.open_position(
        symbol="BTCUSDT",
        direction=Direction.BUY,
        entry_price=45000.0,
        entry_size=0.1,
        horizon_type=TradingHorizon.SWING,
        stop_loss_bps=200.0,
        take_profit_levels=[(200.0, 0.30), (400.0, 0.40)],
        trailing_stop_bps=100.0,
    )
    
    assert position.symbol == "BTCUSDT"
    assert position.direction == Direction.BUY
    assert position.entry_price == 45000.0
    assert position.entry_size == 0.1
    assert position.current_size == 0.1
    assert position.horizon_type == TradingHorizon.SWING
    assert position.stop_loss is not None
    assert len(position.take_profit_levels) == 2
    assert position.trailing_stop is not None
    assert "BTCUSDT" in manager.positions


def test_swing_position_update_price():
    """Test updating position price."""
    config = SwingPositionConfig()
    manager = SwingPositionManager(config)
    
    position = manager.open_position(
        symbol="BTCUSDT",
        direction=Direction.BUY,
        entry_price=45000.0,
        entry_size=0.1,
        horizon_type=TradingHorizon.SWING,
    )
    
    # Update with profit
    position.update_price(46000.0, funding_cost_bps=5.0)
    
    assert position.current_price == 46000.0
    assert position.unrealized_pnl_bps > 0.0
    assert position.unrealized_pnl_usd > 0.0
    assert position.funding_cost_accumulated_bps == 5.0


def test_swing_position_get_holding_duration_hours():
    """Test holding duration calculation."""
    config = SwingPositionConfig()
    manager = SwingPositionManager(config)
    
    position = manager.open_position(
        symbol="BTCUSDT",
        direction=Direction.BUY,
        entry_price=45000.0,
        entry_size=0.1,
        horizon_type=TradingHorizon.SWING,
    )
    
    # Wait a bit
    position.entry_timestamp = datetime.now(timezone.utc) - timedelta(hours=12)
    position.last_updated = datetime.now(timezone.utc)
    
    holding_hours = position.get_holding_duration_hours()
    assert holding_hours == 12.0


def test_swing_position_should_exit_stop_loss():
    """Test stop loss exit."""
    config = SwingPositionConfig()
    manager = SwingPositionManager(config)
    
    position = manager.open_position(
        symbol="BTCUSDT",
        direction=Direction.BUY,
        entry_price=45000.0,
        entry_size=0.1,
        horizon_type=TradingHorizon.SWING,
        stop_loss_bps=200.0,  # 2% stop loss = 44100.0
    )
    
    # Price drops below stop loss
    should_exit, exit_reason, exit_percentage = position.should_exit(
        current_price=44000.0,
        current_regime="TREND",
        funding_cost_bps=0.0,
    )
    
    assert should_exit is True
    assert exit_reason == ExitReason.STOP_LOSS
    assert exit_percentage == 1.0


def test_swing_position_should_exit_take_profit():
    """Test take profit exit."""
    config = SwingPositionConfig()
    manager = SwingPositionManager(config)
    
    position = manager.open_position(
        symbol="BTCUSDT",
        direction=Direction.BUY,
        entry_price=45000.0,
        entry_size=0.1,
        horizon_type=TradingHorizon.SWING,
        take_profit_levels=[(200.0, 0.30)],  # 2% profit = 45900.0
    )
    
    # Price reaches take profit
    should_exit, exit_reason, exit_percentage = position.should_exit(
        current_price=45900.0,
        current_regime="TREND",
        funding_cost_bps=0.0,
    )
    
    assert should_exit is True
    assert exit_reason == ExitReason.TAKE_PROFIT
    assert exit_percentage == 0.30


def test_swing_position_should_exit_time_limit():
    """Test time limit exit."""
    config = SwingPositionConfig()
    manager = SwingPositionManager(config)
    
    position = manager.open_position(
        symbol="BTCUSDT",
        direction=Direction.BUY,
        entry_price=45000.0,
        entry_size=0.1,
        horizon_type=TradingHorizon.SWING,
        max_holding_hours=48.0,
    )
    
    # Set entry time to 49 hours ago
    position.entry_timestamp = datetime.now(timezone.utc) - timedelta(hours=49)
    position.last_updated = datetime.now(timezone.utc)
    
    should_exit, exit_reason, exit_percentage = position.should_exit(
        current_price=45000.0,
        current_regime="TREND",
        funding_cost_bps=0.0,
    )
    
    assert should_exit is True
    assert exit_reason == ExitReason.TIME_LIMIT
    assert exit_percentage == 1.0


def test_swing_position_should_exit_funding_cost():
    """Test funding cost exit."""
    config = SwingPositionConfig()
    manager = SwingPositionManager(config)
    
    position = manager.open_position(
        symbol="BTCUSDT",
        direction=Direction.BUY,
        entry_price=45000.0,
        entry_size=0.1,
        horizon_type=TradingHorizon.SWING,
        max_funding_cost_bps=500.0,
    )
    
    # Accumulate funding cost
    position.funding_cost_accumulated_bps = 500.0
    
    should_exit, exit_reason, exit_percentage = position.should_exit(
        current_price=45000.0,
        current_regime="TREND",
        funding_cost_bps=1.0,  # This would push it over
    )
    
    assert should_exit is True
    assert exit_reason == ExitReason.FUNDING_COST
    assert exit_percentage == 1.0


def test_swing_position_should_exit_regime_change():
    """Test regime change exit."""
    config = SwingPositionConfig(exit_on_panic=True)
    manager = SwingPositionManager(config)
    
    position = manager.open_position(
        symbol="BTCUSDT",
        direction=Direction.BUY,
        entry_price=45000.0,
        entry_size=0.1,
        horizon_type=TradingHorizon.SWING,
    )
    
    # Panic regime
    should_exit, exit_reason, exit_percentage = position.should_exit(
        current_price=45000.0,
        current_regime="PANIC",
        funding_cost_bps=0.0,
    )
    
    assert should_exit is True
    assert exit_reason == ExitReason.REGIME_CHANGE
    assert exit_percentage == 1.0


def test_swing_position_partial_exit():
    """Test partial exit."""
    config = SwingPositionConfig()
    manager = SwingPositionManager(config)
    
    position = manager.open_position(
        symbol="BTCUSDT",
        direction=Direction.BUY,
        entry_price=45000.0,
        entry_size=0.1,
        horizon_type=TradingHorizon.SWING,
    )
    
    # Partial exit at profit
    exit_details = position.partial_exit(0.30, 46000.0)
    
    assert exit_details["exit_size"] == 0.03  # 30% of 0.1
    assert exit_details["exit_price"] == 46000.0
    assert exit_details["realized_pnl_bps"] > 0.0
    assert position.current_size == 0.07  # Remaining 70%
    assert position.realized_pnl_bps > 0.0


def test_swing_position_manager_update_position():
    """Test updating position and checking for exits."""
    config = SwingPositionConfig()
    manager = SwingPositionManager(config)
    
    position = manager.open_position(
        symbol="BTCUSDT",
        direction=Direction.BUY,
        entry_price=45000.0,
        entry_size=0.1,
        horizon_type=TradingHorizon.SWING,
        take_profit_levels=[(200.0, 0.30)],  # 2% profit = 45900.0
    )
    
    # Update with take profit hit
    exit_action = manager.update_position(
        symbol="BTCUSDT",
        current_price=45900.0,
        current_regime="TREND",
        funding_cost_bps=0.0,
    )
    
    assert exit_action is not None
    assert exit_action["exit_reason"] == ExitReason.TAKE_PROFIT.value
    assert exit_action["exit_size"] > 0.0
    assert position.current_size < 0.1  # Partially exited


def test_swing_position_manager_close_position():
    """Test closing a position."""
    config = SwingPositionConfig()
    manager = SwingPositionManager(config)
    
    position = manager.open_position(
        symbol="BTCUSDT",
        direction=Direction.BUY,
        entry_price=45000.0,
        entry_size=0.1,
        horizon_type=TradingHorizon.SWING,
    )
    
    # Update price to profit
    position.update_price(46000.0, funding_cost_bps=0.0)
    
    # Close position
    exit_details = manager.close_position("BTCUSDT", "manual")
    
    assert exit_details["symbol"] == "BTCUSDT"
    assert exit_details["exit_reason"] == "manual"
    assert exit_details["entry_price"] == 45000.0
    assert exit_details["exit_price"] == 46000.0
    assert "BTCUSDT" not in manager.positions  # Position removed


def test_swing_position_manager_get_position():
    """Test getting a position."""
    config = SwingPositionConfig()
    manager = SwingPositionManager(config)
    
    position = manager.open_position(
        symbol="BTCUSDT",
        direction=Direction.BUY,
        entry_price=45000.0,
        entry_size=0.1,
        horizon_type=TradingHorizon.SWING,
    )
    
    retrieved_position = manager.get_position("BTCUSDT")
    assert retrieved_position == position
    
    assert manager.get_position("ETHUSDT") is None


def test_swing_position_trailing_stop():
    """Test trailing stop functionality."""
    config = SwingPositionConfig(use_trailing_stop=True, trailing_stop_distance_bps=100.0)
    manager = SwingPositionManager(config)
    
    position = manager.open_position(
        symbol="BTCUSDT",
        direction=Direction.BUY,
        entry_price=45000.0,
        entry_size=0.1,
        horizon_type=TradingHorizon.SWING,
        trailing_stop_bps=100.0,
    )
    
    # Price moves up
    position.update_price(46000.0, funding_cost_bps=0.0)
    
    # Trailing stop should have moved up
    assert position.trailing_stop is not None
    assert position.trailing_stop.is_trailing is True
    assert position.trailing_stop.price > 45000.0 * (1 - 200.0 / 10000.0)  # Original stop loss
    
    # Price drops to trailing stop
    should_exit, exit_reason, exit_percentage = position.should_exit(
        current_price=position.trailing_stop.price,
        current_regime="TREND",
        funding_cost_bps=0.0,
    )
    
    assert should_exit is True
    assert exit_reason == ExitReason.TRAILING_STOP


def test_swing_position_multiple_take_profit_levels():
    """Test multiple take profit levels."""
    config = SwingPositionConfig()
    manager = SwingPositionManager(config)
    
    position = manager.open_position(
        symbol="BTCUSDT",
        direction=Direction.BUY,
        entry_price=45000.0,
        entry_size=0.1,
        horizon_type=TradingHorizon.SWING,
        take_profit_levels=[
            (200.0, 0.30),  # 30% at 2% profit
            (400.0, 0.40),  # 40% at 4% profit
            (600.0, 0.20),  # 20% at 6% profit
        ],
    )
    
    # First take profit level
    exit_action = manager.update_position(
        symbol="BTCUSDT",
        current_price=45900.0,  # 2% profit
        current_regime="TREND",
        funding_cost_bps=0.0,
    )
    
    assert exit_action is not None
    assert exit_action["exit_reason"] == ExitReason.TAKE_PROFIT.value
    assert position.current_size == 0.07  # 70% remaining
    
    # Second take profit level
    exit_action = manager.update_position(
        symbol="BTCUSDT",
        current_price=46800.0,  # 4% profit
        current_regime="TREND",
        funding_cost_bps=0.0,
    )
    
    assert exit_action is not None
    assert exit_action["exit_reason"] == ExitReason.TAKE_PROFIT.value
    assert position.current_size < 0.07  # Further reduced


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

