"""
Tests for Risk Preset System

Example test suite for risk management.
"""

import pytest

from src.cloud.training.risk.risk_presets import (
    RiskPresetManager,
    RiskPreset,
)


@pytest.fixture
def manager():
    """Test risk preset manager."""
    return RiskPresetManager(default_preset=RiskPreset.BALANCED)


def test_get_limits(manager):
    """Test getting risk limits."""
    limits = manager.get_limits(RiskPreset.CONSERVATIVE)
    
    assert limits.per_trade_risk_pct == 0.5
    assert limits.daily_loss_limit_pct == 1.5
    assert limits.max_leverage == 1.5
    assert limits.min_confidence == 0.65


def test_check_trade_allowed(manager):
    """Test trade validation."""
    # Valid trade
    allowed, reason = manager.check_trade_allowed(
        confidence=0.6,
        position_size_pct=15.0,
        daily_loss_pct=1.0,
        daily_trade_count=50,
        current_leverage=1.5,
    )
    
    assert allowed
    assert reason is None
    
    # Low confidence
    allowed, reason = manager.check_trade_allowed(
        confidence=0.4,  # Below minimum
        position_size_pct=15.0,
        daily_loss_pct=1.0,
        daily_trade_count=50,
        current_leverage=1.5,
    )
    
    assert not allowed
    assert "Confidence" in reason


def test_calculate_position_size(manager):
    """Test position size calculation."""
    equity = 100000
    stop_loss_pct = 2.0
    
    position_size = manager.calculate_position_size(equity, stop_loss_pct)
    
    # Should be based on risk per trade
    limits = manager.get_limits()
    expected_risk = equity * (limits.per_trade_risk_pct / 100.0)
    expected_size = expected_risk / (stop_loss_pct / 100.0)
    
    assert position_size > 0
    assert position_size <= expected_size


def test_daily_limit_check(manager):
    """Test daily loss limit check."""
    # Below limit
    assert not manager.check_daily_limit_exceeded(1.0)
    
    # At limit
    limits = manager.get_limits()
    assert manager.check_daily_limit_exceeded(limits.daily_loss_limit_pct)
    
    # Above limit
    assert manager.check_daily_limit_exceeded(limits.daily_loss_limit_pct + 0.1)

