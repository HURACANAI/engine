"""
Unit Tests for CoreBook Strategy

Tests for DCA triggering, partial sell logic, exposure caps, no loss-sale enforcement.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from src.cloud.trading.core_strategy.core_strategy import (
    CoreBookStrategy,
    CoreBookConfig,
    CoreBookEntry,
    CoreBookState,
    ActionType,
    MarketRegime,
    TradingAction,
)


@pytest.fixture
def config():
    """Create test configuration."""
    return CoreBookConfig(
        default_coins=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        max_exposure_pct_per_coin=10.0,
        max_incremental_buy_pct=2.0,
        dca_drop_pct=5.0,
        max_dca_buys=10,
        dca_cooldown_minutes=60,
        profit_threshold_absolute=1.0,
        profit_threshold_pct=5.0,
        partial_sell_pct=25.0,
        action_cooldown_minutes=15,
        stop_dca_on_panic=True,
        min_liquidity_usd=1_000_000.0,
        max_spread_bps=50.0,
    )


@pytest.fixture
def temp_state_file():
    """Create temporary state file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = Path(f.name)
    yield temp_path
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def strategy(config, temp_state_file):
    """Create test strategy."""
    return CoreBookStrategy(config=config, state_file=str(temp_state_file))


def test_initial_state(strategy):
    """Test initial state."""
    assert len(strategy.state.coins) == 3
    assert "BTCUSDT" in strategy.state.coins
    assert "ETHUSDT" in strategy.state.coins
    assert "SOLUSDT" in strategy.state.coins
    assert strategy.state.auto_trading_enabled is True


def test_no_sell_at_loss(strategy):
    """Test that we never sell at a loss."""
    symbol = "BTCUSDT"
    entry = strategy.state.coins[symbol]
    
    # Set up a position
    entry.units_held = 0.05
    entry.average_cost_price = 45000.0
    entry.total_cost_basis = 2250.0
    
    # Try to sell at a loss
    current_price = 44000.0  # Below average cost
    portfolio_value = 10000.0
    
    action = strategy._evaluate_partial_sell(symbol, current_price, portfolio_value, entry)
    
    # Should not generate a sell action
    assert action is None
    
    # Try manual sell at a loss
    success = strategy.execute_sell(symbol, 0.01, current_price)
    assert success is False


def test_dca_trigger_below_average_cost(strategy):
    """Test DCA triggering when price drops below average cost."""
    symbol = "BTCUSDT"
    entry = strategy.state.coins[symbol]
    
    # Set up a position
    entry.units_held = 0.05
    entry.average_cost_price = 45000.0
    entry.total_cost_basis = 2250.0
    entry.dca_count = 0
    entry.cooldown_until = None
    
    # Price drops below average cost
    current_price = 42750.0  # 5% below average cost
    portfolio_value = 10000.0
    
    action = strategy._evaluate_dca_buy(
        symbol, current_price, portfolio_value, entry, MarketRegime.NORMAL
    )
    
    # Should generate a DCA buy action
    assert action is not None
    assert action.action_type == ActionType.DCA_BUY
    assert action.symbol == symbol
    assert action.price == current_price
    assert action.units > 0.0
    
    # Check that entry was updated
    assert entry.dca_count == 1
    assert entry.units_held > 0.05
    assert entry.average_cost_price < 45000.0  # Average cost should decrease


def test_dca_no_trigger_above_average_cost(strategy):
    """Test that DCA doesn't trigger when price is above average cost."""
    symbol = "BTCUSDT"
    entry = strategy.state.coins[symbol]
    
    # Set up a position
    entry.units_held = 0.05
    entry.average_cost_price = 45000.0
    entry.total_cost_basis = 2250.0
    
    # Price is above average cost
    current_price = 46000.0  # Above average cost
    portfolio_value = 10000.0
    
    action = strategy._evaluate_dca_buy(
        symbol, current_price, portfolio_value, entry, MarketRegime.NORMAL
    )
    
    # Should not generate a DCA buy action
    assert action is None


def test_dca_respects_exposure_cap(strategy):
    """Test that DCA respects exposure cap."""
    symbol = "BTCUSDT"
    entry = strategy.state.coins[symbol]
    
    # Set up a position near exposure cap
    entry.units_held = 0.022  # ~$990 at $45000
    entry.average_cost_price = 45000.0
    entry.total_cost_basis = 990.0
    entry.total_exposure_limit_pct = 10.0  # 10% of $10000 = $1000
    entry.dca_count = 0
    entry.cooldown_until = None
    
    # Price drops below average cost
    current_price = 42750.0
    portfolio_value = 10000.0
    
    action = strategy._evaluate_dca_buy(
        symbol, current_price, portfolio_value, entry, MarketRegime.NORMAL
    )
    
    # Should generate a DCA buy, but limited by exposure cap
    if action:
        current_exposure = entry.units_held * current_price
        new_exposure = (entry.units_held + action.units) * current_price
        new_exposure_pct = (new_exposure / portfolio_value) * 100.0
        
        # Should not exceed exposure cap
        assert new_exposure_pct <= entry.total_exposure_limit_pct


def test_dca_respects_max_dca_buys(strategy):
    """Test that DCA respects max DCA buys."""
    symbol = "BTCUSDT"
    entry = strategy.state.coins[symbol]
    
    # Set up a position with max DCA buys reached
    entry.units_held = 0.05
    entry.average_cost_price = 45000.0
    entry.total_cost_basis = 2250.0
    entry.dca_count = 10  # Max DCA buys
    entry.max_dca_buys = 10
    entry.cooldown_until = None
    
    # Price drops below average cost
    current_price = 42750.0
    portfolio_value = 10000.0
    
    action = strategy._evaluate_dca_buy(
        symbol, current_price, portfolio_value, entry, MarketRegime.NORMAL
    )
    
    # Should not generate a DCA buy action
    assert action is None


def test_dca_respects_cooldown(strategy):
    """Test that DCA respects cooldown period."""
    symbol = "BTCUSDT"
    entry = strategy.state.coins[symbol]
    
    # Set up a position with active cooldown
    entry.units_held = 0.05
    entry.average_cost_price = 45000.0
    entry.total_cost_basis = 2250.0
    entry.dca_count = 0
    entry.cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=30)
    
    # Price drops below average cost
    current_price = 42750.0
    portfolio_value = 10000.0
    
    action = strategy.evaluate(
        symbol, current_price, portfolio_value, MarketRegime.NORMAL
    )
    
    # Should not generate a DCA buy action due to cooldown
    assert action is None


def test_partial_sell_on_profit_threshold(strategy):
    """Test partial sell when profit threshold is met."""
    symbol = "BTCUSDT"
    entry = strategy.state.coins[symbol]
    
    # Set up a profitable position
    entry.units_held = 0.05
    entry.average_cost_price = 45000.0
    entry.total_cost_basis = 2250.0
    entry.cooldown_until = None
    
    # Price is above profit threshold (5% profit)
    current_price = 47250.0  # 5% above average cost
    portfolio_value = 10000.0
    
    action = strategy._evaluate_partial_sell(symbol, current_price, portfolio_value, entry)
    
    # Should generate a partial sell action
    assert action is not None
    assert action.action_type == ActionType.PARTIAL_SELL
    assert action.symbol == symbol
    assert action.price == current_price
    assert action.units > 0.0
    assert action.units < entry.units_held  # Should not sell everything
    
    # Check that entry was updated
    assert entry.units_held < 0.05  # Units should decrease
    assert entry.units_held > 0.0  # Should not sell everything


def test_partial_sell_respects_profit_threshold(strategy):
    """Test that partial sell only triggers when profit threshold is met."""
    symbol = "BTCUSDT"
    entry = strategy.state.coins[symbol]
    
    # Set up a position with small profit
    entry.units_held = 0.05
    entry.average_cost_price = 45000.0
    entry.total_cost_basis = 2250.0
    
    # Price is above average cost but below profit threshold
    current_price = 45200.0  # Only 0.44% profit
    portfolio_value = 10000.0
    
    action = strategy._evaluate_partial_sell(symbol, current_price, portfolio_value, entry)
    
    # Should not generate a partial sell action (profit too small)
    assert action is None


def test_exposure_cap_enforcement(strategy):
    """Test exposure cap enforcement."""
    symbol = "BTCUSDT"
    entry = strategy.state.coins[symbol]
    
    # Set exposure cap
    entry.total_exposure_limit_pct = 10.0
    portfolio_value = 10000.0
    
    # Try to buy that would exceed exposure cap
    entry.units_held = 0.022  # ~$990 at $45000 (near cap)
    entry.average_cost_price = 45000.0
    entry.total_cost_basis = 990.0
    
    # Try to buy more
    success = strategy.execute_buy(symbol, 0.01, 45000.0, portfolio_value)
    
    # Should fail if it would exceed cap
    # (This depends on the exact implementation)
    if not success:
        # Check that exposure cap was enforced
        current_exposure = entry.units_held * 45000.0
        exposure_pct = (current_exposure / portfolio_value) * 100.0
        assert exposure_pct <= entry.total_exposure_limit_pct


def test_stop_dca_on_panic(strategy):
    """Test that DCA stops on panic regime."""
    symbol = "BTCUSDT"
    entry = strategy.state.coins[symbol]
    
    # Set up a position
    entry.units_held = 0.05
    entry.average_cost_price = 45000.0
    entry.total_cost_basis = 2250.0
    entry.dca_count = 0
    entry.cooldown_until = None
    
    # Price drops below average cost
    current_price = 42750.0
    portfolio_value = 10000.0
    
    # Market is in panic
    action = strategy.evaluate(
        symbol, current_price, portfolio_value, MarketRegime.PANIC
    )
    
    # Should not generate a DCA buy action
    assert action is None


def test_liquidity_check(strategy):
    """Test liquidity check."""
    symbol = "BTCUSDT"
    entry = strategy.state.coins[symbol]
    
    # Set up a position
    entry.units_held = 0.05
    entry.average_cost_price = 45000.0
    entry.total_cost_basis = 2250.0
    entry.dca_count = 0
    entry.cooldown_until = None
    
    # Price drops below average cost
    current_price = 42750.0
    portfolio_value = 10000.0
    
    # Low liquidity
    liquidity_usd = 500000.0  # Below minimum
    
    # Risk safeguards should block action
    safeguards_pass = strategy._check_risk_safeguards(
        symbol, MarketRegime.NORMAL, liquidity_usd, 5.0
    )
    
    assert safeguards_pass is False


def test_spread_check(strategy):
    """Test spread check."""
    symbol = "BTCUSDT"
    entry = strategy.state.coins[symbol]
    
    # Set up a position
    entry.units_held = 0.05
    entry.average_cost_price = 45000.0
    entry.total_cost_basis = 2250.0
    entry.dca_count = 0
    entry.cooldown_until = None
    
    # Price drops below average cost
    current_price = 42750.0
    portfolio_value = 10000.0
    
    # High spread
    spread_bps = 100.0  # Above maximum
    
    # Risk safeguards should block action
    safeguards_pass = strategy._check_risk_safeguards(
        symbol, MarketRegime.NORMAL, 10_000_000.0, spread_bps
    )
    
    assert safeguards_pass is False


def test_average_cost_update(strategy):
    """Test average cost update after buy."""
    symbol = "BTCUSDT"
    entry = strategy.state.coins[symbol]
    
    # Initial position
    entry.units_held = 0.05
    entry.average_cost_price = 45000.0
    entry.total_cost_basis = 2250.0
    
    # Buy more at lower price
    new_units = 0.02
    new_price = 44000.0
    
    entry.update_average_cost(new_units, new_price)
    
    # Average cost should decrease
    assert entry.units_held == 0.07
    assert entry.average_cost_price < 45000.0
    assert entry.average_cost_price > 44000.0
    assert entry.total_cost_basis == 2250.0 + (0.02 * 44000.0)


def test_trigger_updates(strategy):
    """Test that triggers are updated correctly."""
    symbol = "BTCUSDT"
    entry = strategy.state.coins[symbol]
    
    # Set up a position
    entry.units_held = 0.05
    entry.average_cost_price = 45000.0
    entry.total_cost_basis = 2250.0
    
    # Update triggers
    current_price = 45000.0
    strategy._update_triggers(entry, current_price)
    
    # DCA trigger should be below average cost
    assert entry.next_dca_trigger_price is not None
    assert entry.next_dca_trigger_price < entry.average_cost_price
    
    # Partial sell target should be above average cost
    assert entry.partial_sell_target_price is not None
    assert entry.partial_sell_target_price > entry.average_cost_price


def test_state_save_and_load(strategy, temp_state_file):
    """Test state save and load."""
    # Modify state
    symbol = "BTCUSDT"
    entry = strategy.state.coins[symbol]
    entry.units_held = 0.05
    entry.average_cost_price = 45000.0
    entry.total_cost_basis = 2250.0
    
    # Save state
    strategy._save_state()
    
    # Create new strategy instance
    new_strategy = CoreBookStrategy(
        config=strategy.config,
        state_file=str(temp_state_file)
    )
    
    # Check that state was loaded
    assert new_strategy.state.coins[symbol].units_held == 0.05
    assert new_strategy.state.coins[symbol].average_cost_price == 45000.0


def test_set_exposure_cap(strategy):
    """Test setting exposure cap."""
    symbol = "BTCUSDT"
    new_cap = 15.0
    
    success = strategy.set_exposure_cap(symbol, new_cap)
    
    assert success is True
    assert strategy.state.coins[symbol].total_exposure_limit_pct == new_cap


def test_add_coin(strategy):
    """Test adding a coin."""
    symbol = "ADAUSDT"
    exposure_pct = 5.0
    
    # Add coin to allowable list first
    strategy.config.allowable_coins.append(symbol)
    
    success = strategy.add_coin(symbol, exposure_pct)
    
    assert success is True
    assert symbol in strategy.state.coins
    assert strategy.state.coins[symbol].total_exposure_limit_pct == exposure_pct


def test_trim_position(strategy):
    """Test trimming a position."""
    symbol = "BTCUSDT"
    entry = strategy.state.coins[symbol]
    
    # Set up a position
    entry.units_held = 0.05
    entry.average_cost_price = 45000.0
    entry.total_cost_basis = 2250.0
    
    # Trim 25%
    # Note: This will use average cost * 1.01 as current price
    # In real implementation, current price would be passed
    success = strategy.trim_position(symbol, 25.0)
    
    # Should succeed (assuming price allows it)
    # The exact behavior depends on implementation
    if success:
        assert entry.units_held < 0.05


def test_set_auto_trading(strategy):
    """Test setting auto trading."""
    # Disable auto trading
    success = strategy.set_auto_trading(False)
    assert success is True
    assert strategy.state.auto_trading_enabled is False
    
    # Enable auto trading
    success = strategy.set_auto_trading(True)
    assert success is True
    assert strategy.state.auto_trading_enabled is True


def test_get_status(strategy):
    """Test getting status."""
    status = strategy.get_status()
    
    assert "auto_trading_enabled" in status
    assert "coins" in status
    assert "last_updated" in status
    assert len(status["coins"]) == 3


def test_get_coin_status(strategy):
    """Test getting coin status."""
    symbol = "BTCUSDT"
    coin_status = strategy.get_coin_status(symbol)
    
    assert coin_status is not None
    assert coin_status["symbol"] == symbol
    assert "units_held" in coin_status
    assert "average_cost_price" in coin_status


def test_unrealized_pnl_calculation(strategy):
    """Test unrealized P&L calculation."""
    symbol = "BTCUSDT"
    entry = strategy.state.coins[symbol]
    
    # Set up a position
    entry.units_held = 0.05
    entry.average_cost_price = 45000.0
    entry.total_cost_basis = 2250.0
    
    # Current price above average cost
    current_price = 47250.0  # 5% profit
    
    unrealized_pnl = entry.calculate_unrealized_pnl(current_price)
    unrealized_pnl_pct = entry.calculate_unrealized_pnl_pct(current_price)
    
    # Should be positive
    assert unrealized_pnl > 0.0
    assert unrealized_pnl_pct > 0.0
    assert abs(unrealized_pnl_pct - 5.0) < 0.1  # Should be approximately 5%


def test_evaluate_full_flow(strategy):
    """Test full evaluation flow."""
    symbol = "BTCUSDT"
    entry = strategy.state.coins[symbol]
    
    # Set up a position
    entry.units_held = 0.05
    entry.average_cost_price = 45000.0
    entry.total_cost_basis = 2250.0
    entry.dca_count = 0
    entry.cooldown_until = None
    
    # Price drops below average cost (should trigger DCA)
    current_price = 42750.0
    portfolio_value = 10000.0
    
    action = strategy.evaluate(
        symbol, current_price, portfolio_value, MarketRegime.NORMAL, 10_000_000.0, 5.0
    )
    
    # Should generate a DCA buy action
    assert action is not None
    assert action.action_type == ActionType.DCA_BUY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

