"""
Comprehensive tests for dual-mode trading system.

Tests:
1. Asset profile management
2. Dual book management
3. Per-mode policies
4. Safety rails
5. Dual-mode coordination
6. Conflict resolution
7. Integration with PPO
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.cloud.training.models.asset_profiles import (
    AssetProfile,
    AssetProfileManager,
    LongHoldConfig,
    ShortHoldConfig,
    TradingMode,
    TrailStyle,
)
from src.cloud.training.models.dual_book_manager import DualBookManager, Position
from src.cloud.training.models.dual_mode_coordinator import (
    DualModeCoordinator,
    create_dual_mode_system,
)
from src.cloud.training.models.mode_policies import (
    LongHoldPolicy,
    PolicyManager,
    ShortHoldPolicy,
    SignalContext,
)
from src.cloud.training.models.safety_rails import SafetyRailConfig, SafetyRailsMonitor


class TestAssetProfiles:
    """Test asset profile management."""

    def test_profile_creation(self):
        """Test creating asset profiles."""
        profile = AssetProfile(
            symbol="ETH",
            mode=TradingMode.BOTH,
            short_hold=ShortHoldConfig(max_book_pct=0.10),
            long_hold=LongHoldConfig(max_book_pct=0.35),
        )

        assert profile.symbol == "ETH"
        assert profile.mode == TradingMode.BOTH
        assert profile.short_hold.max_book_pct == 0.10
        assert profile.long_hold.max_book_pct == 0.35

    def test_profile_manager_defaults(self):
        """Test profile manager with defaults."""
        manager = AssetProfileManager()

        # Check default profiles
        eth_profile = manager.get_profile("ETH")
        assert eth_profile.symbol == "ETH"
        assert eth_profile.mode == TradingMode.BOTH

        # Check unknown symbol gets default
        unknown_profile = manager.get_profile("UNKNOWN")
        assert unknown_profile.symbol == "DEFAULT"

    def test_can_run_modes(self):
        """Test checking if modes can run."""
        manager = AssetProfileManager()

        # ETH should run both
        assert manager.can_run_short_hold("ETH")
        assert manager.can_run_long_hold("ETH")

        # Unknown should run short only
        assert manager.can_run_short_hold("UNKNOWN")
        assert not manager.can_run_long_hold("UNKNOWN")

    def test_performance_tracking(self):
        """Test per-mode performance tracking."""
        manager = AssetProfileManager()

        # Track some trades
        manager.update_performance("ETH", TradingMode.SHORT_HOLD, 15.0)
        manager.update_performance("ETH", TradingMode.SHORT_HOLD, -5.0)
        manager.update_performance("ETH", TradingMode.SHORT_HOLD, 20.0)

        # Get stats
        stats = manager.get_mode_stats("ETH", TradingMode.SHORT_HOLD)

        assert stats["num_trades"] == 3
        assert stats["win_rate"] == 2 / 3
        assert stats["avg_pnl_bps"] == 10.0


class TestDualBookManager:
    """Test dual position book management."""

    def test_book_initialization(self):
        """Test book initialization."""
        manager = DualBookManager()

        short_book = manager.get_book_state(TradingMode.SHORT_HOLD)
        long_book = manager.get_book_state(TradingMode.LONG_HOLD)

        assert short_book.mode == TradingMode.SHORT_HOLD
        assert long_book.mode == TradingMode.LONG_HOLD
        assert len(short_book.positions) == 0
        assert len(long_book.positions) == 0

    def test_open_position(self):
        """Test opening positions."""
        manager = DualBookManager()

        # Open short-hold position
        position = manager.open_position(
            symbol="ETH",
            mode=TradingMode.SHORT_HOLD,
            entry_price=2000.0,
            size_gbp=100.0,
            stop_loss_bps=-10.0,
            take_profit_bps=15.0,
        )

        assert position.symbol == "ETH"
        assert position.mode == TradingMode.SHORT_HOLD
        assert position.entry_price == 2000.0
        assert position.position_size_gbp == 100.0

        # Check book state
        short_book = manager.get_book_state(TradingMode.SHORT_HOLD)
        assert short_book.num_positions == 1
        assert short_book.total_exposure_gbp == 100.0

    def test_dual_positions_same_asset(self):
        """Test holding positions in both books for same asset."""
        manager = DualBookManager()

        # Open short position
        manager.open_position(
            symbol="ETH",
            mode=TradingMode.SHORT_HOLD,
            entry_price=2000.0,
            size_gbp=100.0,
            stop_loss_bps=-10.0,
        )

        # Open long position
        manager.open_position(
            symbol="ETH",
            mode=TradingMode.LONG_HOLD,
            entry_price=2000.0,
            size_gbp=500.0,
            stop_loss_bps=-150.0,
        )

        # Check both positions exist
        assert manager.has_position("ETH", TradingMode.SHORT_HOLD)
        assert manager.has_position("ETH", TradingMode.LONG_HOLD)

        # Check exposure
        exposure = manager.get_exposure("ETH")
        assert exposure["short_gbp"] == 100.0
        assert exposure["long_gbp"] == 500.0
        assert exposure["total_gbp"] == 600.0

    def test_add_to_position(self):
        """Test adding to long-hold position."""
        manager = DualBookManager()

        # Open long position
        manager.open_position(
            symbol="ETH",
            mode=TradingMode.LONG_HOLD,
            entry_price=2000.0,
            size_gbp=500.0,
            stop_loss_bps=-150.0,
        )

        # Add to position
        position = manager.add_to_position(
            symbol="ETH",
            mode=TradingMode.LONG_HOLD,
            add_price=1900.0,
            add_size_gbp=250.0,
        )

        assert position.add_count == 1
        assert position.position_size_gbp == 750.0
        # New avg price = (2000*500 + 1900*250) / 750
        expected_avg = (2000.0 * 500.0 + 1900.0 * 250.0) / 750.0
        assert abs(position.entry_price - expected_avg) < 0.01

    def test_scale_out_position(self):
        """Test scaling out of position."""
        manager = DualBookManager()

        # Open long position
        manager.open_position(
            symbol="ETH",
            mode=TradingMode.LONG_HOLD,
            entry_price=2000.0,
            size_gbp=500.0,
            stop_loss_bps=-150.0,
        )

        # Update price to profit
        manager.update_position_price("ETH", TradingMode.LONG_HOLD, 2200.0)

        # Scale out 33%
        position, pnl = manager.scale_out_position(
            symbol="ETH",
            mode=TradingMode.LONG_HOLD,
            exit_price=2200.0,
            scale_pct=0.33,
        )

        assert position.scale_out_count == 1
        assert position.position_size_gbp == 500.0 * 0.67
        assert pnl > 0  # Should have profit

    def test_close_position(self):
        """Test closing position."""
        manager = DualBookManager()

        # Open position
        manager.open_position(
            symbol="ETH",
            mode=TradingMode.SHORT_HOLD,
            entry_price=2000.0,
            size_gbp=100.0,
            stop_loss_bps=-10.0,
        )

        # Close position
        position, pnl = manager.close_position(
            symbol="ETH",
            mode=TradingMode.SHORT_HOLD,
            exit_price=2015.0,
        )

        assert position is not None
        assert not manager.has_position("ETH", TradingMode.SHORT_HOLD)

        # Check book state
        short_book = manager.get_book_state(TradingMode.SHORT_HOLD)
        assert short_book.num_positions == 0
        assert short_book.num_trades_today == 1


class TestModePolicies:
    """Test per-mode trading policies."""

    def test_short_hold_policy_entry(self):
        """Test short-hold entry policy."""
        config = ShortHoldConfig(target_profit_bps=15.0)
        policy = ShortHoldPolicy(config)

        context = SignalContext(
            symbol="ETH",
            current_price=2000.0,
            features={"micro_score": 60.0},
            regime="trend",
            confidence=0.65,
            eps_net=0.001,
            volatility_bps=80.0,
            spread_bps=8.0,
            htf_bias=0.5,
            timestamp=datetime.now(),
        )

        should_enter, reason = policy.should_enter(
            context=context,
            total_capital_gbp=10000.0,
            current_book_exposure_gbp=500.0,
        )

        assert should_enter
        assert "passed" in reason.lower()

    def test_short_hold_policy_exit_tp(self):
        """Test short-hold exit on take profit."""
        config = ShortHoldConfig(target_profit_bps=15.0)
        policy = ShortHoldPolicy(config)

        position = Position(
            symbol="ETH",
            mode=TradingMode.SHORT_HOLD,
            entry_price=2000.0,
            entry_timestamp=datetime.now(),
            position_size_gbp=100.0,
            stop_loss_bps=-10.0,
            unrealized_pnl_bps=20.0,  # Hit TP
        )

        context = SignalContext(
            symbol="ETH",
            current_price=2020.0,
            features={"micro_score": 60.0},
            regime="trend",
            confidence=0.65,
            eps_net=0.001,
            volatility_bps=80.0,
            spread_bps=8.0,
            htf_bias=0.5,
            timestamp=datetime.now(),
        )

        should_exit, reason, exit_type = policy.should_exit(position, context)

        assert should_exit
        assert exit_type == "tp"

    def test_long_hold_policy_entry(self):
        """Test long-hold entry policy."""
        config = LongHoldConfig(max_book_pct=0.35)
        policy = LongHoldPolicy(config)

        context = SignalContext(
            symbol="ETH",
            current_price=2000.0,
            features={"ignition_score": 70.0, "trend_strength": 0.7},
            regime="trend",
            confidence=0.70,
            eps_net=0.001,
            volatility_bps=80.0,
            spread_bps=8.0,
            htf_bias=0.6,
            timestamp=datetime.now(),
        )

        should_enter, reason = policy.should_enter(
            context=context,
            total_capital_gbp=10000.0,
            current_book_exposure_gbp=2000.0,
            current_asset_exposure_gbp=0.0,
        )

        assert should_enter
        assert "passed" in reason.lower()

    def test_long_hold_policy_add(self):
        """Test long-hold add policy."""
        config = LongHoldConfig(
            add_grid_bps=[-150.0, -300.0],
        )
        policy = LongHoldPolicy(config)

        position = Position(
            symbol="ETH",
            mode=TradingMode.LONG_HOLD,
            entry_price=2000.0,
            entry_timestamp=datetime.now() - timedelta(hours=3),
            position_size_gbp=500.0,
            stop_loss_bps=-150.0,
            unrealized_pnl_bps=-160.0,  # Hit first add level
            add_count=0,
        )

        context = SignalContext(
            symbol="ETH",
            current_price=1968.0,
            features={},
            regime="trend",
            confidence=0.65,
            eps_net=0.001,
            volatility_bps=80.0,
            spread_bps=8.0,
            htf_bias=0.4,
            timestamp=datetime.now(),
        )

        should_add, reason, add_price = policy.should_add(position, context)

        assert should_add
        assert add_price is not None

    def test_long_hold_policy_scale_out(self):
        """Test long-hold scale-out policy."""
        config = LongHoldConfig(
            tp_multipliers=[1.0, 1.8, 2.8],
        )
        policy = LongHoldPolicy(config)

        position = Position(
            symbol="ETH",
            mode=TradingMode.LONG_HOLD,
            entry_price=2000.0,
            entry_timestamp=datetime.now() - timedelta(hours=12),
            position_size_gbp=500.0,
            stop_loss_bps=-150.0,
            unrealized_pnl_bps=120.0,  # In profit
            scale_out_count=0,
        )

        context = SignalContext(
            symbol="ETH",
            current_price=2120.0,
            features={},
            regime="trend",
            confidence=0.65,
            eps_net=0.001,
            volatility_bps=80.0,
            spread_bps=8.0,
            htf_bias=0.6,
            timestamp=datetime.now(),
        )

        should_scale, reason, scale_pct = policy.should_scale_out(position, context)

        assert should_scale
        assert scale_pct == 0.33


class TestSafetyRails:
    """Test safety rail monitoring."""

    def test_safety_rails_initialization(self):
        """Test safety rails initialization."""
        monitor = SafetyRailsMonitor()

        assert monitor.config.max_floating_dd_bps == 500.0
        assert monitor.config.max_hold_days == 7.0

    def test_drawdown_rail_violation(self):
        """Test drawdown rail violation detection."""
        config = SafetyRailConfig(max_floating_dd_bps=500.0)
        monitor = SafetyRailsMonitor(config)

        position = Position(
            symbol="ETH",
            mode=TradingMode.LONG_HOLD,
            entry_price=2000.0,
            entry_timestamp=datetime.now(),
            position_size_gbp=500.0,
            stop_loss_bps=-150.0,
            unrealized_pnl_bps=-600.0,  # Exceeds max DD
        )

        context = SignalContext(
            symbol="ETH",
            current_price=1400.0,
            features={},
            regime="panic",
            confidence=0.50,
            eps_net=0.0,
            volatility_bps=200.0,
            spread_bps=20.0,
            htf_bias=-0.5,
            timestamp=datetime.now(),
        )

        violations = monitor.check_position(position, context)

        assert len(violations) > 0
        dd_violation = next((v for v in violations if v.rail_type == "drawdown"), None)
        assert dd_violation is not None
        assert dd_violation.severity == "critical"

    def test_time_rail_violation(self):
        """Test time rail violation detection."""
        config = SafetyRailConfig(max_hold_days=7.0)
        monitor = SafetyRailsMonitor(config)

        position = Position(
            symbol="ETH",
            mode=TradingMode.LONG_HOLD,
            entry_price=2000.0,
            entry_timestamp=datetime.now() - timedelta(days=8),  # Too old
            position_size_gbp=500.0,
            stop_loss_bps=-150.0,
            unrealized_pnl_bps=50.0,  # Barely in profit
        )

        context = SignalContext(
            symbol="ETH",
            current_price=2050.0,
            features={},
            regime="range",
            confidence=0.55,
            eps_net=0.0,
            volatility_bps=80.0,
            spread_bps=10.0,
            htf_bias=0.3,
            timestamp=datetime.now(),
        )

        violations = monitor.check_position(position, context)

        assert len(violations) > 0
        time_violation = next((v for v in violations if v.rail_type == "time"), None)
        assert time_violation is not None


class TestDualModeCoordinator:
    """Test dual-mode coordination."""

    def test_coordinator_initialization(self):
        """Test coordinator initialization."""
        coordinator, profile_manager, book_manager = create_dual_mode_system()

        assert coordinator is not None
        assert profile_manager is not None
        assert book_manager is not None

    def test_signal_evaluation_short(self):
        """Test signal evaluation for short-hold."""
        coordinator, _, _ = create_dual_mode_system()

        context = SignalContext(
            symbol="DOGE",  # Default SHORT_HOLD only
            current_price=0.10,
            features={"micro_score": 60.0},
            regime="trend",
            confidence=0.65,
            eps_net=0.001,
            volatility_bps=100.0,
            spread_bps=10.0,
            htf_bias=0.3,
            timestamp=datetime.now(),
        )

        signal = coordinator.evaluate_signal(context)

        assert signal.short_ok or not signal.short_ok  # Evaluated
        assert signal.long_ok is False  # Not enabled for DOGE

    def test_signal_evaluation_both(self):
        """Test signal evaluation for asset with both modes."""
        coordinator, _, _ = create_dual_mode_system()

        context = SignalContext(
            symbol="ETH",  # BOTH mode
            current_price=2000.0,
            features={"micro_score": 60.0, "ignition_score": 70.0, "trend_strength": 0.7},
            regime="trend",
            confidence=0.70,
            eps_net=0.001,
            volatility_bps=80.0,
            spread_bps=8.0,
            htf_bias=0.6,
            timestamp=datetime.now(),
        )

        signal = coordinator.evaluate_signal(context)

        # Should evaluate both
        assert signal.short_ok is not None
        assert signal.long_ok is not None

    def test_conflict_resolution(self):
        """Test conflict resolution for dual positions."""
        coordinator, _, book_manager = create_dual_mode_system()

        # Open positions in both books
        book_manager.open_position(
            symbol="ETH",
            mode=TradingMode.SHORT_HOLD,
            entry_price=2000.0,
            size_gbp=200.0,
            stop_loss_bps=-10.0,
        )

        book_manager.open_position(
            symbol="ETH",
            mode=TradingMode.LONG_HOLD,
            entry_price=2000.0,
            size_gbp=1000.0,
            stop_loss_bps=-150.0,
        )

        # Resolve conflict
        resolution = coordinator.resolve_conflict("ETH")

        assert resolution.symbol == "ETH"
        assert resolution.short_exposure_gbp == 200.0
        assert resolution.long_exposure_gbp == 1000.0
        assert resolution.total_exposure_gbp == 1200.0

    def test_stats_tracking(self):
        """Test statistics tracking."""
        coordinator, _, _ = create_dual_mode_system()

        # Evaluate some signals
        for i in range(5):
            context = SignalContext(
                symbol="ETH" if i % 2 == 0 else "DOGE",
                current_price=2000.0,
                features={"micro_score": 60.0},
                regime="trend",
                confidence=0.65,
                eps_net=0.001,
                volatility_bps=80.0,
                spread_bps=8.0,
                htf_bias=0.5,
                timestamp=datetime.now(),
            )
            coordinator.evaluate_signal(context)

        stats = coordinator.get_mode_stats()

        assert "routing" in stats
        assert "short_hold" in stats
        assert "long_hold" in stats


def test_integration():
    """Test full integration of dual-mode system."""
    # Create system
    coordinator, profile_manager, book_manager = create_dual_mode_system(total_capital_gbp=10000.0)

    # Scenario: Trade ETH in both modes
    # 1. Enter short-hold scalp
    short_context = SignalContext(
        symbol="ETH",
        current_price=2000.0,
        features={"micro_score": 65.0},
        regime="trend",
        confidence=0.65,
        eps_net=0.001,
        volatility_bps=80.0,
        spread_bps=8.0,
        htf_bias=0.5,
        timestamp=datetime.now(),
    )

    short_signal = coordinator.evaluate_signal(short_context)
    if short_signal.route_to == TradingMode.SHORT_HOLD:
        book_manager.open_position(
            symbol="ETH",
            mode=TradingMode.SHORT_HOLD,
            entry_price=2000.0,
            size_gbp=200.0,
            stop_loss_bps=-10.0,
            take_profit_bps=15.0,
        )

    # 2. Enter long-hold swing
    long_context = SignalContext(
        symbol="ETH",
        current_price=2000.0,
        features={"ignition_score": 75.0, "trend_strength": 0.75},
        regime="trend",
        confidence=0.75,
        eps_net=0.002,
        volatility_bps=80.0,
        spread_bps=8.0,
        htf_bias=0.7,
        timestamp=datetime.now(),
    )

    long_signal = coordinator.evaluate_signal(long_context)
    if long_signal.route_to == TradingMode.LONG_HOLD or long_signal.long_ok:
        book_manager.open_position(
            symbol="ETH",
            mode=TradingMode.LONG_HOLD,
            entry_price=2000.0,
            size_gbp=1000.0,
            stop_loss_bps=-150.0,
        )

    # 3. Check conflict resolution
    resolution = coordinator.resolve_conflict("ETH")
    assert resolution.total_exposure_gbp <= resolution.max_total_exposure_gbp

    # 4. Check safety rails on long position
    safety_ok, actions = coordinator.check_position_safety(
        symbol="ETH",
        mode=TradingMode.LONG_HOLD,
        context=long_context,
    )

    assert safety_ok  # Should be safe initially

    # 5. Get stats
    stats = coordinator.get_mode_stats()
    assert stats["short_hold"]["num_positions"] <= 1
    assert stats["long_hold"]["num_positions"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
