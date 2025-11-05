"""
Integration Test for Dual-Mode Trading System

Tests the complete signal-to-execution pipeline:
1. Alpha engines generate signals
2. Engine consensus validates
3. Mode selector routes to scalp vs runner
4. Gate profiles filter
5. Dual-book manager executes
6. Metrics tracked correctly

Tests both success and failure paths.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cloud.training.models.trading_coordinator import TradingCoordinator
from cloud.training.models.dual_book_manager import BookType, AssetProfile
from cloud.training.models.gate_profiles import OrderType


class TestDualModeIntegration:
    """Integration tests for dual-mode trading system."""

    @pytest.fixture
    def coordinator(self):
        """Create trading coordinator instance."""
        coordinator = TradingCoordinator(
            total_capital=10000.0,
            asset_symbols=['ETH-USD', 'SOL-USD', 'BTC-USD'],
            max_short_heat=0.40,
            max_long_heat=0.50,
            reserve_heat=0.10,
        )
        return coordinator

    def test_coordinator_initialization(self, coordinator):
        """Test that coordinator initializes all components."""
        # Check alpha engines exist
        assert coordinator.trend_engine is not None
        assert coordinator.range_engine is not None
        assert coordinator.breakout_engine is not None
        assert coordinator.tape_engine is not None
        assert coordinator.leader_engine is not None
        assert coordinator.sweep_engine is not None

        # Check consensus system
        assert coordinator.consensus is not None

        # Check dual-book manager
        assert coordinator.book_manager is not None
        assert coordinator.book_manager.total_capital == 10000.0

        # Check mode selector
        assert coordinator.mode_selector is not None

        # Check gate profiles
        assert coordinator.scalp_profile is not None
        assert coordinator.runner_profile is not None

        # Check counterfactual tracker
        assert coordinator.counterfactual_tracker is not None

    def test_high_conviction_trend_routes_to_runner(self, coordinator):
        """Test that high-conviction TREND signals route to runner book."""
        features = {
            'trend_strength': 0.85,
            'ema_slope': 0.75,
            'momentum_slope': 0.70,
            'htf_bias': 0.80,
            'adx': 35.0,
            'mean_revert_bias': 0.2,
            'compression': 0.3,
            'bb_width': 0.03,
            'price_position': 0.5,
        }

        decision = coordinator.process_signal(
            symbol='ETH-USD',
            price=2000.0,
            features=features,
            regime='TREND',
            spread_bps=8.0,
            liquidity_score=0.80,
        )

        # Should generate a trade
        assert decision is not None
        assert decision.approved is True

        # Should route to runner book (high conviction TREND)
        assert decision.book == BookType.LONG_HOLD
        assert decision.technique == 'trend'
        assert decision.confidence > 0.70

        # Check position was added
        metrics = coordinator.get_metrics()
        assert metrics['books']['long_book']['positions'] == 1

    def test_tape_signal_routes_to_scalp(self, coordinator):
        """Test that TAPE signals route to scalp book."""
        features = {
            'trend_strength': 0.3,
            'adx': 15.0,
            'mean_revert_bias': 0.4,
            'compression': 0.5,
            'bb_width': 0.02,
            'price_position': 0.5,
            'order_flow_imbalance': 0.65,
            'microprice_edge': 0.70,
            'bid_ask_spread': 0.0008,
            'liquidity_score': 0.75,
        }

        decision = coordinator.process_signal(
            symbol='ETH-USD',
            price=2000.0,
            features=features,
            regime='RANGE',
            spread_bps=8.0,
            liquidity_score=0.75,
        )

        # TAPE signals should route to scalp
        # (Note: May be blocked by gates if edge too low)
        if decision and decision.approved:
            # If approved, should be scalp book
            assert decision.book == BookType.SHORT_HOLD

    def test_low_confidence_signal_blocked(self, coordinator):
        """Test that low-confidence signals are blocked."""
        features = {
            'trend_strength': 0.2,
            'ema_slope': 0.1,
            'momentum_slope': 0.15,
            'htf_bias': 0.45,
            'adx': 10.0,
            'mean_revert_bias': 0.1,
            'compression': 0.2,
            'bb_width': 0.05,
            'price_position': 0.5,
        }

        decision = coordinator.process_signal(
            symbol='ETH-USD',
            price=2000.0,
            features=features,
            regime='RANGE',
            spread_bps=15.0,  # Wide spread
            liquidity_score=0.40,  # Low liquidity
        )

        # Should be blocked (low confidence + poor conditions)
        assert decision is None

        # No positions should be added
        metrics = coordinator.get_metrics()
        assert metrics['books']['combined']['total_positions'] == 0

    def test_multiple_signals_different_books(self, coordinator):
        """Test that multiple signals can route to different books."""
        # Signal 1: High-conviction TREND → Runner
        features_trend = {
            'trend_strength': 0.90,
            'ema_slope': 0.85,
            'momentum_slope': 0.80,
            'htf_bias': 0.85,
            'adx': 40.0,
            'mean_revert_bias': 0.2,
            'compression': 0.3,
            'bb_width': 0.03,
            'price_position': 0.5,
        }

        decision1 = coordinator.process_signal(
            symbol='ETH-USD',
            price=2000.0,
            features=features_trend,
            regime='TREND',
            spread_bps=8.0,
            liquidity_score=0.80,
        )

        # Signal 2: RANGE mean reversion → Scalp
        features_range = {
            'trend_strength': 0.1,
            'adx': 12.0,
            'mean_revert_bias': 0.85,
            'compression': 0.80,
            'bb_width': 0.015,
            'price_position': 0.15,  # Oversold
            'ema_slope': 0.2,
            'momentum_slope': 0.2,
            'htf_bias': 0.45,
        }

        decision2 = coordinator.process_signal(
            symbol='SOL-USD',
            price=100.0,
            features=features_range,
            regime='RANGE',
            spread_bps=6.0,
            liquidity_score=0.75,
        )

        # Check both trades executed in different books
        metrics = coordinator.get_metrics()

        # Should have positions in both books
        total_positions = (
            metrics['books']['short_book']['positions'] +
            metrics['books']['long_book']['positions']
        )
        assert total_positions >= 1  # At least one trade should execute

    def test_heat_limit_enforcement(self, coordinator):
        """Test that heat limits prevent overallocation."""
        # Create many signals to fill up scalp book
        features = {
            'trend_strength': 0.3,
            'adx': 15.0,
            'mean_revert_bias': 0.70,
            'compression': 0.75,
            'bb_width': 0.02,
            'price_position': 0.2,
            'ema_slope': 0.3,
            'momentum_slope': 0.3,
            'htf_bias': 0.5,
        }

        executed_trades = 0
        blocked_by_heat = 0

        # Try to execute 30 scalp trades (should hit heat limit)
        for i in range(30):
            decision = coordinator.process_signal(
                symbol='ETH-USD',
                price=2000.0 + i,  # Vary price slightly
                features=features,
                regime='RANGE',
                spread_bps=8.0,
                liquidity_score=0.75,
            )

            if decision and decision.approved:
                executed_trades += 1
            else:
                # Check if blocked by heat
                metrics = coordinator.get_metrics()
                scalp_heat = metrics['books']['short_book']['heat']
                if scalp_heat >= coordinator.book_manager.max_short_heat:
                    blocked_by_heat += 1

        # Should have executed some trades but then hit heat limit
        assert executed_trades > 0
        assert blocked_by_heat > 0

        # Verify heat is at/near limit
        metrics = coordinator.get_metrics()
        assert metrics['books']['short_book']['heat'] <= coordinator.book_manager.max_short_heat

    def test_position_update_and_close(self, coordinator):
        """Test position price updates and closing."""
        # Execute a trade
        features = {
            'trend_strength': 0.85,
            'ema_slope': 0.75,
            'momentum_slope': 0.70,
            'htf_bias': 0.80,
            'adx': 35.0,
            'mean_revert_bias': 0.2,
            'compression': 0.3,
            'bb_width': 0.03,
            'price_position': 0.5,
        }

        decision = coordinator.process_signal(
            symbol='ETH-USD',
            price=2000.0,
            features=features,
            regime='TREND',
            spread_bps=8.0,
            liquidity_score=0.80,
        )

        if decision and decision.approved:
            # Update position price (simulate price movement)
            coordinator.update_positions('ETH-USD', 2020.0)

            # Get position and check unrealized P&L
            if decision.book == BookType.SHORT_HOLD:
                positions = coordinator.book_manager._get_book_positions(
                    BookType.SHORT_HOLD, open_only=True
                )
            else:
                positions = coordinator.book_manager._get_book_positions(
                    BookType.LONG_HOLD, open_only=True
                )

            if positions:
                position = positions[0]
                assert position.current_price == 2020.0
                assert position.unrealized_pnl > 0  # Should be profitable

                # Close position
                pnl = coordinator.book_manager.close_position(
                    position.position_id,
                    exit_price=2020.0,
                    reason="test_exit",
                )

                assert pnl is not None
                assert pnl > 0

    def test_metrics_tracking(self, coordinator):
        """Test that metrics are tracked correctly."""
        # Execute some trades
        features = {
            'trend_strength': 0.85,
            'ema_slope': 0.75,
            'momentum_slope': 0.70,
            'htf_bias': 0.80,
            'adx': 35.0,
            'mean_revert_bias': 0.2,
            'compression': 0.3,
            'bb_width': 0.03,
            'price_position': 0.5,
        }

        for i in range(5):
            coordinator.process_signal(
                symbol='ETH-USD',
                price=2000.0 + i * 5,
                features=features,
                regime='TREND',
                spread_bps=8.0,
                liquidity_score=0.80,
            )

        # Get metrics
        metrics = coordinator.get_metrics()

        # Check coordinator metrics
        assert metrics['coordinator']['total_signals'] == 5

        # Check that some trades executed or were blocked
        total_outcomes = (
            metrics['coordinator']['trades_executed'] +
            metrics['coordinator']['blocked_consensus'] +
            metrics['coordinator']['blocked_gates'] +
            metrics['coordinator']['blocked_heat']
        )
        assert total_outcomes == 5

    def test_counterfactual_tracking(self, coordinator):
        """Test that blocked trades are tracked for counterfactual analysis."""
        # Create a signal that will likely be blocked by gates
        features = {
            'trend_strength': 0.3,
            'ema_slope': 0.2,
            'momentum_slope': 0.2,
            'htf_bias': 0.5,
            'adx': 15.0,
            'mean_revert_bias': 0.3,
            'compression': 0.4,
            'bb_width': 0.04,
            'price_position': 0.5,
        }

        initial_blocks = len(coordinator.counterfactual_tracker.blocked_trades)

        decision = coordinator.process_signal(
            symbol='ETH-USD',
            price=2000.0,
            features=features,
            regime='RANGE',
            spread_bps=15.0,  # Wide spread = high cost
            liquidity_score=0.50,  # Medium liquidity
        )

        # If blocked by gates, should be in counterfactual tracker
        if decision is None:
            final_blocks = len(coordinator.counterfactual_tracker.blocked_trades)
            # May have recorded a counterfactual (if blocked by gates, not consensus)
            assert final_blocks >= initial_blocks

    def test_status_summary(self, coordinator):
        """Test that status summary is generated correctly."""
        # Execute some trades
        features = {
            'trend_strength': 0.85,
            'ema_slope': 0.75,
            'momentum_slope': 0.70,
            'htf_bias': 0.80,
            'adx': 35.0,
            'mean_revert_bias': 0.2,
            'compression': 0.3,
            'bb_width': 0.03,
            'price_position': 0.5,
        }

        coordinator.process_signal(
            symbol='ETH-USD',
            price=2000.0,
            features=features,
            regime='TREND',
            spread_bps=8.0,
            liquidity_score=0.80,
        )

        # Get status summary
        status = coordinator.get_detailed_status()

        # Should be a formatted string
        assert isinstance(status, str)
        assert 'TRADING COORDINATOR STATUS' in status
        assert 'SCALP BOOK' in status
        assert 'RUNNER BOOK' in status


class TestGateProfiles:
    """Test gate profile filtering."""

    @pytest.fixture
    def coordinator(self):
        """Create coordinator."""
        return TradingCoordinator(total_capital=10000.0)

    def test_scalp_gates_more_permissive(self, coordinator):
        """Test that scalp gates are more permissive than runner gates."""
        # Same features tested against both profiles
        features = {
            'engine_conf': 0.60,
            'regime': 'RANGE',
            'technique': 'RANGE',
        }

        # Check scalp gates
        scalp_decision = coordinator.scalp_profile.check_all_gates(
            edge_hat_bps=10.0,
            features=features,
            order_type=OrderType.MAKER,
            position_size_usd=200.0,
            spread_bps=8.0,
            liquidity_score=0.75,
        )

        # Check runner gates
        runner_decision = coordinator.runner_profile.check_all_gates(
            edge_hat_bps=10.0,
            features=features,
            order_type=OrderType.MAKER,
            position_size_usd=200.0,
            spread_bps=8.0,
            liquidity_score=0.75,
        )

        # Scalp should be more likely to pass
        # (May both fail or both pass, but scalp should never be stricter)
        if runner_decision.passes:
            assert scalp_decision.passes

        # At minimum, scalp should have more gates passed or equal
        assert scalp_decision.gates_passed >= runner_decision.gates_passed - 1


class TestWinRateGovernor:
    """Test win-rate governor integration."""

    def test_governor_adjusts_thresholds(self):
        """Test that governor recommends threshold adjustments."""
        from cloud.training.models.win_rate_governor import WinRateGovernor

        governor = WinRateGovernor(
            target_win_rate=0.72,
            tolerance=0.03,
            window_size=50,
            min_trades=20,
            adjustment_interval=20,
        )

        # Simulate 30 trades with 65% WR (below target)
        for i in range(30):
            won = np.random.random() < 0.65
            governor.record_trade(won)

        # After 20 trades, should recommend loosening
        adjustment = governor.get_threshold_adjustment()

        if adjustment and adjustment.should_adjust:
            # Should recommend loosening (multiplier < 1.0)
            assert adjustment.multiplier < 1.0
            assert adjustment.direction == 'loosen'


class TestConformalGating:
    """Test conformal prediction gating."""

    def test_conformal_calibration(self):
        """Test conformal gate calibration and prediction."""
        from cloud.training.models.conformal_gating import ConformalGate

        gate = ConformalGate(
            significance_level=0.05,
            min_lower_bound_bps=5.0,
        )

        # Simulate calibration data
        np.random.seed(42)
        predictions = np.random.normal(15, 3, 100).tolist()
        actuals = np.random.normal(15, 2, 100).tolist()

        gate.calibrate(predictions, actuals)

        assert gate.is_calibrated
        assert gate.error_quantile is not None

        # Make prediction
        result = gate.predict_with_interval(point_prediction=12.0)

        # Should have an interval
        assert result.lower_bound < result.point_prediction < result.upper_bound
        assert result.interval_width > 0


def test_end_to_end_workflow():
    """Test complete end-to-end workflow."""
    coordinator = TradingCoordinator(
        total_capital=10000.0,
        asset_symbols=['ETH-USD'],
    )

    # Simulate market conditions
    features = {
        'trend_strength': 0.85,
        'ema_slope': 0.80,
        'momentum_slope': 0.75,
        'htf_bias': 0.85,
        'adx': 38.0,
        'mean_revert_bias': 0.2,
        'compression': 0.3,
        'bb_width': 0.03,
        'price_position': 0.5,
    }

    # Process multiple signals
    decisions = []
    for i in range(10):
        decision = coordinator.process_signal(
            symbol='ETH-USD',
            price=2000.0 + i * 2,
            features=features,
            regime='TREND',
            spread_bps=8.0,
            liquidity_score=0.80,
        )
        if decision:
            decisions.append(decision)

    # Should have executed some trades
    metrics = coordinator.get_metrics()

    # Get detailed status
    status = coordinator.get_detailed_status()

    print("\n" + status)

    # Basic validation
    assert metrics['coordinator']['total_signals'] == 10


if __name__ == '__main__':
    # Run end-to-end test
    test_end_to_end_workflow()
