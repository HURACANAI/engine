"""
Win-Rate Governor - Feedback Controller for Target Win Rates

Key Problem:
We want specific win rates for different modes:
- Scalp mode: Target 70-75% WR (accept more volume)
- Runner mode: Target 95%+ WR (ultra-selective)

But gates have fixed thresholds that may drift over time.

Solution: Feedback Controller
- Monitors actual win rate vs target
- Adjusts gate thresholds dynamically
- Uses PID control for smooth adjustments
- Prevents overcorrection

How It Works:
    Actual WR: 68% < Target: 72%
    → Error: -4%
    → Action: LOOSEN gates (lower thresholds)
    → More trades pass → WR converges to target

    Actual WR: 78% > Target: 72%
    → Error: +6%
    → Action: TIGHTEN gates (raise thresholds)
    → Fewer trades pass → WR converges to target

PID Controller:
- P (Proportional): Respond to current error
- I (Integral): Correct persistent bias
- D (Derivative): Dampen oscillations

Example:
    governor = WinRateGovernor(
        target_win_rate=0.72,
        tolerance=0.03,  # ±3%
    )

    # After each trade
    governor.record_trade(won=True)

    # Periodically check
    adjustment = governor.get_threshold_adjustment()
    if adjustment:
        cost_gate.buffer_bps *= adjustment.multiplier
        meta_gate.threshold *= adjustment.multiplier

Benefits:
- Automatic threshold tuning
- Maintains target WR over time
- Adapts to changing market conditions
- Prevents manual tweaking

Usage:
    # Initialize for scalp mode
    scalp_governor = WinRateGovernor(
        target_win_rate=0.72,
        tolerance=0.03,
        window_size=50,
    )

    # After each scalp trade
    scalp_governor.record_trade(won=trade_pnl > 0)

    # Check if adjustment needed
    adj = scalp_governor.get_threshold_adjustment()
    if adj and adj.should_adjust:
        # Apply to all gates in scalp profile
        scalp_profile.cost_gate.buffer_bps *= adj.multiplier
        scalp_profile.meta_label.threshold *= adj.multiplier
        print(f"Adjusted gates by {adj.multiplier:.3f}: {adj.reason}")
"""

from dataclasses import dataclass
from collections import deque
from typing import Deque, Optional
import time

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ThresholdAdjustment:
    """Threshold adjustment recommendation."""

    should_adjust: bool
    multiplier: float  # Apply to thresholds (e.g., 0.95 = loosen 5%)
    direction: str  # 'loosen' or 'tighten'
    reason: str
    current_win_rate: float
    target_win_rate: float
    error: float


class WinRateGovernor:
    """
    PID-based feedback controller for maintaining target win rate.

    Architecture:
        Actual WR → Error Calculation → PID Controller → Threshold Adjustment
                        ↑___________________|

    PID Formula:
        adjustment = Kp * error + Ki * integral + Kd * derivative

    where:
        error = actual_wr - target_wr
        integral = sum of past errors
        derivative = change in error

    Positive error (WR too high) → Tighten gates
    Negative error (WR too low) → Loosen gates
    """

    def __init__(
        self,
        target_win_rate: float,
        tolerance: float = 0.03,  # ±3%
        window_size: int = 50,  # Recent trades to consider
        min_trades: int = 20,  # Min trades before adjusting
        kp: float = 0.5,  # Proportional gain
        ki: float = 0.1,  # Integral gain
        kd: float = 0.2,  # Derivative gain
        max_adjustment: float = 0.10,  # Max 10% adjustment per step
        adjustment_interval: int = 20,  # Adjust every N trades
    ):
        """
        Initialize win-rate governor.

        Args:
            target_win_rate: Target win rate (e.g., 0.72 for 72%)
            tolerance: Acceptable deviation (e.g., 0.03 = ±3%)
            window_size: Number of recent trades to track
            min_trades: Minimum trades before making adjustments
            kp: Proportional gain (responsiveness to current error)
            ki: Integral gain (correction of persistent bias)
            kd: Derivative gain (dampening of oscillations)
            max_adjustment: Maximum adjustment per step (prevents overcorrection)
            adjustment_interval: Adjust every N trades
        """
        self.target_wr = target_win_rate
        self.tolerance = tolerance
        self.window_size = window_size
        self.min_trades = min_trades
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_adjustment = max_adjustment
        self.adjustment_interval = adjustment_interval

        # Trade history (1 = win, 0 = loss)
        self.trade_history: Deque[bool] = deque(maxlen=window_size)

        # PID state
        self.error_integral: float = 0.0
        self.previous_error: Optional[float] = None

        # Statistics
        self.total_trades = 0
        self.total_wins = 0
        self.adjustments_made = 0
        self.last_adjustment_trade = 0

        logger.info(
            "win_rate_governor_initialized",
            target_wr=target_win_rate,
            tolerance=tolerance,
            kp=kp,
            ki=ki,
            kd=kd,
        )

    def record_trade(self, won: bool) -> None:
        """
        Record trade outcome.

        Args:
            won: True if trade won, False if lost
        """
        self.trade_history.append(won)
        self.total_trades += 1

        if won:
            self.total_wins += 1

        logger.debug(
            "trade_recorded",
            won=won,
            total_trades=self.total_trades,
            window_wr=self.get_current_win_rate(),
        )

    def get_current_win_rate(self) -> float:
        """Get current win rate from recent trades."""
        if not self.trade_history:
            return 0.0

        return sum(self.trade_history) / len(self.trade_history)

    def _calculate_pid_output(self, error: float) -> float:
        """
        Calculate PID controller output.

        Args:
            error: Current error (actual_wr - target_wr)

        Returns:
            PID output (adjustment factor)
        """
        # Proportional term
        p_term = self.kp * error

        # Integral term (accumulated error)
        self.error_integral += error
        i_term = self.ki * self.error_integral

        # Derivative term (rate of change)
        if self.previous_error is not None:
            d_term = self.kd * (error - self.previous_error)
        else:
            d_term = 0.0

        self.previous_error = error

        # Total PID output
        pid_output = p_term + i_term + d_term

        return pid_output

    def get_threshold_adjustment(
        self,
        force: bool = False,
    ) -> Optional[ThresholdAdjustment]:
        """
        Get threshold adjustment recommendation.

        Args:
            force: Force adjustment even if interval not reached

        Returns:
            ThresholdAdjustment or None if no adjustment needed
        """
        # Check if we have enough data
        if len(self.trade_history) < self.min_trades:
            return ThresholdAdjustment(
                should_adjust=False,
                multiplier=1.0,
                direction='none',
                reason=f"Only {len(self.trade_history)} trades (need {self.min_trades})",
                current_win_rate=self.get_current_win_rate(),
                target_win_rate=self.target_wr,
                error=0.0,
            )

        # Check if enough trades since last adjustment
        trades_since_last = self.total_trades - self.last_adjustment_trade
        if not force and trades_since_last < self.adjustment_interval:
            return None

        # Calculate current win rate
        current_wr = self.get_current_win_rate()

        # Calculate error (positive = WR too high, negative = WR too low)
        error = current_wr - self.target_wr

        # Check if within tolerance
        if abs(error) <= self.tolerance:
            return ThresholdAdjustment(
                should_adjust=False,
                multiplier=1.0,
                direction='none',
                reason=f"WR {current_wr:.1%} within tolerance of {self.target_wr:.1%}",
                current_win_rate=current_wr,
                target_win_rate=self.target_wr,
                error=error,
            )

        # Calculate PID output
        pid_output = self._calculate_pid_output(error)

        # Convert PID output to threshold multiplier
        # Positive error (WR too high) → Tighten → Increase thresholds → multiplier > 1.0
        # Negative error (WR too low) → Loosen → Decrease thresholds → multiplier < 1.0
        raw_multiplier = 1.0 + pid_output

        # Clamp to max adjustment
        min_multiplier = 1.0 - self.max_adjustment
        max_multiplier = 1.0 + self.max_adjustment
        multiplier = np.clip(raw_multiplier, min_multiplier, max_multiplier)

        # Determine direction
        if multiplier > 1.0:
            direction = 'tighten'
            reason = f"WR too high ({current_wr:.1%} > {self.target_wr:.1%}), tighten by {(multiplier-1)*100:.1f}%"
        else:
            direction = 'loosen'
            reason = f"WR too low ({current_wr:.1%} < {self.target_wr:.1%}), loosen by {(1-multiplier)*100:.1f}%"

        # Record adjustment
        self.adjustments_made += 1
        self.last_adjustment_trade = self.total_trades

        logger.info(
            "threshold_adjustment_recommended",
            current_wr=current_wr,
            target_wr=self.target_wr,
            error=error,
            direction=direction,
            multiplier=multiplier,
            pid_output=pid_output,
        )

        return ThresholdAdjustment(
            should_adjust=True,
            multiplier=multiplier,
            direction=direction,
            reason=reason,
            current_win_rate=current_wr,
            target_win_rate=self.target_wr,
            error=error,
        )

    def reset_pid_state(self) -> None:
        """Reset PID controller state (e.g., after regime change)."""
        self.error_integral = 0.0
        self.previous_error = None

        logger.info("win_rate_governor_pid_reset")

    def get_statistics(self) -> dict:
        """Get governor statistics."""
        current_wr = self.get_current_win_rate()
        lifetime_wr = self.total_wins / self.total_trades if self.total_trades > 0 else 0.0

        return {
            'target_win_rate': self.target_wr,
            'current_win_rate': current_wr,
            'lifetime_win_rate': lifetime_wr,
            'error': current_wr - self.target_wr,
            'within_tolerance': abs(current_wr - self.target_wr) <= self.tolerance,
            'total_trades': self.total_trades,
            'total_wins': self.total_wins,
            'window_size': len(self.trade_history),
            'adjustments_made': self.adjustments_made,
            'pid_state': {
                'error_integral': self.error_integral,
                'previous_error': self.previous_error,
            },
        }


class DualModeGovernor:
    """
    Manages separate governors for scalp and runner modes.

    Each mode has different target:
    - Scalp: 70-75% WR (accept more volume)
    - Runner: 95%+ WR (ultra-selective)
    """

    def __init__(
        self,
        scalp_target_wr: float = 0.72,
        runner_target_wr: float = 0.95,
        scalp_tolerance: float = 0.03,
        runner_tolerance: float = 0.02,
    ):
        """
        Initialize dual-mode governor.

        Args:
            scalp_target_wr: Target WR for scalp mode
            runner_target_wr: Target WR for runner mode
            scalp_tolerance: Tolerance for scalp mode
            runner_tolerance: Tolerance for runner mode
        """
        self.scalp_governor = WinRateGovernor(
            target_win_rate=scalp_target_wr,
            tolerance=scalp_tolerance,
            window_size=50,
            min_trades=20,
            adjustment_interval=20,
        )

        self.runner_governor = WinRateGovernor(
            target_win_rate=runner_target_wr,
            tolerance=runner_tolerance,
            window_size=30,  # Fewer trades, smaller window
            min_trades=10,
            adjustment_interval=10,
        )

        logger.info(
            "dual_mode_governor_initialized",
            scalp_target=scalp_target_wr,
            runner_target=runner_target_wr,
        )

    def record_scalp_trade(self, won: bool) -> None:
        """Record scalp trade outcome."""
        self.scalp_governor.record_trade(won)

    def record_runner_trade(self, won: bool) -> None:
        """Record runner trade outcome."""
        self.runner_governor.record_trade(won)

    def get_scalp_adjustment(self) -> Optional[ThresholdAdjustment]:
        """Get scalp mode threshold adjustment."""
        return self.scalp_governor.get_threshold_adjustment()

    def get_runner_adjustment(self) -> Optional[ThresholdAdjustment]:
        """Get runner mode threshold adjustment."""
        return self.runner_governor.get_threshold_adjustment()

    def get_summary(self) -> dict:
        """Get summary of both governors."""
        return {
            'scalp': self.scalp_governor.get_statistics(),
            'runner': self.runner_governor.get_statistics(),
        }


def run_governor_example():
    """Example usage of win-rate governor."""
    # Simulate trading with drift
    np.random.seed(42)

    governor = WinRateGovernor(
        target_win_rate=0.72,
        tolerance=0.03,
        window_size=50,
        min_trades=20,
        adjustment_interval=20,
    )

    # Simulate 200 trades with changing WR
    print("=" * 70)
    print("WIN-RATE GOVERNOR SIMULATION")
    print("=" * 70)

    for i in range(200):
        # Simulate WR drift (starts low, increases over time)
        true_wr = 0.65 + (i / 200) * 0.15  # 65% → 80%
        won = np.random.random() < true_wr

        governor.record_trade(won)

        # Check for adjustments every 20 trades
        if (i + 1) % 20 == 0:
            adj = governor.get_threshold_adjustment()

            if adj and adj.should_adjust:
                print(f"\nTrade {i+1}: {adj.reason}")
                print(f"  Multiplier: {adj.multiplier:.3f} ({adj.direction})")
                print(f"  Current WR: {adj.current_win_rate:.1%}")
                print(f"  Target WR: {adj.target_win_rate:.1%}")
                print(f"  Error: {adj.error:+.1%}")

    # Final statistics
    print(f"\n{'=' * 70}")
    print("FINAL STATISTICS")
    print("=" * 70)
    stats = governor.get_statistics()
    print(f"Target WR: {stats['target_win_rate']:.1%}")
    print(f"Current WR: {stats['current_win_rate']:.1%}")
    print(f"Lifetime WR: {stats['lifetime_win_rate']:.1%}")
    print(f"Within Tolerance: {stats['within_tolerance']}")
    print(f"Total Trades: {stats['total_trades']}")
    print(f"Adjustments Made: {stats['adjustments_made']}")


if __name__ == "__main__":
    run_governor_example()
