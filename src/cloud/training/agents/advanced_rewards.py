"""
Advanced Reward Shaping for RL Agent

Multi-component reward function that teaches the agent to:
1. Maximize profit (base objective)
2. Improve Sharpe ratio (risk-adjusted returns)
3. Minimize drawdowns (capital preservation)
4. Avoid overtrading (reduce transaction costs)
5. Align with market regime (trade with the trend)

This creates a more robust, risk-aware trading agent.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TradeResult:
    """Complete trade result for reward calculation."""

    pnl_bps: float  # Profit/loss in basis points
    entry_price: float
    exit_price: float
    position_size: float
    hold_duration_minutes: int
    entry_regime: str  # "trend", "range", "panic", etc.
    exit_regime: str
    max_unrealized_drawdown_bps: float  # Worst drawdown during trade
    max_unrealized_profit_bps: float  # Best profit during trade


@dataclass
class RewardComponents:
    """Breakdown of reward calculation."""

    profit_component: float
    sharpe_component: float
    drawdown_penalty: float
    frequency_penalty: float
    regime_alignment_bonus: float
    total_reward: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "profit": self.profit_component,
            "sharpe": self.sharpe_component,
            "drawdown": self.drawdown_penalty,
            "frequency": self.frequency_penalty,
            "regime": self.regime_alignment_bonus,
            "total": self.total_reward,
        }


class AdvancedRewardCalculator:
    """
    Multi-component reward calculator for RL agent.

    Combines multiple objectives into a single reward signal:
    - Profit (0.5 weight)
    - Sharpe ratio (0.2 weight)
    - Drawdown penalty (0.15 weight)
    - Trading frequency penalty (0.1 weight)
    - Regime alignment (0.05 weight)
    """

    def __init__(
        self,
        profit_weight: float = 0.5,
        sharpe_weight: float = 0.2,
        drawdown_weight: float = 0.15,
        frequency_weight: float = 0.1,
        regime_weight: float = 0.05,
        returns_window: int = 100,  # Track last 100 trades for Sharpe
        target_trade_frequency_per_day: float = 5.0,  # Ideal trades per day
    ):
        """
        Initialize reward calculator.

        Args:
            profit_weight: Weight for profit component
            sharpe_weight: Weight for Sharpe ratio component
            drawdown_weight: Weight for drawdown penalty
            frequency_weight: Weight for frequency penalty
            regime_weight: Weight for regime alignment
            returns_window: Number of recent returns to track for Sharpe
            target_trade_frequency_per_day: Ideal trading frequency
        """
        # Validate weights sum to 1.0
        total_weight = profit_weight + sharpe_weight + drawdown_weight + frequency_weight + regime_weight
        assert abs(total_weight - 1.0) < 0.01, f"Weights must sum to 1.0, got {total_weight}"

        self.profit_weight = profit_weight
        self.sharpe_weight = sharpe_weight
        self.drawdown_weight = drawdown_weight
        self.frequency_weight = frequency_weight
        self.regime_weight = regime_weight

        # Rolling window of returns for Sharpe calculation
        self.returns_window = returns_window
        self.recent_returns: deque = deque(maxlen=returns_window)

        # Trading frequency tracking
        self.target_freq = target_trade_frequency_per_day
        self.recent_trade_times: deque = deque(maxlen=100)  # Last 100 trade timestamps

        # Max drawdown tracking
        self.peak_equity = 0.0
        self.max_drawdown_bps = 0.0

        logger.info(
            "advanced_reward_calculator_initialized",
            profit_weight=profit_weight,
            sharpe_weight=sharpe_weight,
            drawdown_weight=drawdown_weight,
            frequency_weight=frequency_weight,
            regime_weight=regime_weight,
        )

    def calculate_reward(
        self,
        trade_result: TradeResult,
        current_timestamp: Optional[float] = None,
    ) -> Tuple[float, RewardComponents]:
        """
        Calculate multi-component reward.

        Args:
            trade_result: Complete trade result
            current_timestamp: Current time (for frequency calculation)

        Returns:
            (total_reward, reward_components) tuple
        """
        # 1. Profit Component (0.5 weight)
        profit_comp = self._calculate_profit_component(trade_result.pnl_bps)

        # 2. Sharpe Component (0.2 weight)
        sharpe_comp = self._calculate_sharpe_component(trade_result.pnl_bps)

        # 3. Drawdown Penalty (0.15 weight)
        drawdown_penalty = self._calculate_drawdown_penalty(
            trade_result.max_unrealized_drawdown_bps,
            trade_result.pnl_bps,
        )

        # 4. Frequency Penalty (0.1 weight)
        freq_penalty = self._calculate_frequency_penalty(current_timestamp)

        # 5. Regime Alignment Bonus (0.05 weight)
        regime_bonus = self._calculate_regime_bonus(
            trade_result.entry_regime,
            trade_result.exit_regime,
            trade_result.pnl_bps,
        )

        # Combine weighted components
        total_reward = (
            self.profit_weight * profit_comp +
            self.sharpe_weight * sharpe_comp +
            self.drawdown_weight * drawdown_penalty +
            self.frequency_weight * freq_penalty +
            self.regime_weight * regime_bonus
        )

        components = RewardComponents(
            profit_component=profit_comp,
            sharpe_component=sharpe_comp,
            drawdown_penalty=drawdown_penalty,
            frequency_penalty=freq_penalty,
            regime_alignment_bonus=regime_bonus,
            total_reward=total_reward,
        )

        logger.debug(
            "reward_calculated",
            total=total_reward,
            components=components.to_dict(),
        )

        return total_reward, components

    def _calculate_profit_component(self, pnl_bps: float) -> float:
        """
        Calculate profit component.

        Simple scaling of PnL:
        - Positive PnL → positive reward
        - Negative PnL → negative reward
        - Scaled to reasonable range [-1, 1]

        Args:
            pnl_bps: Profit/loss in basis points

        Returns:
            Profit component in [-1, 1]
        """
        # Scale: 100 bps (1%) → reward of 0.5
        # 200 bps (2%) → reward of 1.0
        # -100 bps → reward of -0.5
        scaled = pnl_bps / 200.0

        # Clip to [-1, 1]
        return np.clip(scaled, -1.0, 1.0)

    def _calculate_sharpe_component(self, pnl_bps: float) -> float:
        """
        Calculate Sharpe ratio component.

        Rewards trades that improve rolling Sharpe ratio.
        Sharpe = mean(returns) / std(returns)

        Args:
            pnl_bps: Current trade PnL

        Returns:
            Sharpe component in [-1, 1]
        """
        # Add to rolling returns
        self.recent_returns.append(pnl_bps)

        if len(self.recent_returns) < 20:
            # Not enough data for reliable Sharpe
            return 0.0

        returns_array = np.array(self.recent_returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)

        if std_return < 1e-6:
            # No volatility
            return 0.0

        sharpe = mean_return / std_return

        # Scale Sharpe to [-1, 1]
        # Sharpe of 2.0 → reward of 1.0
        # Sharpe of -2.0 → reward of -1.0
        scaled = sharpe / 2.0

        return np.clip(scaled, -1.0, 1.0)

    def _calculate_drawdown_penalty(
        self,
        max_unrealized_dd_bps: float,
        final_pnl_bps: float,
    ) -> float:
        """
        Calculate drawdown penalty.

        Penalizes trades with large unrealized drawdowns, even if they eventually win.
        This encourages better entry timing and stop placement.

        Args:
            max_unrealized_dd_bps: Worst unrealized drawdown during trade
            final_pnl_bps: Final PnL

        Returns:
            Penalty in [-1, 0] (always non-positive)
        """
        # If trade made money but had large drawdown → still penalize
        # Encourages clean wins without drama

        if max_unrealized_dd_bps >= 0:
            # No drawdown
            return 0.0

        # Exponential penalty for large drawdowns
        # -50 bps DD → -0.1 penalty
        # -100 bps DD → -0.25 penalty
        # -200 bps DD → -0.5 penalty
        # -500 bps DD → -1.0 penalty

        abs_dd = abs(max_unrealized_dd_bps)
        penalty = -1.0 * (1.0 - np.exp(-abs_dd / 200.0))

        return np.clip(penalty, -1.0, 0.0)

    def _calculate_frequency_penalty(
        self,
        current_timestamp: Optional[float] = None,
    ) -> float:
        """
        Calculate trading frequency penalty.

        Penalizes overtrading (too many trades → high costs).
        Rewards maintaining optimal trade frequency.

        Args:
            current_timestamp: Current time in seconds

        Returns:
            Penalty in [-1, 0.2]
        """
        if current_timestamp is None:
            return 0.0

        self.recent_trade_times.append(current_timestamp)

        if len(self.recent_trade_times) < 10:
            # Not enough data
            return 0.0

        # Calculate trades per day
        time_span_seconds = self.recent_trade_times[-1] - self.recent_trade_times[0]
        time_span_days = time_span_seconds / 86400.0

        if time_span_days < 0.01:
            return 0.0

        trades_per_day = len(self.recent_trade_times) / time_span_days

        # Penalty based on deviation from target
        deviation = abs(trades_per_day - self.target_freq)

        if deviation < 1.0:
            # Within 1 trade/day of target → small bonus
            return 0.2
        elif deviation < 3.0:
            # Within 3 trades/day → neutral
            return 0.0
        else:
            # Too far from target → penalty
            penalty = -0.5 * (deviation / 10.0)
            return np.clip(penalty, -1.0, 0.0)

    def _calculate_regime_bonus(
        self,
        entry_regime: str,
        exit_regime: str,
        pnl_bps: float,
    ) -> float:
        """
        Calculate regime alignment bonus.

        Rewards trading WITH the regime:
        - Long in TREND regime → bonus if win
        - Mean reversion in RANGE regime → bonus if win
        - Avoiding trades in PANIC → bonus

        Args:
            entry_regime: Regime at entry
            exit_regime: Regime at exit
            pnl_bps: Trade PnL

        Returns:
            Bonus in [-0.5, 0.5]
        """
        if entry_regime == "unknown":
            return 0.0

        # Define good regime alignments
        good_alignments = {
            "trend": pnl_bps > 0,  # Long in trend should win
            "range": pnl_bps > 0,  # Mean reversion should win
            "panic": False,  # Should avoid trading in panic
        }

        entry_regime_lower = entry_regime.lower()

        if entry_regime_lower in good_alignments:
            aligned = good_alignments[entry_regime_lower]

            if aligned:
                # Good alignment + win → bonus
                return 0.5 if pnl_bps > 0 else -0.2
            else:
                # Poor alignment (e.g., trading in panic)
                return -0.5

        return 0.0

    def update_equity(self, current_equity_bps: float) -> None:
        """
        Update equity tracking for drawdown calculation.

        Args:
            current_equity_bps: Current equity in basis points
        """
        if current_equity_bps > self.peak_equity:
            self.peak_equity = current_equity_bps

        current_dd = current_equity_bps - self.peak_equity
        if current_dd < self.max_drawdown_bps:
            self.max_drawdown_bps = current_dd

    def get_current_sharpe(self) -> float:
        """
        Get current Sharpe ratio.

        Returns:
            Sharpe ratio or 0.0 if insufficient data
        """
        if len(self.recent_returns) < 20:
            return 0.0

        returns_array = np.array(self.recent_returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)

        if std_return < 1e-6:
            return 0.0

        return mean_return / std_return

    def get_stats(self) -> Dict[str, float]:
        """
        Get current statistics.

        Returns:
            Dictionary of current stats
        """
        return {
            "sharpe_ratio": self.get_current_sharpe(),
            "num_recent_returns": len(self.recent_returns),
            "mean_return_bps": np.mean(self.recent_returns) if self.recent_returns else 0.0,
            "std_return_bps": np.std(self.recent_returns) if self.recent_returns else 0.0,
            "max_drawdown_bps": self.max_drawdown_bps,
            "peak_equity_bps": self.peak_equity,
        }

    def reset(self) -> None:
        """Reset all tracking state."""
        self.recent_returns.clear()
        self.recent_trade_times.clear()
        self.peak_equity = 0.0
        self.max_drawdown_bps = 0.0

        logger.info("advanced_reward_calculator_reset")
