"""
Enhanced Reward Calculator

Multi-objective reward function for RL training.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl
import structlog

from .config import RewardConfig, BALANCED_CONFIG

logger = structlog.get_logger(__name__)


@dataclass
class RewardComponents:
    """
    Breakdown of reward components

    Useful for debugging and understanding reward attribution.
    """
    pnl_component: float
    risk_component: float
    cost_component: float
    sharpe_component: float
    regime_component: float

    total_reward: float

    # Metrics used
    pnl_bps: float
    max_drawdown_pct: float
    total_costs_bps: float
    sharpe_ratio: Optional[float] = None
    regime: Optional[str] = None


class EnhancedRewardCalculator:
    """
    Enhanced RL Reward Calculator

    Calculates multi-objective rewards balancing:
    - PnL (profit)
    - Risk (drawdown, volatility)
    - Costs (fees, slippage)
    - Sharpe ratio
    - Regime consistency

    Example:
        calculator = EnhancedRewardCalculator()

        # Calculate reward for a trade/episode
        reward = calculator.calculate_reward(
            pnl_bps=25.0,
            max_drawdown_pct=5.0,
            total_fees_bps=3.0,
            slippage_bps=2.0,
            regime="trending"
        )

        # Get breakdown
        components = calculator.get_components()
        print(f"PnL: {components.pnl_component:.2f}")
        print(f"Risk: {components.risk_component:.2f}")
        print(f"Cost: {components.cost_component:.2f}")
    """

    def __init__(
        self,
        config: Optional[RewardConfig] = None
    ):
        """
        Initialize reward calculator

        Args:
            config: Reward configuration (None = balanced default)
        """
        self.config = config or BALANCED_CONFIG

        # Track last components for debugging
        self._last_components: Optional[RewardComponents] = None

        # Live feedback statistics (updated from execution feedback)
        self._avg_slippage_bps: float = self.config.expected_slippage_bps
        self._avg_fees_bps: float = self.config.expected_fees_bps

    def calculate_reward(
        self,
        pnl_bps: float,
        max_drawdown_pct: float = 0.0,
        total_fees_bps: float = 0.0,
        slippage_bps: float = 0.0,
        sharpe_ratio: Optional[float] = None,
        regime: Optional[str] = None,
        expected_regime: Optional[str] = None
    ) -> float:
        """
        Calculate reward for a trade or episode

        Args:
            pnl_bps: Profit/loss in basis points
            max_drawdown_pct: Maximum drawdown percentage [0-100]
            total_fees_bps: Total fees in basis points
            slippage_bps: Slippage in basis points
            sharpe_ratio: Sharpe ratio (if available)
            regime: Current market regime
            expected_regime: Expected regime for this strategy

        Returns:
            Total reward (float)
        """
        # 1. PnL Component
        pnl_component = pnl_bps * self.config.pnl_weight

        # 2. Risk Component (penalty for drawdown)
        risk_component = self._calculate_risk_component(max_drawdown_pct)

        # 3. Cost Component (penalty for fees/slippage)
        total_costs_bps = total_fees_bps + slippage_bps
        cost_component = self._calculate_cost_component(
            total_costs_bps,
            slippage_bps
        )

        # 4. Sharpe Component (bonus for risk-adjusted returns)
        sharpe_component = self._calculate_sharpe_component(sharpe_ratio)

        # 5. Regime Component (bonus/penalty for regime match)
        regime_component = self._calculate_regime_component(
            regime,
            expected_regime
        )

        # Total reward
        total_reward = (
            pnl_component +
            risk_component +
            cost_component +
            sharpe_component +
            regime_component
        )

        # Store components for debugging
        self._last_components = RewardComponents(
            pnl_component=pnl_component,
            risk_component=risk_component,
            cost_component=cost_component,
            sharpe_component=sharpe_component,
            regime_component=regime_component,
            total_reward=total_reward,
            pnl_bps=pnl_bps,
            max_drawdown_pct=max_drawdown_pct,
            total_costs_bps=total_costs_bps,
            sharpe_ratio=sharpe_ratio,
            regime=regime
        )

        return total_reward

    def _calculate_risk_component(self, max_drawdown_pct: float) -> float:
        """
        Calculate risk penalty from drawdown

        Args:
            max_drawdown_pct: Maximum drawdown percentage

        Returns:
            Risk component (negative)
        """
        # Cliff penalty for exceeding max acceptable drawdown
        if max_drawdown_pct > self.config.max_acceptable_drawdown_pct:
            return self.config.drawdown_cliff_penalty

        # Linear penalty proportional to drawdown
        risk_penalty = -max_drawdown_pct * self.config.risk_penalty

        return risk_penalty

    def _calculate_cost_component(
        self,
        total_costs_bps: float,
        slippage_bps: float
    ) -> float:
        """
        Calculate cost penalty from fees and slippage

        Args:
            total_costs_bps: Total costs (fees + slippage)
            slippage_bps: Slippage component

        Returns:
            Cost component (negative)
        """
        # Base cost penalty
        cost_penalty = -total_costs_bps * self.config.cost_penalty

        # Extra penalty if slippage exceeds expected
        if self.config.use_live_feedback:
            slippage_excess = max(0, slippage_bps - self._avg_slippage_bps)
            cost_penalty -= slippage_excess * self.config.cost_penalty * 0.5

        return cost_penalty

    def _calculate_sharpe_component(
        self,
        sharpe_ratio: Optional[float]
    ) -> float:
        """
        Calculate Sharpe bonus

        Args:
            sharpe_ratio: Sharpe ratio (None if not available)

        Returns:
            Sharpe component (positive if good Sharpe)
        """
        if sharpe_ratio is None:
            return 0.0

        # Bonus for exceeding target Sharpe
        sharpe_excess = max(0, sharpe_ratio - self.config.target_sharpe_ratio)

        sharpe_bonus = (
            sharpe_excess *
            self.config.sharpe_bonus *
            self.config.sharpe_scaling_factor
        )

        return sharpe_bonus

    def _calculate_regime_component(
        self,
        regime: Optional[str],
        expected_regime: Optional[str]
    ) -> float:
        """
        Calculate regime consistency bonus/penalty

        Args:
            regime: Current market regime
            expected_regime: Expected regime for this strategy

        Returns:
            Regime component (positive if match, negative if mismatch)
        """
        if regime is None or expected_regime is None:
            return 0.0

        # Bonus for trading in expected regime
        if regime == expected_regime:
            return self.config.regime_consistency_bonus

        # Penalty for trading in wrong regime
        regime_penalty_mult = self.config.get_regime_penalty(regime)
        regime_penalty = -self.config.regime_mismatch_penalty * regime_penalty_mult

        return regime_penalty

    def get_components(self) -> Optional[RewardComponents]:
        """
        Get breakdown of last reward calculation

        Returns:
            RewardComponents (None if no calculation yet)
        """
        return self._last_components

    def update_from_feedback(
        self,
        feedback_df: pl.DataFrame
    ) -> None:
        """
        Update reward parameters from live execution feedback

        Args:
            feedback_df: Execution feedback DataFrame (from Hamilton)

        Example:
            # Load feedback
            feedback_df = pl.read_parquet("execution_results.parquet")

            # Update reward calculator
            calculator.update_from_feedback(feedback_df)
        """
        if len(feedback_df) == 0:
            logger.warning("empty_feedback_for_reward_update")
            return

        # Calculate average slippage and fees from live data
        new_avg_slippage = feedback_df['actual_slippage_bps'].mean()
        new_avg_fees = feedback_df['total_fees_bps'].mean()

        # Exponential moving average update
        lr = self.config.feedback_learning_rate

        self._avg_slippage_bps = (
            (1 - lr) * self._avg_slippage_bps +
            lr * new_avg_slippage
        )

        self._avg_fees_bps = (
            (1 - lr) * self._avg_fees_bps +
            lr * new_avg_fees
        )

        logger.info(
            "reward_updated_from_feedback",
            num_trades=len(feedback_df),
            avg_slippage_bps=self._avg_slippage_bps,
            avg_fees_bps=self._avg_fees_bps
        )

    def get_stats(self) -> dict:
        """
        Get current reward calculator statistics

        Returns:
            Dict with current parameters and statistics
        """
        return {
            "config": self.config.to_dict(),
            "avg_slippage_bps": self._avg_slippage_bps,
            "avg_fees_bps": self._avg_fees_bps,
            "total_avg_costs_bps": self._avg_slippage_bps + self._avg_fees_bps
        }


class TrajectoryRewardCalculator:
    """
    Trajectory-Level Reward Calculator

    Calculates rewards for entire trading trajectories (episodes).

    Useful for episodic RL where reward is only given at end of episode.

    Example:
        trajectory_calc = TrajectoryRewardCalculator()

        # Simulate episode
        episode_data = {
            "trades": [...],
            "returns": np.array([...]),
            "drawdowns": np.array([...])
        }

        total_reward = trajectory_calc.calculate_trajectory_reward(
            returns=episode_data['returns'],
            trades=episode_data['trades'],
            regime="trending"
        )
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        """
        Initialize trajectory reward calculator

        Args:
            config: Reward configuration
        """
        self.config = config or BALANCED_CONFIG
        self.step_calculator = EnhancedRewardCalculator(config)

    def calculate_trajectory_reward(
        self,
        returns: np.ndarray,
        trades: int,
        regime: Optional[str] = None
    ) -> float:
        """
        Calculate reward for entire trajectory

        Args:
            returns: Array of returns for each step
            trades: Number of trades executed
            regime: Primary regime during episode

        Returns:
            Total trajectory reward
        """
        if len(returns) == 0:
            return 0.0

        # Calculate trajectory metrics
        total_pnl_bps = returns.sum() * 10000  # Convert to bps

        # Calculate Sharpe
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Calculate max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown_pct = abs(drawdown.min()) * 100

        # Estimate costs
        avg_cost_per_trade_bps = (
            self.step_calculator._avg_slippage_bps +
            self.step_calculator._avg_fees_bps
        )
        total_costs_bps = trades * avg_cost_per_trade_bps

        # Calculate reward
        reward = self.step_calculator.calculate_reward(
            pnl_bps=total_pnl_bps,
            max_drawdown_pct=max_drawdown_pct,
            total_fees_bps=total_costs_bps,
            slippage_bps=0,  # Already included in total_costs
            sharpe_ratio=sharpe,
            regime=regime
        )

        return reward
