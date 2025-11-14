"""
Alpha Bandit

Multi-armed bandit for exploring alpha strategies.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import structlog

from .algorithms import (
    BanditAlgorithm,
    ThompsonSampling,
    UCB,
    EpsilonGreedy,
    ContextualBandit
)

logger = structlog.get_logger(__name__)


@dataclass
class BanditArm:
    """
    Single bandit arm (strategy)

    Tracks performance of one alpha strategy.
    """
    name: str
    num_pulls: int = 0
    total_reward: float = 0.0
    total_squared_reward: float = 0.0

    def mean_reward(self) -> float:
        """Calculate mean reward"""
        if self.num_pulls == 0:
            return 0.0
        return self.total_reward / self.num_pulls

    def reward_variance(self) -> float:
        """Calculate reward variance"""
        if self.num_pulls == 0:
            return 0.0

        mean = self.mean_reward()
        mean_of_squares = self.total_squared_reward / self.num_pulls

        return max(0, mean_of_squares - mean ** 2)

    def reward_std(self) -> float:
        """Calculate reward standard deviation"""
        return np.sqrt(self.reward_variance())


@dataclass
class BanditStats:
    """Bandit statistics"""
    total_pulls: int
    best_arm: str
    best_arm_mean_reward: float

    arm_stats: Dict[str, dict]


class AlphaBandit:
    """
    Alpha Bandit

    Multi-armed bandit for exploring different alpha strategies.

    Automatically balances exploration (trying new strategies)
    with exploitation (using best known strategy).

    Example:
        bandit = AlphaBandit(
            strategies=["momentum", "mean_reversion", "breakout"],
            algorithm="thompson_sampling"
        )

        for _ in range(1000):
            # Select strategy
            strategy = bandit.select_strategy()

            # Execute and get reward (e.g., Sharpe ratio)
            reward = execute_strategy(strategy)

            # Update bandit
            bandit.update(strategy, reward)

        # Check results
        stats = bandit.get_statistics()
        print(f"Best: {stats['best_arm']} ({stats['best_arm_mean_reward']:.2f})")
    """

    def __init__(
        self,
        strategies: List[str],
        algorithm: str = "thompson_sampling",
        context_aware: bool = False,
        context_dim: int = 10
    ):
        """
        Initialize Alpha Bandit

        Args:
            strategies: List of strategy names (arms)
            algorithm: Algorithm to use (thompson_sampling, ucb, epsilon_greedy)
            context_aware: Whether to use contextual bandit
            context_dim: Context dimension (if context_aware)
        """
        self.strategies = strategies
        self.num_arms = len(strategies)

        # Initialize arms
        self.arms: Dict[str, BanditArm] = {
            strategy: BanditArm(name=strategy)
            for strategy in strategies
        }

        # Select algorithm
        self.context_aware = context_aware

        if context_aware:
            self.algorithm = ContextualBandit(
                num_arms=self.num_arms,
                context_dim=context_dim
            )
        else:
            if algorithm == "thompson_sampling":
                self.algorithm = ThompsonSampling()
            elif algorithm == "ucb":
                self.algorithm = UCB()
            elif algorithm == "epsilon_greedy":
                self.algorithm = EpsilonGreedy()
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

        self.algorithm_name = algorithm

        # Track history
        self.selection_history: List[str] = []
        self.reward_history: List[float] = []

        logger.info(
            "alpha_bandit_initialized",
            num_strategies=len(strategies),
            strategies=strategies,
            algorithm=algorithm,
            context_aware=context_aware
        )

    def select_strategy(
        self,
        context: Optional[Dict] = None
    ) -> str:
        """
        Select strategy to use

        Args:
            context: Optional context (e.g., {"regime": "trending"})

        Returns:
            Selected strategy name

        Example:
            strategy = bandit.select_strategy(context={"regime": "trending"})
        """
        # Get arm statistics
        num_pulls = np.array([self.arms[s].num_pulls for s in self.strategies])
        total_rewards = np.array([self.arms[s].total_reward for s in self.strategies])

        # Select arm using algorithm
        if self.context_aware and context is not None:
            # Convert context to feature vector
            context_vector = self._context_to_vector(context)
            selected_arm_idx = self.algorithm.select_arm(context=context_vector)
        else:
            # Context-free selection
            selected_arm_idx = self.algorithm.select_arm(
                num_pulls=num_pulls,
                rewards=total_rewards,
                total_pulls=sum(num_pulls)
            )

        selected_strategy = self.strategies[selected_arm_idx]

        self.selection_history.append(selected_strategy)

        logger.debug(
            "strategy_selected",
            strategy=selected_strategy,
            total_pulls=self.arms[selected_strategy].num_pulls
        )

        return selected_strategy

    def update(
        self,
        strategy: str,
        reward: float,
        context: Optional[Dict] = None
    ) -> None:
        """
        Update bandit with observed reward

        Args:
            strategy: Strategy that was executed
            reward: Observed reward (e.g., Sharpe ratio, PnL)
            context: Optional context used

        Example:
            bandit.update("momentum", reward=1.5)
        """
        if strategy not in self.arms:
            raise ValueError(f"Unknown strategy: {strategy}")

        arm = self.arms[strategy]

        # Update arm statistics
        arm.num_pulls += 1
        arm.total_reward += reward
        arm.total_squared_reward += reward ** 2

        self.reward_history.append(reward)

        # Update contextual bandit if applicable
        if self.context_aware and context is not None:
            context_vector = self._context_to_vector(context)
            arm_idx = self.strategies.index(strategy)
            self.algorithm.update(arm_idx, context_vector, reward)

        logger.debug(
            "bandit_updated",
            strategy=strategy,
            reward=reward,
            mean_reward=arm.mean_reward(),
            num_pulls=arm.num_pulls
        )

    def get_statistics(self) -> BanditStats:
        """
        Get bandit statistics

        Returns:
            BanditStats
        """
        # Find best arm
        best_arm = max(self.arms.values(), key=lambda arm: arm.mean_reward())

        # Arm statistics
        arm_stats = {}
        for strategy, arm in self.arms.items():
            arm_stats[strategy] = {
                "num_pulls": arm.num_pulls,
                "mean_reward": arm.mean_reward(),
                "reward_std": arm.reward_std(),
                "total_reward": arm.total_reward
            }

        stats = BanditStats(
            total_pulls=sum(arm.num_pulls for arm in self.arms.values()),
            best_arm=best_arm.name,
            best_arm_mean_reward=best_arm.mean_reward(),
            arm_stats=arm_stats
        )

        return stats

    def get_arm_probabilities(self) -> Dict[str, float]:
        """
        Get current selection probabilities for each arm

        Returns:
            Dict of {strategy: probability}
        """
        # Run Monte Carlo simulation to estimate probabilities
        num_simulations = 1000
        selections = {strategy: 0 for strategy in self.strategies}

        num_pulls = np.array([self.arms[s].num_pulls for s in self.strategies])
        total_rewards = np.array([self.arms[s].total_reward for s in self.strategies])

        for _ in range(num_simulations):
            arm_idx = self.algorithm.select_arm(
                num_pulls=num_pulls,
                rewards=total_rewards,
                total_pulls=sum(num_pulls)
            )
            selected_strategy = self.strategies[arm_idx]
            selections[selected_strategy] += 1

        # Convert to probabilities
        probabilities = {
            strategy: count / num_simulations
            for strategy, count in selections.items()
        }

        return probabilities

    def reset(self) -> None:
        """Reset bandit (clear all statistics)"""
        for arm in self.arms.values():
            arm.num_pulls = 0
            arm.total_reward = 0.0
            arm.total_squared_reward = 0.0

        self.selection_history = []
        self.reward_history = []

        logger.info("bandit_reset")

    def _context_to_vector(self, context: Dict) -> np.ndarray:
        """
        Convert context dict to feature vector

        Args:
            context: Context dict (e.g., {"regime": "trending"})

        Returns:
            Feature vector
        """
        # Simple encoding - in production, use proper feature engineering
        features = []

        # Regime encoding
        regime = context.get("regime", "unknown")
        regime_encoding = {
            "trending": [1, 0, 0],
            "choppy": [0, 1, 0],
            "volatile": [0, 0, 1],
            "unknown": [0, 0, 0]
        }
        features.extend(regime_encoding.get(regime, [0, 0, 0]))

        # Volatility
        volatility = context.get("volatility", 0.5)
        features.append(volatility)

        # Volume
        volume_z = context.get("volume_zscore", 0.0)
        features.append(volume_z)

        # Pad to context_dim
        while len(features) < self.algorithm.context_dim:
            features.append(0.0)

        return np.array(features[:self.algorithm.context_dim])

    def generate_report(self) -> str:
        """
        Generate human-readable report

        Returns:
            Report string
        """
        stats = self.get_statistics()

        lines = [
            "=" * 80,
            "ALPHA BANDIT REPORT",
            "=" * 80,
            "",
            f"Algorithm: {self.algorithm_name}",
            f"Total Pulls: {stats.total_pulls}",
            f"Best Strategy: {stats.best_arm} ({stats.best_arm_mean_reward:.3f})",
            "",
            "STRATEGY PERFORMANCE:",
            "-" * 80,
        ]

        # Sort strategies by mean reward
        sorted_arms = sorted(
            stats.arm_stats.items(),
            key=lambda x: x[1]['mean_reward'],
            reverse=True
        )

        for strategy, arm_stat in sorted_arms:
            lines.append(
                f"  {strategy:20s} | "
                f"Pulls: {arm_stat['num_pulls']:4d} | "
                f"Mean: {arm_stat['mean_reward']:6.3f} | "
                f"Std: {arm_stat['reward_std']:6.3f}"
            )

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)
