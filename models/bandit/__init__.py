"""
Alpha Bandit for Exploration

Multi-armed bandit for exploring different alpha strategies.

Key Features:
- Thompson Sampling for strategy selection
- UCB (Upper Confidence Bound) algorithm
- Contextual bandits with regime awareness
- Automatic exploration/exploitation balance
- Performance tracking per arm

Usage:
    from models.bandit import AlphaBandit

    # Initialize bandit with multiple strategies
    bandit = AlphaBandit(
        strategies=["momentum", "mean_reversion", "breakout"],
        algorithm="thompson_sampling"
    )

    # Select strategy for current context
    regime = "trending"
    selected_strategy = bandit.select_strategy(context={"regime": regime})

    # Execute strategy and get reward
    reward = execute_strategy(selected_strategy)

    # Update bandit with observed reward
    bandit.update(
        strategy=selected_strategy,
        reward=reward,
        context={"regime": regime}
    )

    # View performance
    stats = bandit.get_statistics()
    print(f"Best strategy: {stats['best_strategy']}")
"""

from .bandit import (
    AlphaBandit,
    BanditAlgorithm,
    BanditArm,
    BanditStats
)
from .algorithms import (
    ThompsonSampling,
    UCB,
    EpsilonGreedy
)

__all__ = [
    # Bandit
    "AlphaBandit",
    "BanditAlgorithm",
    "BanditArm",
    "BanditStats",

    # Algorithms
    "ThompsonSampling",
    "UCB",
    "EpsilonGreedy",
]
