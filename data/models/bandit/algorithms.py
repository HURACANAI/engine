"""
Bandit Algorithms

Implementation of different multi-armed bandit algorithms.
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BanditAlgorithm(ABC):
    """Base class for bandit algorithms"""

    @abstractmethod
    def select_arm(
        self,
        num_pulls: np.ndarray,
        rewards: np.ndarray,
        **kwargs
    ) -> int:
        """
        Select which arm to pull

        Args:
            num_pulls: Number of pulls per arm
            rewards: Cumulative rewards per arm
            **kwargs: Additional context

        Returns:
            Index of arm to pull
        """
        pass


class ThompsonSampling(BanditAlgorithm):
    """
    Thompson Sampling

    Bayesian approach that samples from posterior distributions.
    Works well for exploration-exploitation tradeoff.
    """

    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        """
        Initialize Thompson Sampling

        Args:
            alpha_prior: Prior alpha (successes)
            beta_prior: Prior beta (failures)
        """
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

    def select_arm(
        self,
        num_pulls: np.ndarray,
        rewards: np.ndarray,
        **kwargs
    ) -> int:
        """
        Select arm using Thompson Sampling

        Args:
            num_pulls: Number of pulls per arm
            rewards: Cumulative rewards per arm

        Returns:
            Selected arm index
        """
        num_arms = len(num_pulls)

        # Sample from Beta distribution for each arm
        samples = np.zeros(num_arms)

        for i in range(num_arms):
            if num_pulls[i] == 0:
                # No data - sample from prior
                alpha = self.alpha_prior
                beta = self.beta_prior
            else:
                # Update posterior with observed data
                # Assuming rewards are in [0, 1] range
                successes = rewards[i]  # Total reward
                failures = num_pulls[i] - successes

                alpha = self.alpha_prior + successes
                beta = self.beta_prior + max(0, failures)

            # Sample from Beta(alpha, beta)
            samples[i] = np.random.beta(alpha, beta)

        # Select arm with highest sample
        selected_arm = int(np.argmax(samples))

        return selected_arm


class UCB(BanditAlgorithm):
    """
    Upper Confidence Bound (UCB1)

    Selects arm with highest upper confidence bound.
    Balances exploitation (high mean) with exploration (high uncertainty).
    """

    def __init__(self, exploration_param: float = 2.0):
        """
        Initialize UCB

        Args:
            exploration_param: Exploration parameter (higher = more exploration)
        """
        self.exploration_param = exploration_param

    def select_arm(
        self,
        num_pulls: np.ndarray,
        rewards: np.ndarray,
        total_pulls: int = None,
        **kwargs
    ) -> int:
        """
        Select arm using UCB1

        Args:
            num_pulls: Number of pulls per arm
            rewards: Cumulative rewards per arm
            total_pulls: Total pulls across all arms

        Returns:
            Selected arm index
        """
        num_arms = len(num_pulls)

        # If any arm hasn't been pulled, pull it
        for i in range(num_arms):
            if num_pulls[i] == 0:
                return i

        # Calculate total pulls if not provided
        if total_pulls is None:
            total_pulls = num_pulls.sum()

        # Calculate UCB for each arm
        ucb_values = np.zeros(num_arms)

        for i in range(num_arms):
            # Mean reward
            mean_reward = rewards[i] / num_pulls[i]

            # Confidence bound
            confidence = np.sqrt(
                (self.exploration_param * np.log(total_pulls)) / num_pulls[i]
            )

            ucb_values[i] = mean_reward + confidence

        # Select arm with highest UCB
        selected_arm = int(np.argmax(ucb_values))

        return selected_arm


class EpsilonGreedy(BanditAlgorithm):
    """
    Epsilon-Greedy

    With probability epsilon, explore randomly.
    With probability 1-epsilon, exploit best arm.
    """

    def __init__(self, epsilon: float = 0.1, decay: bool = True):
        """
        Initialize Epsilon-Greedy

        Args:
            epsilon: Exploration probability
            decay: Whether to decay epsilon over time
        """
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.decay = decay
        self.num_selections = 0

    def select_arm(
        self,
        num_pulls: np.ndarray,
        rewards: np.ndarray,
        **kwargs
    ) -> int:
        """
        Select arm using epsilon-greedy

        Args:
            num_pulls: Number of pulls per arm
            rewards: Cumulative rewards per arm

        Returns:
            Selected arm index
        """
        num_arms = len(num_pulls)

        # Decay epsilon over time
        if self.decay:
            self.epsilon = self.initial_epsilon / (1 + self.num_selections * 0.01)

        # Explore with probability epsilon
        if np.random.random() < self.epsilon:
            # Random exploration
            selected_arm = np.random.randint(0, num_arms)
        else:
            # Exploit best arm
            # Calculate mean rewards
            mean_rewards = np.zeros(num_arms)
            for i in range(num_arms):
                if num_pulls[i] > 0:
                    mean_rewards[i] = rewards[i] / num_pulls[i]
                else:
                    mean_rewards[i] = 0.0

            selected_arm = int(np.argmax(mean_rewards))

        self.num_selections += 1

        return selected_arm


class ContextualBandit(BanditAlgorithm):
    """
    Contextual Bandit (LinUCB)

    Uses context features to make decisions.
    Useful for regime-aware strategy selection.
    """

    def __init__(
        self,
        num_arms: int,
        context_dim: int,
        alpha: float = 1.0
    ):
        """
        Initialize contextual bandit

        Args:
            num_arms: Number of arms
            context_dim: Dimension of context features
            alpha: Exploration parameter
        """
        self.num_arms = num_arms
        self.context_dim = context_dim
        self.alpha = alpha

        # Initialize parameters for each arm
        self.A = [
            np.identity(context_dim)
            for _ in range(num_arms)
        ]
        self.b = [
            np.zeros((context_dim, 1))
            for _ in range(num_arms)
        ]

    def select_arm(
        self,
        context: np.ndarray,
        **kwargs
    ) -> int:
        """
        Select arm based on context

        Args:
            context: Context feature vector

        Returns:
            Selected arm index
        """
        context = np.array(context).reshape(-1, 1)

        ucb_values = np.zeros(self.num_arms)

        for arm in range(self.num_arms):
            # Estimate parameters
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv.dot(self.b[arm])

            # Calculate UCB
            mean_reward = theta.T.dot(context)[0, 0]
            confidence = self.alpha * np.sqrt(
                context.T.dot(A_inv).dot(context)[0, 0]
            )

            ucb_values[arm] = mean_reward + confidence

        selected_arm = int(np.argmax(ucb_values))

        return selected_arm

    def update(
        self,
        arm: int,
        context: np.ndarray,
        reward: float
    ) -> None:
        """
        Update arm parameters

        Args:
            arm: Arm that was pulled
            context: Context used
            reward: Observed reward
        """
        context = np.array(context).reshape(-1, 1)

        self.A[arm] += context.dot(context.T)
        self.b[arm] += reward * context
