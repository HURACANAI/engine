"""
Hierarchical Reinforcement Learning

Implements a two-level hierarchy:
1. High-Level Agent (Manager): Selects trading strategy
2. Low-Level Agent (Worker): Executes the selected strategy

This decomposes the trading problem into:
- Strategy selection (what to do): trend-follow, mean-revert, breakout, etc.
- Execution (how to do it): entry timing, position sizing, exit management

Benefits:
- Temporal abstraction: Manager operates on longer timescales
- Specialization: Workers become experts at their assigned strategies
- Explainability: Clear separation between "why" and "how"
- Transfer learning: Workers can be reused across different assets

Example:
Manager observes market: "High momentum + breakout setup"
Manager selects strategy: "BREAKOUT_LONG"
Worker executes: Waits for confirmation, enters on volume spike, manages stop/take-profit
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import structlog

from .rl_agent import RLTradingAgent, TradingState, TradingAction, PPOConfig

logger = structlog.get_logger(__name__)


class TradingStrategy(Enum):
    """High-level trading strategies."""

    TREND_FOLLOW_LONG = "trend_follow_long"  # Ride uptrends
    TREND_FOLLOW_SHORT = "trend_follow_short"  # Ride downtrends
    MEAN_REVERT_LONG = "mean_revert_long"  # Buy oversold
    MEAN_REVERT_SHORT = "mean_revert_short"  # Sell overbought
    BREAKOUT_LONG = "breakout_long"  # Buy breakouts
    BREAKOUT_SHORT = "breakout_short"  # Sell breakdowns
    HOLD = "hold"  # Stay in cash


@dataclass
class StrategyGoal:
    """Goal specification for worker agents."""

    strategy: TradingStrategy
    target_return_bps: float  # Expected return
    max_duration_minutes: int  # Time horizon
    risk_tolerance: float  # 0-1, how much risk to take
    urgency: float  # 0-1, how urgent is entry


@dataclass
class HierarchicalConfig:
    """Hierarchical RL configuration."""

    manager_decision_interval: int = 10  # Manager decides every N minutes
    num_strategies: int = 7  # Number of distinct strategies
    worker_hidden_dim: int = 128  # Worker network size
    manager_hidden_dim: int = 256  # Manager network size
    intrinsic_reward_weight: float = 0.1  # Weight for goal completion
    enable_option_termination: bool = True  # Allow early strategy termination


class ManagerAgent(nn.Module):
    """
    High-level agent that selects trading strategies.

    Input: Market state (features, regime, momentum, etc.)
    Output: Strategy selection + goal specification

    Operates on a slower timescale than workers (e.g., every 10 minutes).
    """

    def __init__(
        self,
        state_dim: int,
        num_strategies: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.num_strategies = num_strategies

        # Strategy selection network
        self.strategy_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_strategies),
        )

        # Goal specification network (target return, duration, risk)
        self.goal_network = nn.Sequential(
            nn.Linear(state_dim + num_strategies, hidden_dim),  # +strategy one-hot
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # [target_return, max_duration, risk_tolerance]
            nn.Tanh(),  # Bound outputs to [-1, 1]
        )

        # Value network for manager
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select strategy and specify goals.

        Returns:
            (strategy_logits, goal_params, state_value)
        """
        # Select strategy
        strategy_logits = self.strategy_network(state)
        strategy_probs = torch.softmax(strategy_logits, dim=-1)

        # Specify goals (conditional on selected strategy)
        strategy_onehot = torch.nn.functional.one_hot(
            torch.argmax(strategy_probs, dim=-1),
            num_classes=self.num_strategies,
        ).float()

        goal_input = torch.cat([state, strategy_onehot], dim=-1)
        goal_params = self.goal_network(goal_input)

        # Estimate state value
        value = self.value_network(state)

        return strategy_logits, goal_params, value


class WorkerAgent(nn.Module):
    """
    Low-level agent that executes a specific strategy.

    Input: Market state + goal specification
    Output: Concrete trading actions (LONG/SHORT/HOLD)

    Each worker is conditioned on its assigned goal, learning to optimize
    for that specific objective.
    """

    def __init__(
        self,
        state_dim: int,
        goal_dim: int = 3,  # [target_return, max_duration, risk_tolerance]
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.goal_dim = goal_dim

        # Policy network (state + goal → action)
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # [LONG, SHORT, HOLD]
        )

        # Value network
        self.value_network = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Termination network (decides if strategy is complete)
        self.termination_network = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Probability of termination
        )

    def forward(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute strategy given state and goal.

        Returns:
            (action_logits, state_value, termination_prob)
        """
        # Concatenate state and goal
        combined = torch.cat([state, goal], dim=-1)

        # Get action distribution
        action_logits = self.policy_network(combined)

        # Estimate value
        value = self.value_network(combined)

        # Termination probability
        termination_prob = self.termination_network(combined)

        return action_logits, value, termination_prob


class HierarchicalRLAgent:
    """
    Complete hierarchical RL system.

    Manages the two-level hierarchy and coordinates manager-worker interaction.
    """

    def __init__(
        self,
        state_dim: int,
        config: HierarchicalConfig,
        device: str = "cpu",
    ):
        """
        Initialize hierarchical agent.

        Args:
            state_dim: State dimension
            config: Hierarchical configuration
            device: Device for computation
        """
        self.state_dim = state_dim
        self.config = config
        self.device = device

        # Manager agent
        self.manager = ManagerAgent(
            state_dim=state_dim,
            num_strategies=config.num_strategies,
            hidden_dim=config.manager_hidden_dim,
        ).to(device)

        # Worker agent
        self.worker = WorkerAgent(
            state_dim=state_dim,
            goal_dim=3,
            hidden_dim=config.worker_hidden_dim,
        ).to(device)

        # Optimizers
        self.manager_optimizer = torch.optim.Adam(
            self.manager.parameters(),
            lr=0.0003,
        )
        self.worker_optimizer = torch.optim.Adam(
            self.worker.parameters(),
            lr=0.001,
        )

        # Tracking
        self.current_strategy: Optional[TradingStrategy] = None
        self.current_goal: Optional[StrategyGoal] = None
        self.strategy_start_time: int = 0
        self.steps_since_manager_decision: int = 0

        logger.info(
            "hierarchical_rl_initialized",
            manager_interval=config.manager_decision_interval,
            num_strategies=config.num_strategies,
        )

    def select_action(
        self,
        state: TradingState,
        timestep: int,
        deterministic: bool = False,
    ) -> Tuple[TradingAction, float, Dict]:
        """
        Select action using hierarchical policy.

        Args:
            state: Current trading state
            timestep: Current timestep
            deterministic: Whether to use deterministic actions

        Returns:
            (action, confidence, metadata)
        """
        state_tensor = self._state_to_tensor(state)

        # Manager decision (every N steps or if strategy terminated)
        if self._should_manager_decide(timestep):
            strategy, goal = self._manager_decision(state_tensor, deterministic)
            self.current_strategy = strategy
            self.current_goal = goal
            self.strategy_start_time = timestep
            self.steps_since_manager_decision = 0

            logger.debug(
                "manager_decision",
                strategy=strategy.value,
                target_return_bps=goal.target_return_bps,
                timestep=timestep,
            )

        # Worker execution
        action, confidence, should_terminate = self._worker_execution(
            state_tensor,
            self.current_goal,
            deterministic,
        )

        # Check termination
        if should_terminate and self.config.enable_option_termination:
            self.current_strategy = None  # Force manager re-decision

        self.steps_since_manager_decision += 1

        metadata = {
            "strategy": self.current_strategy.value if self.current_strategy else "none",
            "goal_target_return": self.current_goal.target_return_bps if self.current_goal else 0.0,
            "should_terminate": should_terminate,
            "steps_in_strategy": self.steps_since_manager_decision,
        }

        return action, confidence, metadata

    def _should_manager_decide(self, timestep: int) -> bool:
        """Check if manager should make a new decision."""
        # Initial decision
        if self.current_strategy is None:
            return True

        # Regular interval
        if self.steps_since_manager_decision >= self.config.manager_decision_interval:
            return True

        return False

    def _manager_decision(
        self,
        state: torch.Tensor,
        deterministic: bool,
    ) -> Tuple[TradingStrategy, StrategyGoal]:
        """Manager selects strategy and specifies goal."""
        with torch.no_grad():
            strategy_logits, goal_params, _ = self.manager(state)

            # Select strategy
            if deterministic:
                strategy_idx = torch.argmax(strategy_logits, dim=-1).item()
            else:
                strategy_probs = torch.softmax(strategy_logits, dim=-1)
                strategy_idx = torch.multinomial(strategy_probs, 1).item()

            # Map to strategy enum
            strategies = list(TradingStrategy)
            strategy = strategies[strategy_idx]

            # Parse goal parameters (tanh outputs in [-1, 1])
            target_return_bps = goal_params[0].item() * 200.0  # [-200, 200] bps
            max_duration = int((goal_params[1].item() + 1.0) * 60)  # [0, 120] minutes
            risk_tolerance = (goal_params[2].item() + 1.0) / 2.0  # [0, 1]

            goal = StrategyGoal(
                strategy=strategy,
                target_return_bps=target_return_bps,
                max_duration_minutes=max_duration,
                risk_tolerance=risk_tolerance,
                urgency=0.5,  # Default urgency
            )

            return strategy, goal

    def _worker_execution(
        self,
        state: torch.Tensor,
        goal: StrategyGoal,
        deterministic: bool,
    ) -> Tuple[TradingAction, float, bool]:
        """Worker executes the strategy."""
        # Convert goal to tensor
        goal_tensor = torch.tensor(
            [
                goal.target_return_bps / 200.0,  # Normalize to [-1, 1]
                (goal.max_duration_minutes / 60.0) - 1.0,  # Normalize to [-1, 1]
                goal.risk_tolerance * 2.0 - 1.0,  # Normalize to [-1, 1]
            ],
            dtype=torch.float32,
        ).to(self.device)

        with torch.no_grad():
            action_logits, _, termination_prob = self.worker(state, goal_tensor)

            # Select action
            if deterministic:
                action_idx = torch.argmax(action_logits, dim=-1).item()
            else:
                action_probs = torch.softmax(action_logits, dim=-1)
                action_idx = torch.multinomial(action_probs, 1).item()

            # Map to action
            actions = [TradingAction.LONG, TradingAction.SHORT, TradingAction.HOLD]
            action = actions[action_idx]

            # Get confidence
            confidence = torch.softmax(action_logits, dim=-1)[action_idx].item()

            # Check termination
            should_terminate = termination_prob.item() > 0.5

            return action, confidence, should_terminate

    def _state_to_tensor(self, state: TradingState) -> torch.Tensor:
        """Convert state to tensor."""
        # Simplified - would extract full feature vector
        features = []
        if hasattr(state, "features"):
            features = state.features
        return torch.tensor(features, dtype=torch.float32).to(self.device)

    def update(
        self,
        manager_experiences: List,
        worker_experiences: List,
    ) -> Dict[str, float]:
        """
        Update both manager and worker.

        Args:
            manager_experiences: High-level transitions
            worker_experiences: Low-level transitions

        Returns:
            Update metrics
        """
        # Update worker
        worker_loss = self._update_worker(worker_experiences)

        # Update manager
        manager_loss = self._update_manager(manager_experiences)

        return {
            "worker_loss": worker_loss,
            "manager_loss": manager_loss,
        }

    def _update_worker(self, experiences: List) -> float:
        """Update worker with PPO."""
        # Simplified - would implement full PPO update
        return 0.0

    def _update_manager(self, experiences: List) -> float:
        """Update manager with PPO."""
        # Simplified - would implement full PPO update with intrinsic rewards
        return 0.0

    def get_stats(self) -> Dict[str, any]:
        """Get hierarchical RL statistics."""
        return {
            "current_strategy": self.current_strategy.value if self.current_strategy else "none",
            "steps_in_strategy": self.steps_since_manager_decision,
            "manager_interval": self.config.manager_decision_interval,
        }

    def save(self, path: str) -> None:
        """Save hierarchical agent."""
        torch.save(
            {
                "manager_state_dict": self.manager.state_dict(),
                "worker_state_dict": self.worker.state_dict(),
                "manager_optimizer": self.manager_optimizer.state_dict(),
                "worker_optimizer": self.worker_optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )
        logger.info("hierarchical_agent_saved", path=path)

    def load(self, path: str) -> None:
        """Load hierarchical agent."""
        checkpoint = torch.load(path, map_location=self.device)
        self.manager.load_state_dict(checkpoint["manager_state_dict"])
        self.worker.load_state_dict(checkpoint["worker_state_dict"])
        self.manager_optimizer.load_state_dict(checkpoint["manager_optimizer"])
        self.worker_optimizer.load_state_dict(checkpoint["worker_optimizer"])
        logger.info("hierarchical_agent_loaded", path=path)
