"""
Meta-Learning for Fast Adaptation to New Coins

Implements Model-Agnostic Meta-Learning (MAML) to enable the RL agent to quickly
adapt to new trading pairs with minimal data.

Key idea: Learn an initialization that can be fine-tuned quickly for any new coin.

Traditional RL: Train from scratch on each coin (slow, data-hungry)
Meta-Learning: Learn how to learn, then adapt quickly (fast, data-efficient)

Example:
- Base training on BTC/ETH/SOL/AVAX (meta-learning phase)
- New coin appears (e.g., SUI/USD)
- Fine-tune with just 100 trades → performance comparable to 10,000 trades
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import structlog

from .rl_agent import RLTradingAgent, TradingState, TradingAction

logger = structlog.get_logger(__name__)


@dataclass
class MetaLearningConfig:
    """Meta-learning hyperparameters."""

    inner_lr: float = 0.01  # Learning rate for task-specific adaptation
    meta_lr: float = 0.001  # Learning rate for meta-update
    inner_steps: int = 5  # Gradient steps per task during adaptation
    meta_batch_size: int = 4  # Number of tasks per meta-update
    support_size: int = 100  # Training examples per task
    query_size: int = 50  # Test examples per task
    max_meta_iterations: int = 1000  # Total meta-training iterations


class MetaLearner:
    """
    Meta-learner that enables fast adaptation to new trading pairs.

    Uses MAML (Model-Agnostic Meta-Learning):
    1. Sample batch of tasks (e.g., different coins or market conditions)
    2. For each task:
       a. Clone current policy
       b. Adapt on support set (inner loop)
       c. Evaluate on query set
    3. Meta-update based on query set performance (outer loop)

    Result: Policy that can adapt to new tasks with few gradient steps.
    """

    def __init__(
        self,
        base_agent: RLTradingAgent,
        config: MetaLearningConfig,
        device: str = "cpu",
    ):
        """
        Initialize meta-learner.

        Args:
            base_agent: Base RL agent to meta-train
            config: Meta-learning configuration
            device: Device for computation
        """
        self.base_agent = base_agent
        self.config = config
        self.device = device

        # Meta-optimizer (updates base policy)
        self.meta_optimizer = optim.Adam(
            self.base_agent.policy.parameters(),
            lr=config.meta_lr,
        )

        # Track meta-learning progress
        self.meta_iteration = 0
        self.meta_losses = []
        self.adaptation_speeds = []  # How quickly tasks adapt

        logger.info(
            "meta_learner_initialized",
            inner_lr=config.inner_lr,
            meta_lr=config.meta_lr,
            inner_steps=config.inner_steps,
            meta_batch_size=config.meta_batch_size,
        )

    def meta_train_step(
        self,
        task_batch: List[Dict[str, List[Tuple[TradingState, TradingAction, float]]]],
    ) -> Dict[str, float]:
        """
        Single meta-training step across a batch of tasks.

        Args:
            task_batch: List of tasks, each with 'support' and 'query' sets
                       Format: {"support": [(state, action, reward), ...],
                                "query": [(state, action, reward), ...]}

        Returns:
            Meta-training metrics
        """
        meta_loss = 0.0
        task_losses = []
        adaptation_improvements = []

        for task in task_batch:
            # 1. Clone current policy for task-specific adaptation
            adapted_policy = self._clone_policy()

            # 2. Inner loop: Adapt on support set
            initial_loss = self._evaluate_policy(adapted_policy, task["support"])

            for _ in range(self.config.inner_steps):
                self._inner_update(adapted_policy, task["support"])

            # 3. Evaluate adapted policy on query set
            query_loss = self._evaluate_policy(adapted_policy, task["query"])

            # Track adaptation improvement
            improvement = initial_loss - query_loss
            adaptation_improvements.append(improvement)

            # Accumulate meta-loss
            meta_loss += query_loss
            task_losses.append(query_loss)

        # 4. Meta-update: Update base policy based on query performance
        meta_loss = meta_loss / len(task_batch)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.base_agent.policy.parameters(), 1.0)
        self.meta_optimizer.step()

        # Update tracking
        self.meta_iteration += 1
        self.meta_losses.append(meta_loss.item())
        avg_improvement = sum(adaptation_improvements) / len(adaptation_improvements)
        self.adaptation_speeds.append(avg_improvement)

        metrics = {
            "meta_loss": meta_loss.item(),
            "avg_task_loss": sum(task_losses) / len(task_losses),
            "avg_adaptation_improvement": avg_improvement,
            "meta_iteration": self.meta_iteration,
        }

        logger.debug("meta_train_step_complete", **metrics)
        return metrics

    def fast_adapt(
        self,
        new_task_data: List[Tuple[TradingState, TradingAction, float]],
        adaptation_steps: int = 10,
    ) -> RLTradingAgent:
        """
        Quickly adapt to a new task (e.g., new coin).

        Args:
            new_task_data: Training data for new task
            adaptation_steps: Number of gradient steps for adaptation

        Returns:
            Adapted agent ready for new task
        """
        # Clone base policy
        adapted_policy = self._clone_policy()

        # Fine-tune on new task data
        for step in range(adaptation_steps):
            loss = self._inner_update(adapted_policy, new_task_data)

            if step % 5 == 0:
                logger.debug(
                    "adaptation_step",
                    step=step,
                    loss=loss.item() if isinstance(loss, torch.Tensor) else loss,
                )

        # Create new agent with adapted policy
        adapted_agent = RLTradingAgent(
            state_dim=self.base_agent.state_dim,
            memory_store=self.base_agent.memory_store,
            config=self.base_agent.config,
            device=self.device,
        )
        adapted_agent.policy = adapted_policy

        logger.info(
            "fast_adaptation_complete",
            adaptation_steps=adaptation_steps,
            data_size=len(new_task_data),
        )

        return adapted_agent

    def _clone_policy(self) -> nn.Module:
        """Clone the current policy for task-specific adaptation."""
        cloned_policy = type(self.base_agent.policy)(
            self.base_agent.state_dim,
            self.base_agent.policy.hidden_dim,
        ).to(self.device)

        cloned_policy.load_state_dict(self.base_agent.policy.state_dict())
        return cloned_policy

    def _inner_update(
        self,
        policy: nn.Module,
        task_data: List[Tuple[TradingState, TradingAction, float]],
    ) -> torch.Tensor:
        """
        Single inner loop update (task-specific adaptation).

        Args:
            policy: Policy to adapt
            task_data: Training data for this task

        Returns:
            Loss value
        """
        # Create inner optimizer
        inner_optimizer = optim.SGD(policy.parameters(), lr=self.config.inner_lr)

        # Sample batch from task data
        batch_size = min(32, len(task_data))
        batch_indices = torch.randperm(len(task_data))[:batch_size]
        batch = [task_data[i] for i in batch_indices]

        # Compute loss
        total_loss = 0.0
        for state, action, reward in batch:
            # Convert state to tensor
            state_tensor = self._state_to_tensor(state)

            # Get action logits
            logits, _ = policy(state_tensor)

            # Policy loss (simplified - would use PPO loss in practice)
            action_idx = self._action_to_idx(action)
            log_prob = torch.log_softmax(logits, dim=-1)[action_idx]
            loss = -log_prob * reward  # REINFORCE-style loss

            total_loss += loss

        avg_loss = total_loss / batch_size

        # Gradient step
        inner_optimizer.zero_grad()
        avg_loss.backward()
        inner_optimizer.step()

        return avg_loss

    def _evaluate_policy(
        self,
        policy: nn.Module,
        eval_data: List[Tuple[TradingState, TradingAction, float]],
    ) -> torch.Tensor:
        """
        Evaluate policy on evaluation data.

        Args:
            policy: Policy to evaluate
            eval_data: Evaluation data

        Returns:
            Average loss
        """
        with torch.no_grad():
            total_loss = 0.0

            for state, action, reward in eval_data:
                state_tensor = self._state_to_tensor(state)
                logits, _ = policy(state_tensor)

                action_idx = self._action_to_idx(action)
                log_prob = torch.log_softmax(logits, dim=-1)[action_idx]
                loss = -log_prob * reward

                total_loss += loss

            avg_loss = total_loss / len(eval_data)
            return avg_loss

    def _state_to_tensor(self, state: TradingState) -> torch.Tensor:
        """Convert TradingState to tensor."""
        # Extract features from state
        # This would depend on TradingState structure
        features = []

        # Add price features
        if hasattr(state, "price"):
            features.append(state.price)

        # Add technical indicators
        if hasattr(state, "features"):
            features.extend(state.features)

        # Convert to tensor
        return torch.tensor(features, dtype=torch.float32).to(self.device)

    def _action_to_idx(self, action: TradingAction) -> int:
        """Convert TradingAction to index."""
        # Map action to discrete index
        if action == TradingAction.LONG:
            return 0
        elif action == TradingAction.SHORT:
            return 1
        else:  # HOLD
            return 2

    def get_meta_stats(self) -> Dict[str, float]:
        """Get meta-learning statistics."""
        if not self.meta_losses:
            return {
                "meta_iterations": 0,
                "avg_meta_loss": 0.0,
                "avg_adaptation_speed": 0.0,
            }

        return {
            "meta_iterations": self.meta_iteration,
            "avg_meta_loss": sum(self.meta_losses) / len(self.meta_losses),
            "recent_meta_loss": self.meta_losses[-1] if self.meta_losses else 0.0,
            "avg_adaptation_speed": sum(self.adaptation_speeds) / len(self.adaptation_speeds),
            "recent_adaptation_speed": self.adaptation_speeds[-1] if self.adaptation_speeds else 0.0,
        }

    def save(self, path: str) -> None:
        """Save meta-learned policy."""
        torch.save(
            {
                "policy_state_dict": self.base_agent.policy.state_dict(),
                "meta_optimizer_state_dict": self.meta_optimizer.state_dict(),
                "config": self.config,
                "meta_iteration": self.meta_iteration,
                "meta_losses": self.meta_losses,
                "adaptation_speeds": self.adaptation_speeds,
            },
            path,
        )
        logger.info("meta_learner_saved", path=path)

    def load(self, path: str) -> None:
        """Load meta-learned policy."""
        checkpoint = torch.load(path, map_location=self.device)
        self.base_agent.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.meta_optimizer.load_state_dict(checkpoint["meta_optimizer_state_dict"])
        self.meta_iteration = checkpoint["meta_iteration"]
        self.meta_losses = checkpoint["meta_losses"]
        self.adaptation_speeds = checkpoint["adaptation_speeds"]
        logger.info("meta_learner_loaded", path=path, iterations=self.meta_iteration)


def create_task_from_trades(
    trades: List,
    support_ratio: float = 0.7,
) -> Dict[str, List[Tuple[TradingState, TradingAction, float]]]:
    """
    Create a meta-learning task from a list of trades.

    Args:
        trades: List of historical trades
        support_ratio: Fraction of data for support set (rest is query)

    Returns:
        Task dict with support and query sets
    """
    # Split into support and query
    split_idx = int(len(trades) * support_ratio)
    support_trades = trades[:split_idx]
    query_trades = trades[split_idx:]

    # Convert trades to (state, action, reward) tuples
    support_set = []
    for trade in support_trades:
        if hasattr(trade, "entry_state") and hasattr(trade, "action") and hasattr(trade, "reward"):
            support_set.append((trade.entry_state, trade.action, trade.reward))

    query_set = []
    for trade in query_trades:
        if hasattr(trade, "entry_state") and hasattr(trade, "action") and hasattr(trade, "reward"):
            query_set.append((trade.entry_state, trade.action, trade.reward))

    return {"support": support_set, "query": query_set}
