"""Reinforcement Learning agent using PPO for trading decisions."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import structlog

from ..memory.store import MemoryStore
from ..services.costs import CostBreakdown

logger = structlog.get_logger(__name__)


class TradingAction(Enum):
    """Discrete actions the agent can take."""
    DO_NOTHING = 0
    ENTER_LONG_SMALL = 1  # 0.5x position
    ENTER_LONG_NORMAL = 2  # 1.0x position
    ENTER_LONG_LARGE = 3  # 1.5x position
    EXIT_POSITION = 4
    HOLD_POSITION = 5


@dataclass
class TradingState:
    """Complete state representation for the RL agent."""

    # Market features (from FeatureRecipe)
    market_features: np.ndarray  # Shape: (n_features,)

    # Historical context from memory
    similar_pattern_win_rate: float
    similar_pattern_avg_profit: float
    similar_pattern_reliability: float

    # Current position state
    has_position: bool
    position_size_multiplier: float  # 0.0 if no position
    unrealized_pnl_bps: float
    hold_duration_minutes: int

    # Market regime
    volatility_bps: float
    spread_bps: float
    regime_code: int  # 0=low_vol, 1=medium, 2=high, 3=trending, 4=ranging

    # Risk metrics
    current_drawdown_gbp: float
    trades_today: int
    win_rate_today: float
    symbol: Optional[str] = None


class RunningNormalizer:
    """Keeps running statistics for feature normalization."""

    def __init__(self, size: int, eps: float = 1e-6) -> None:
        self._mean = torch.zeros(size, dtype=torch.float32)
        self._var = torch.ones(size, dtype=torch.float32)
        self._count = eps
        self._eps = eps

    def _update(self, batch: torch.Tensor) -> None:
        if batch.numel() == 0:
            return
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        batch_count = batch.shape[0]

        delta = batch_mean - self._mean
        total_count = self._count + batch_count

        new_mean = self._mean + delta * batch_count / total_count
        m_a = self._var * self._count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta.pow(2) * self._count * batch_count / total_count)
        new_var = m2 / total_count

        self._mean = new_mean.detach()
        self._var = torch.clamp(new_var.detach(), min=self._eps)
        self._count = float(total_count)

    def normalize(self, batch: torch.Tensor, update_stats: bool = True) -> torch.Tensor:
        input_batch = batch if batch.dim() > 1 else batch.unsqueeze(0)
        if update_stats:
            self._update(input_batch.cpu())
        mean = self._mean.to(input_batch.device)
        var = self._var.to(input_batch.device)
        normalized = (input_batch - mean) / torch.sqrt(var + self._eps)
        return normalized if batch.dim() > 1 else normalized.squeeze(0)


class ActorCritic(nn.Module):
    """Neural network for PPO: outputs policy (actor) and value estimate (critic)."""

    def __init__(self, state_dim: int, n_actions: int, hidden_size: int = 256):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Policy head (actor)
        self.policy = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, n_actions),
        )

        # Value head (critic)
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            action_logits: Un-normalized logits for each action
            state_value: Estimated value of the state
        """
        features = self.shared(state)
        action_logits = self.policy(features)
        state_value = self.value(features)
        return action_logits, state_value


@dataclass
class PPOConfig:
    """Hyperparameters for PPO training."""
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    clip_epsilon: float = 0.2  # PPO clipping parameter
    value_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus for exploration
    max_grad_norm: float = 0.5
    n_epochs: int = 10  # Epochs per update
    batch_size: int = 64
    gae_lambda: float = 0.95
    min_pattern_reliability: float = 0.15
    large_position_penalty: float = 2.0
    inaction_penalty: float = 1.0
    drawdown_exit_trigger_bps: float = 5.0
    drawdown_exit_boost: float = 2.0
    action_temperature: float = 1.0


class RLTradingAgent:
    """
    PPO-based trading agent with memory-augmented decision making.

    The agent learns optimal entry/exit timing and position sizing by:
    1. Observing market state + historical pattern performance
    2. Choosing actions (enter/hold/exit with various sizes)
    3. Receiving rewards based on actual P&L
    4. Updating policy to maximize expected cumulative reward
    """

    def __init__(
        self,
        state_dim: int,
        memory_store: MemoryStore,
        config: Optional[PPOConfig] = None,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.n_actions = len(TradingAction)
        self.memory_store = memory_store
        self.config = config or PPOConfig()
        self.device = torch.device(device)

        # Initialize network
        self.network = ActorCritic(state_dim, self.n_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        self.normalizer = RunningNormalizer(state_dim)

        # Experience buffer
        self.states: List[torch.Tensor] = []
        self.state_context: List[TradingState] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.dones: List[bool] = []
        self.bootstrap_value: float = 0.0

        logger.info("rl_agent_initialized", state_dim=state_dim, n_actions=self.n_actions)

    def state_to_tensor(self, state: TradingState) -> np.ndarray:
        """Convert TradingState to flat numpy array."""
        features = [
            *state.market_features,
            state.similar_pattern_win_rate,
            state.similar_pattern_avg_profit,
            state.similar_pattern_reliability,
            float(state.has_position),
            state.position_size_multiplier,
            state.unrealized_pnl_bps,
            state.hold_duration_minutes / 1440.0,  # Normalize to days
            state.volatility_bps / 100.0,
            state.spread_bps / 10.0,
            state.regime_code / 4.0,  # Normalize
            state.current_drawdown_gbp / 100.0,
            state.trades_today / 50.0,  # Normalize
            state.win_rate_today,
        ]
        return np.array(features, dtype=np.float32)

    def select_action(
        self,
        state: TradingState,
        deterministic: bool = False,
    ) -> Tuple[TradingAction, float]:
        """
        Select action based on current state.

        Args:
            state: Current trading state
            deterministic: If True, select best action; if False, sample from policy

        Returns:
            action: Selected action
            log_prob: Log probability of the action (for training)
        """
        state_array = self.state_to_tensor(state)
        state_tensor = torch.from_numpy(state_array).to(self.device)
        normalized_state = self.normalizer.normalize(state_tensor, update_stats=not deterministic)
        network_input = normalized_state.unsqueeze(0)

        with torch.no_grad():
            logits, state_value = self.network(network_input)

        logits = logits.squeeze(0) / max(self.config.action_temperature, 1e-6)
        logits = self._apply_contextual_bias(logits, state)
        dist = Categorical(logits=logits)

        if deterministic:
            action_idx = torch.argmax(logits).item()
            action_tensor = torch.tensor(action_idx, device=self.device)
        else:
            action_tensor = dist.sample()
            action_idx = action_tensor.item()

        log_prob_tensor = dist.log_prob(action_tensor)
        action = TradingAction(action_idx)

        # Store for training
        self.states.append(normalized_state.detach().clone().to(self.device))
        self.state_context.append(copy.deepcopy(state))
        self.actions.append(action_idx)
        self.log_probs.append(log_prob_tensor.detach())
        self.values.append(state_value.squeeze().detach())

        return action, float(log_prob_tensor.item())

    def _apply_contextual_bias(self, logits: torch.Tensor, state: TradingState) -> torch.Tensor:
        """Mask invalid actions and inject context-aware biases."""
        adjusted = logits.clone()

        # Mask actions that are invalid given position state
        mask = torch.zeros_like(adjusted, dtype=torch.bool)
        if state.has_position:
            mask[TradingAction.ENTER_LONG_SMALL.value] = True
            mask[TradingAction.ENTER_LONG_NORMAL.value] = True
            mask[TradingAction.ENTER_LONG_LARGE.value] = True
        else:
            mask[TradingAction.EXIT_POSITION.value] = True
            mask[TradingAction.HOLD_POSITION.value] = True

        adjusted[mask] = -1e9

        # Reliability-aware penalties to curb oversized entries
        reliability = float(state.similar_pattern_reliability)
        reliability = max(0.0, min(reliability, 1.0))
        if reliability < self.config.min_pattern_reliability:
            penalty = (self.config.min_pattern_reliability - reliability) * self.config.large_position_penalty
            adjusted[TradingAction.ENTER_LONG_LARGE.value] -= penalty
        else:
            adjusted[TradingAction.DO_NOTHING.value] -= (1.0 - reliability) * self.config.inaction_penalty

        # Encourage exits when underwater beyond configured drawdown
        if state.has_position and state.unrealized_pnl_bps <= -self.config.drawdown_exit_trigger_bps:
            adjusted[TradingAction.EXIT_POSITION.value] += self.config.drawdown_exit_boost

        return adjusted

    def store_reward(self, reward: float, done: bool) -> None:
        """Store reward for the last action."""
        self.rewards.append(reward)
        self.dones.append(done)

    def update(self, next_state: Optional[TradingState] = None) -> Dict[str, float]:
        """Update policy using PPO with GAE advantages and contextual logits."""
        if len(self.states) < self.config.batch_size:
            logger.warning("insufficient_experience", size=len(self.states))
            return {}

        if next_state is not None:
            next_array = self.state_to_tensor(next_state)
            next_tensor = torch.from_numpy(next_array).to(self.device)
            normalized_next = self.normalizer.normalize(next_tensor, update_stats=False)
            with torch.no_grad():
                _, next_value = self.network(normalized_next.unsqueeze(0))
            self.bootstrap_value = float(next_value.squeeze().item())
        else:
            self.bootstrap_value = 0.0

        states_tensor = torch.stack(self.states).to(self.device)
        actions_tensor = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        old_log_probs_tensor = torch.stack(self.log_probs).to(self.device).detach()
        rewards_tensor = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(self.dones, dtype=torch.float32, device=self.device)
        values_tensor = torch.stack(self.values).to(self.device)

        value_targets = torch.cat(
            [values_tensor, torch.tensor([self.bootstrap_value], dtype=torch.float32, device=self.device)],
            dim=0,
        )

        advantages = torch.zeros_like(rewards_tensor, device=self.device)
        gae = torch.tensor(0.0, device=self.device)
        for idx in reversed(range(rewards_tensor.shape[0])):
            mask = 1.0 - dones_tensor[idx]
            delta = rewards_tensor[idx] + self.config.gamma * value_targets[idx + 1] * mask - value_targets[idx]
            gae = delta + self.config.gamma * self.config.gae_lambda * mask * gae
            advantages[idx] = gae

        returns_tensor = advantages + value_targets[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages_detached = advantages.detach()
        returns_tensor = returns_tensor.detach()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.config.n_epochs):
            logits_batch, state_values = self.network(states_tensor)
            adjusted_logits = torch.stack(
                [
                    self._apply_contextual_bias(logit_vec, ctx)
                    for logit_vec, ctx in zip(logits_batch, self.state_context)
                ]
            )

            dist = Categorical(logits=adjusted_logits)
            new_log_probs = dist.log_prob(actions_tensor)
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)

            surr1 = ratio * advantages_detached
            surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * advantages_detached
            policy_loss = -torch.min(surr1, surr2).mean()

            value_pred = state_values.squeeze()
            value_loss = nn.MSELoss()(value_pred, returns_tensor)

            entropy = dist.entropy().mean()

            loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            n_updates += 1

        self.states.clear()
        self.state_context.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        self.bootstrap_value = 0.0

        metrics = {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

        logger.info("agent_updated", **metrics)
        return metrics

    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, path)
        logger.info("model_saved", path=path)

    def load(self, path: str) -> None:
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("model_loaded", path=path)

    def calculate_reward(
        self,
        action: TradingAction,
        profit_gbp: float,
        missed_profit_gbp: float,
        hold_duration_minutes: int,
        *,
        position_size_gbp: float,
        costs: Optional[CostBreakdown] = None,
        state: Optional[TradingState] = None,
        target_hold_minutes: int = 30,
    ) -> float:
        """Calculate shaped reward incorporating costs, reliability, and time efficiency."""

        notional = max(position_size_gbp, 1e-6)
        pnl_bps = (profit_gbp / notional) * 10_000.0
        missed_bps = (missed_profit_gbp / notional) * 10_000.0
        reward = pnl_bps

        if costs:
            reward -= costs.total_costs_bps

        if missed_bps > 5.0:
            reward -= 0.5 * missed_bps

        if hold_duration_minutes > target_hold_minutes * 2:
            reward -= 2.0
        elif profit_gbp > 0.0 and hold_duration_minutes <= target_hold_minutes:
            reward += 1.0

        if action == TradingAction.DO_NOTHING:
            reward -= self.config.inaction_penalty

        if state is not None:
            reliability = float(state.similar_pattern_reliability)
            reward += reliability * 5.0
            if reliability < self.config.min_pattern_reliability:
                reward -= (self.config.min_pattern_reliability - reliability) * self.config.large_position_penalty

            if state.has_position and state.unrealized_pnl_bps <= -self.config.drawdown_exit_trigger_bps:
                reward -= self.config.drawdown_exit_boost

        if action == TradingAction.ENTER_LONG_LARGE and state is not None:
            reliability = float(state.similar_pattern_reliability)
            if reliability < self.config.min_pattern_reliability:
                reward -= self.config.large_position_penalty * 5.0

        return float(reward)
