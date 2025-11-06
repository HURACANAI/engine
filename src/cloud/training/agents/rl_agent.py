"""Reinforcement Learning agent using PPO for trading decisions."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.utils.clip_grad import clip_grad_norm_
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
    # Dual-mode actions
    SCRATCH = 6  # Fast exit for short-hold (minimize loss)
    ADD_GRID = 7  # Add to position (DCA for long-hold)
    SCALE_OUT = 8  # Partial exit (long-hold)
    TRAIL_RUNNER = 9  # Activate trailing stop (long-hold)


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
    recent_return_1m: float
    recent_return_5m: float
    recent_return_30m: float
    volume_zscore: float
    volatility_zscore: float
    estimated_transaction_cost_bps: float
    trend_flag_5m: float = 0.0
    trend_flag_1h: float = 0.0
    orderbook_imbalance: float = 0.0
    flow_trend_score: float = 0.0
    symbol: Optional[str] = None

    # Dual-mode fields
    trading_mode: str = "short_hold"  # "short_hold" or "long_hold"
    has_short_position: bool = False
    has_long_position: bool = False
    short_position_pnl_bps: float = 0.0
    long_position_pnl_bps: float = 0.0
    long_position_age_hours: float = 0.0
    num_adds: int = 0  # Number of adds for long-hold
    be_lock_active: bool = False
    trail_active: bool = False


class ExperienceReplayBuffer:
    """
    Keeps a curriculum buffer of past experiences for regime-aware replay.

    Supports context-aware sampling that prioritizes experiences from:
    1. Current market regime (70% weight)
    2. Other regimes (30% weight)

    This implements "curriculum learning" - focus training on what's relevant NOW.
    """

    def __init__(self, capacity: int, regime_focus_weight: float = 0.7) -> None:
        """
        Initialize experience replay buffer.

        Args:
            capacity: Maximum number of experiences to store
            regime_focus_weight: Weight for current regime experiences (0-1, default 0.7)
        """
        self._capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)
        self._regime_focus_weight = regime_focus_weight

        logger.info(
            "experience_replay_buffer_initialized",
            capacity=capacity,
            regime_focus_weight=regime_focus_weight,
        )

    def add_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        contexts: List[TradingState],
    ) -> None:
        """Add batch of experiences to buffer with regime tracking."""
        for idx in range(actions.shape[0]):
            # Extract regime from context (map regime_code to string)
            regime_code = contexts[idx].regime_code if hasattr(contexts[idx], "regime_code") else 0
            regime_name = self._regime_code_to_name(regime_code)

            self._buffer.append({
                "state": states[idx].detach().cpu(),
                "action": actions[idx].detach().cpu(),
                "log_prob": log_probs[idx].detach().cpu(),
                "advantage": advantages[idx].detach().cpu(),
                "return": returns[idx].detach().cpu(),
                "context": copy.deepcopy(contexts[idx]),
                "regime": regime_name,  # Track regime for weighted sampling
            })

    def sample(
        self,
        count: int,
        current_regime: Optional[str] = None,
        use_regime_weighting: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Sample experiences with optional regime-weighted prioritization.

        Args:
            count: Number of experiences to sample
            current_regime: Current market regime for weighted sampling
            use_regime_weighting: Enable regime-weighted sampling

        Returns:
            Dictionary of sampled experiences, or None if buffer empty
        """
        available = len(self._buffer)
        if available == 0 or count <= 0:
            return None

        count = min(count, available)

        # Regime-weighted sampling (curriculum learning)
        if use_regime_weighting and current_regime:
            indices = self._sample_regime_weighted(count, current_regime)
        else:
            # Standard uniform sampling
            indices = np.random.choice(available, count, replace=False)

        selected = [self._buffer[idx] for idx in indices]

        states = torch.stack([item["state"] for item in selected])
        actions = torch.stack([item["action"] for item in selected])
        log_probs = torch.stack([item["log_prob"] for item in selected])
        advantages = torch.stack([item["advantage"] for item in selected])
        returns = torch.stack([item["return"] for item in selected])
        contexts = [item["context"] for item in selected]

        return {
            "states": states,
            "actions": actions,
            "log_probs": log_probs,
            "advantages": advantages,
            "returns": returns,
            "contexts": contexts,
        }

    def _sample_regime_weighted(self, count: int, current_regime: str) -> np.ndarray:
        """
        Sample experiences with higher probability for current regime.

        Implements curriculum learning: 70% from current regime, 30% from others.

        Args:
            count: Number of samples
            current_regime: Current market regime

        Returns:
            Array of selected indices
        """
        available = len(self._buffer)

        # Separate indices by regime match
        matching_indices = []
        other_indices = []

        for idx in range(available):
            if self._buffer[idx].get("regime") == current_regime:
                matching_indices.append(idx)
            else:
                other_indices.append(idx)

        # Calculate split (70% current regime, 30% others)
        target_matching = int(count * self._regime_focus_weight)
        target_other = count - target_matching

        # Sample from each group
        selected_indices = []

        if matching_indices and target_matching > 0:
            actual_matching = min(target_matching, len(matching_indices))
            selected_indices.extend(
                np.random.choice(matching_indices, actual_matching, replace=False)
            )

        if other_indices and target_other > 0:
            actual_other = min(target_other, len(other_indices))
            selected_indices.extend(
                np.random.choice(other_indices, actual_other, replace=False)
            )

        # If we didn't get enough samples, fill from whatever's available
        if len(selected_indices) < count:
            remaining = count - len(selected_indices)
            all_indices = list(range(available))
            # Remove already selected
            available_indices = [idx for idx in all_indices if idx not in selected_indices]
            if available_indices:
                additional = min(remaining, len(available_indices))
                selected_indices.extend(
                    np.random.choice(available_indices, additional, replace=False)
                )

        return np.array(selected_indices)

    def _regime_code_to_name(self, regime_code: int) -> str:
        """Map regime code to string name."""
        regime_map = {
            0: "low_vol",
            1: "medium_vol",
            2: "high_vol",
            3: "trending",
            4: "ranging",
        }
        return regime_map.get(regime_code, "unknown")

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._buffer)


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
    cost_penalty_scale: float = 0.5
    replay_buffer_size: int = 4096
    replay_sample_ratio: float = 0.3
    replay_min_samples: int = 256


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
        
        # Adaptive learning rate scheduler
        try:
            from src.cloud.training.optimization.adaptive_lr_scheduler import AdaptiveLRScheduler
            self.lr_scheduler = AdaptiveLRScheduler(
                optimizer=self.optimizer,
                base_lr=self.config.learning_rate,
                min_lr=1e-5,
                max_lr=1e-3,
            )
            self.use_adaptive_lr = True
        except ImportError:
            self.lr_scheduler = None
            self.use_adaptive_lr = False
            logger.warning("adaptive_lr_scheduler_not_available")
        
        self.normalizer = RunningNormalizer(state_dim)
        
        # Prioritized experience replay (if available)
        try:
            from .prioritized_replay_buffer import PrioritizedReplayBuffer
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=self.config.replay_buffer_size,
                alpha=0.6,  # Priority exponent
                beta=0.4,  # Importance sampling exponent
            )
            self.use_prioritized_replay = True
            logger.info("prioritized_replay_buffer_enabled")
        except ImportError:
            self.replay_buffer = ExperienceReplayBuffer(self.config.replay_buffer_size)
            self.use_prioritized_replay = False
            logger.info("using_standard_replay_buffer")
        self.last_update_metrics: Dict[str, Any] = {}
        self._tail_feature_count = 31  # Updated for dual-mode (23 + 8 new fields)
        self.market_feature_dim = max(1, state_dim - self._tail_feature_count)

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
        market_features = state.market_features
        if market_features.shape[0] != self.market_feature_dim:
            adjusted = np.zeros(self.market_feature_dim, dtype=np.float32)
            length = min(len(market_features), self.market_feature_dim)
            adjusted[:length] = market_features[:length]
            market_features = adjusted

        features = [
            *market_features,
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
            state.recent_return_1m,
            state.recent_return_5m,
            state.recent_return_30m,
            state.volume_zscore,
            state.volatility_zscore,
            state.estimated_transaction_cost_bps / 10.0,
            state.trend_flag_5m,
            state.trend_flag_1h,
            state.orderbook_imbalance,
            state.flow_trend_score,
            # Dual-mode features
            1.0 if state.trading_mode == "long_hold" else 0.0,
            float(state.has_short_position),
            float(state.has_long_position),
            state.short_position_pnl_bps / 100.0,
            state.long_position_pnl_bps / 100.0,
            state.long_position_age_hours / 24.0,  # Normalize to days
            state.num_adds / 5.0,  # Normalize (max ~5 adds)
            float(state.be_lock_active),
            float(state.trail_active),
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
            action_idx = int(torch.argmax(logits).item())
            action_tensor = torch.tensor(action_idx, device=self.device)
        else:
            action_tensor = dist.sample()
            action_idx = int(action_tensor.item())

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

        # Determine which mode we're in
        is_short_mode = state.trading_mode == "short_hold"
        is_long_mode = state.trading_mode == "long_hold"

        # Mask actions that are invalid given position state
        mask = torch.zeros_like(adjusted, dtype=torch.bool)

        if is_short_mode:
            # SHORT-HOLD mode: mask long-hold specific actions
            mask[TradingAction.ADD_GRID.value] = True
            mask[TradingAction.SCALE_OUT.value] = True
            mask[TradingAction.TRAIL_RUNNER.value] = True

            if state.has_short_position:
                # Already have short position, can't enter
                mask[TradingAction.ENTER_LONG_SMALL.value] = True
                mask[TradingAction.ENTER_LONG_NORMAL.value] = True
                mask[TradingAction.ENTER_LONG_LARGE.value] = True
            else:
                # No short position, can't exit/hold/scratch
                mask[TradingAction.EXIT_POSITION.value] = True
                mask[TradingAction.HOLD_POSITION.value] = True
                mask[TradingAction.SCRATCH.value] = True

        elif is_long_mode:
            # LONG-HOLD mode: mask short-hold specific actions
            mask[TradingAction.SCRATCH.value] = True

            if state.has_long_position:
                # Already have long position
                mask[TradingAction.ENTER_LONG_SMALL.value] = True
                mask[TradingAction.ENTER_LONG_NORMAL.value] = True
                mask[TradingAction.ENTER_LONG_LARGE.value] = True

                # Can't add more if already added too many times
                if state.num_adds >= 3:
                    mask[TradingAction.ADD_GRID.value] = True
            else:
                # No long position, can't exit/hold/add/scale/trail
                mask[TradingAction.EXIT_POSITION.value] = True
                mask[TradingAction.HOLD_POSITION.value] = True
                mask[TradingAction.ADD_GRID.value] = True
                mask[TradingAction.SCALE_OUT.value] = True
                mask[TradingAction.TRAIL_RUNNER.value] = True

        adjusted[mask] = -1e9

        # Reliability-aware penalties to curb oversized entries
        reliability = float(state.similar_pattern_reliability)
        reliability = max(0.0, min(reliability, 1.0))
        if reliability < self.config.min_pattern_reliability:
            penalty = (self.config.min_pattern_reliability - reliability) * self.config.large_position_penalty
            adjusted[TradingAction.ENTER_LONG_LARGE.value] -= penalty
        else:
            adjusted[TradingAction.DO_NOTHING.value] -= (1.0 - reliability) * self.config.inaction_penalty

        # Short-hold: encourage SCRATCH when underwater
        if is_short_mode and state.has_short_position and state.short_position_pnl_bps <= -self.config.drawdown_exit_trigger_bps:
            adjusted[TradingAction.SCRATCH.value] += self.config.drawdown_exit_boost

        # Long-hold: encourage SCALE_OUT when in profit, TRAIL_RUNNER when running
        if is_long_mode and state.has_long_position:
            if state.long_position_pnl_bps > 100.0:  # In good profit
                adjusted[TradingAction.SCALE_OUT.value] += 2.0
                adjusted[TradingAction.TRAIL_RUNNER.value] += 1.5
            elif state.long_position_pnl_bps <= -self.config.drawdown_exit_trigger_bps:
                # Deep underwater, consider exit
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
        td_errors = torch.zeros_like(rewards_tensor, device=self.device)  # For prioritized replay
        gae = torch.tensor(0.0, device=self.device)
        for idx in reversed(range(rewards_tensor.shape[0])):
            mask = 1.0 - dones_tensor[idx]
            delta = rewards_tensor[idx] + self.config.gamma * value_targets[idx + 1] * mask - value_targets[idx]
            gae = delta + self.config.gamma * self.config.gae_lambda * mask * gae
            advantages[idx] = gae
            td_errors[idx] = abs(delta)  # TD-error for prioritized replay

        returns_tensor = advantages + value_targets[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages_detached = advantages.detach()
        returns_tensor = returns_tensor.detach()
        td_errors_detached = td_errors.detach()

        # Add latest batch to replay buffer before sampling
        if self.use_prioritized_replay:
            # Prioritized replay buffer
            self.replay_buffer.add_batch(
                states_tensor.detach().cpu(),
                actions_tensor.detach().cpu(),
                old_log_probs_tensor.detach().cpu(),
                advantages_detached.detach().cpu(),
                returns_tensor.cpu(),
                list(self.state_context),
                td_errors=td_errors_detached.detach().cpu(),
            )
        else:
            # Standard replay buffer
            self.replay_buffer.add_batch(
                states_tensor.detach().cpu(),
                actions_tensor.detach().cpu(),
                old_log_probs_tensor.detach().cpu(),
                advantages_detached.detach().cpu(),
                returns_tensor.cpu(),
                list(self.state_context),
            )

        training_state_context = list(self.state_context)
        if len(self.replay_buffer) >= self.config.replay_min_samples:
            replay_sample_size = int(actions_tensor.shape[0] * self.config.replay_sample_ratio)
            
            # Get current regime for regime-weighted sampling
            current_regime = None
            if self.state_context:
                last_state = self.state_context[-1]
                if hasattr(last_state, 'regime_code'):
                    regime_code = last_state.regime_code
                    regime_map = {0: 'trend', 1: 'range', 2: 'panic'}
                    current_regime = regime_map.get(regime_code, 'unknown')
            
            # Sample from replay buffer
            if self.use_prioritized_replay:
                replay_batch = self.replay_buffer.sample(
                    count=replay_sample_size,
                    current_regime=current_regime,
                    use_regime_weighting=True,
                )
            else:
                replay_batch = self.replay_buffer.sample(
                    count=replay_sample_size,
                    current_regime=current_regime,
                    use_regime_weighting=True,
                )
            
            if replay_batch:
                states_tensor = torch.cat([
                    states_tensor,
                    replay_batch["states"].to(self.device),
                ], dim=0)
                actions_tensor = torch.cat([
                    actions_tensor,
                    replay_batch["actions"].to(self.device).long(),
                ], dim=0)
                old_log_probs_tensor = torch.cat([
                    old_log_probs_tensor,
                    replay_batch["log_probs"].to(self.device),
                ], dim=0)
                advantages_detached = torch.cat([
                    advantages_detached,
                    replay_batch["advantages"].to(self.device),
                ], dim=0)
                returns_tensor = torch.cat([
                    returns_tensor,
                    replay_batch["returns"].to(self.device),
                ], dim=0)
                training_state_context.extend(replay_batch["contexts"])
                
                # Apply importance sampling weights if using prioritized replay
                if self.use_prioritized_replay and "importance_weights" in replay_batch:
                    importance_weights = replay_batch["importance_weights"].to(self.device)
                    # Apply weights to advantages and returns
                    if advantages_detached.dim() > 1:
                        advantages_detached = advantages_detached * importance_weights.unsqueeze(1)
                        returns_tensor = returns_tensor * importance_weights.unsqueeze(1)
                    else:
                        advantages_detached = advantages_detached * importance_weights
                        returns_tensor = returns_tensor * importance_weights
        else:
            replay_batch = None

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.config.n_epochs):
            logits_batch, state_values = self.network(states_tensor)
            adjusted_logits = torch.stack(
                [
                    self._apply_contextual_bias(logit_vec, ctx)
                    for logit_vec, ctx in zip(logits_batch, training_state_context)
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
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            
            # Update adaptive learning rate scheduler (only once per epoch loop)
            if _ == self.config.n_epochs - 1 and self.use_adaptive_lr and self.lr_scheduler:
                # Calculate win rate from recent rewards
                recent_rewards = self.rewards[-100:] if len(self.rewards) >= 100 else self.rewards
                win_rate = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards) if recent_rewards else 0.0
                
                # Get current regime from state context
                current_regime = None
                if self.state_context:
                    last_state = self.state_context[-1]
                    if hasattr(last_state, 'regime_code'):
                        regime_code = last_state.regime_code
                        regime_map = {0: 'trend', 1: 'range', 2: 'panic'}
                        current_regime = regime_map.get(regime_code, 'unknown')
                
                # Step scheduler
                self.lr_scheduler.step(win_rate=win_rate, current_regime=current_regime)

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

        action_hist = torch.bincount(actions_tensor, minlength=self.n_actions).float()
        action_prob = action_hist / max(action_hist.sum(), 1.0)
        action_entropy = float(
            -torch.sum(action_prob * torch.log(action_prob + 1e-8)).item()
        )
        avg_advantage = float(advantages_detached.mean().item())
        std_advantage = float(advantages_detached.std().item())

        metrics = {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "action_entropy": action_entropy,
            "avg_advantage": avg_advantage,
            "std_advantage": std_advantage,
            "replay_buffer": len(self.replay_buffer),
        }

        self.last_update_metrics = metrics
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

    def diagnostics(self) -> Dict[str, Any]:
        """Return diagnostics from the latest update cycle."""
        return dict(self.last_update_metrics)

    def ingest_memory_replay(self, samples: List[Dict[str, Any]]) -> None:
        """Add experiences sourced from historical memory into replay buffer."""

        if not samples:
            return

        states_list: List[torch.Tensor] = []
        actions_list: List[int] = []
        log_probs_list: List[float] = []
        advantages_list: List[float] = []
        returns_list: List[float] = []
        contexts: List[TradingState] = []

        for item in samples:
            embedding = np.array(item.get("entry_embedding", []), dtype=np.float32)
            market_features = np.zeros(self.market_feature_dim, dtype=np.float32)
            if embedding.size:
                length = min(self.market_feature_dim, embedding.size)
                market_features[:length] = embedding[:length]

            position_size = float(item.get("position_size_gbp", 1_000.0) or 1_000.0)
            net_profit = float(item.get("net_profit_gbp", 0.0) or 0.0)
            reward_bps = (net_profit / max(position_size, 1e-6)) * 10_000.0

            state = TradingState(
                market_features=market_features,
                similar_pattern_win_rate=float(item.get("win_rate", 0.5) or 0.5),
                similar_pattern_avg_profit=float(item.get("avg_profit_gbp", 0.0) or 0.0),
                similar_pattern_reliability=float(item.get("reliability_score", 0.5) or 0.5),
                has_position=False,
                position_size_multiplier=0.0,
                unrealized_pnl_bps=0.0,
                hold_duration_minutes=int(item.get("hold_duration_minutes", 0) or 0),
                volatility_bps=float(item.get("volatility_bps", 0.0) or 0.0),
                spread_bps=float(item.get("spread_bps", 5.0) or 5.0),
                regime_code=self._regime_code_from_string(item.get("market_regime")),
                current_drawdown_gbp=0.0,
                trades_today=0,
                win_rate_today=float(item.get("win_rate_today", 0.5) or 0.5),
                recent_return_1m=0.0,
                recent_return_5m=0.0,
                recent_return_30m=0.0,
                volume_zscore=0.0,
                volatility_zscore=0.0,
                estimated_transaction_cost_bps=float(item.get("spread_bps", 5.0) or 5.0),
                trend_flag_5m=0.0,
                trend_flag_1h=0.0,
                orderbook_imbalance=float(item.get("orderbook_imbalance", 0.0) or 0.0),
                flow_trend_score=float(item.get("flow_trend_score", 0.0) or 0.0),
                symbol=item.get("symbol"),
            )

            state_tensor = torch.from_numpy(self.state_to_tensor(state))
            self.normalizer.normalize(state_tensor, update_stats=True)

            states_list.append(state_tensor)
            actions_list.append(TradingAction.ENTER_LONG_NORMAL.value)
            log_probs_list.append(0.0)
            advantages_list.append(reward_bps / 100.0)
            returns_list.append(reward_bps / 100.0)
            contexts.append(state)

        if not states_list:
            return

        states_tensor = torch.stack(states_list)
        actions_tensor = torch.tensor(actions_list, dtype=torch.long)
        log_probs_tensor = torch.tensor(log_probs_list, dtype=torch.float32)
        advantages_tensor = torch.tensor(advantages_list, dtype=torch.float32)
        returns_tensor = torch.tensor(returns_list, dtype=torch.float32)

        self.replay_buffer.add_batch(
            states_tensor,
            actions_tensor,
            log_probs_tensor,
            advantages_tensor,
            returns_tensor,
            contexts,
        )

    @staticmethod
    def _regime_code_from_string(regime: Optional[str]) -> int:
        mapping = {
            "trend": 3,
            "range": 4,
            "panic": 2,
            "unknown": 1,
            None: 1,
        }
        return mapping.get(str(regime).lower() if regime else None, 1)

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

            if state.estimated_transaction_cost_bps > 0:
                reward -= state.estimated_transaction_cost_bps * self.config.cost_penalty_scale

        if action == TradingAction.ENTER_LONG_LARGE and state is not None:
            reliability = float(state.similar_pattern_reliability)
            if reliability < self.config.min_pattern_reliability:
                reward -= self.config.large_position_penalty * 5.0

        return float(reward)
