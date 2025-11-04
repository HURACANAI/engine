"""
Multi-Agent Ensemble System

Implements an ensemble of specialized agents for different market conditions:
- Bull Agent: Optimized for trending up markets (RISK_ON)
- Bear Agent: Optimized for trending down markets (RISK_OFF)
- Sideways Agent: Optimized for ranging markets (ROTATION)
- Volatility Agent: Optimized for high-volatility environments (PANIC)

Key benefits:
1. Specialization: Each agent masters its domain
2. Robustness: Ensemble reduces overfit to single regime
3. Adaptability: Automatic regime detection → agent selection
4. Performance: Specialist agents outperform generalist in their regime

Architecture:
- Train 4 specialized agents on regime-filtered data
- Meta-agent learns to weight specialists based on current regime
- Ensemble decision = weighted combination of specialist predictions
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import structlog

from .rl_agent import RLTradingAgent, TradingState, TradingAction, PPOConfig
from ..models.market_structure import MarketRegime

logger = structlog.get_logger(__name__)


class AgentRole(Enum):
    """Specialized agent roles."""

    BULL = "bull"  # Trend-following in uptrends
    BEAR = "bear"  # Trend-following in downtrends
    SIDEWAYS = "sideways"  # Mean-reversion in ranges
    VOLATILITY = "volatility"  # Breakout trading in volatility


@dataclass
class EnsembleConfig:
    """Ensemble hyperparameters."""

    num_specialists: int = 4  # Number of specialized agents
    meta_learning_rate: float = 0.001  # LR for meta-agent
    temperature: float = 1.0  # Softmax temperature for mixing
    min_regime_confidence: float = 0.6  # Minimum confidence to use specialist
    enable_dynamic_weighting: bool = True  # Learn weights vs fixed
    ensemble_mode: str = "weighted"  # "weighted", "voting", or "best"


class SpecialistAgent:
    """
    Specialized agent for a specific market regime.

    Each specialist is trained only on data from its target regime,
    making it an expert in that market condition.
    """

    def __init__(
        self,
        role: AgentRole,
        target_regime: MarketRegime,
        state_dim: int,
        config: PPOConfig,
        device: str = "cpu",
    ):
        """
        Initialize specialist agent.

        Args:
            role: Agent's specialization
            target_regime: Market regime this agent specializes in
            state_dim: State dimension
            config: PPO configuration
            device: Device for computation
        """
        self.role = role
        self.target_regime = target_regime
        self.state_dim = state_dim
        self.device = device

        # Create underlying RL agent
        self.agent = RLTradingAgent(
            state_dim=state_dim,
            memory_store=None,  # Specialists share memory
            config=config,
            device=device,
        )

        # Track specialist performance
        self.regime_accuracy = 0.0  # How often correct in its regime
        self.regime_sharpe = 0.0  # Sharpe ratio in its regime
        self.total_trades = 0

        logger.info(
            "specialist_agent_initialized",
            role=role.value,
            target_regime=target_regime.value,
        )

    def select_action(
        self,
        state: TradingState,
        deterministic: bool = False,
    ) -> Tuple[TradingAction, float, Dict]:
        """Select action for given state."""
        return self.agent.select_action(state, deterministic)

    def update_performance(self, win_rate: float, sharpe: float, trades: int) -> None:
        """Update specialist performance metrics."""
        self.regime_accuracy = win_rate
        self.regime_sharpe = sharpe
        self.total_trades = trades

    def get_stats(self) -> Dict[str, float]:
        """Get specialist statistics."""
        return {
            "role": self.role.value,
            "target_regime": self.target_regime.value,
            "accuracy": self.regime_accuracy,
            "sharpe": self.regime_sharpe,
            "total_trades": self.total_trades,
        }


class MetaAgent(nn.Module):
    """
    Meta-agent that learns to weight specialist agents.

    Input: Current market state + regime indicators
    Output: Weights for each specialist (sums to 1.0)

    This learns which specialist to trust in different market conditions.
    """

    def __init__(
        self,
        state_dim: int,
        num_specialists: int,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.num_specialists = num_specialists

        # Network that outputs specialist weights
        self.weight_network = nn.Sequential(
            nn.Linear(state_dim + 10, hidden_dim),  # +10 for regime features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_specialists),
        )

    def forward(
        self,
        state_features: torch.Tensor,
        regime_features: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute weights for specialists.

        Args:
            state_features: Market state features
            regime_features: Regime indicators (volatility, trend, etc.)
            temperature: Softmax temperature (higher = more uniform)

        Returns:
            Specialist weights (sums to 1.0)
        """
        # Concatenate state and regime features
        combined = torch.cat([state_features, regime_features], dim=-1)

        # Compute logits
        logits = self.weight_network(combined)

        # Convert to probabilities
        weights = torch.softmax(logits / temperature, dim=-1)

        return weights


class AgentEnsemble:
    """
    Ensemble of specialized agents with meta-learning.

    Manages multiple specialist agents and learns to combine them optimally.
    """

    def __init__(
        self,
        state_dim: int,
        base_config: PPOConfig,
        config: EnsembleConfig,
        device: str = "cpu",
    ):
        """
        Initialize agent ensemble.

        Args:
            state_dim: State dimension
            base_config: Base PPO configuration for specialists
            config: Ensemble configuration
            device: Device for computation
        """
        self.state_dim = state_dim
        self.config = config
        self.device = device

        # Create specialist agents
        self.specialists: Dict[AgentRole, SpecialistAgent] = {}

        # Bull agent (RISK_ON, uptrends)
        self.specialists[AgentRole.BULL] = SpecialistAgent(
            role=AgentRole.BULL,
            target_regime=MarketRegime.RISK_ON,
            state_dim=state_dim,
            config=base_config,
            device=device,
        )

        # Bear agent (RISK_OFF, downtrends)
        self.specialists[AgentRole.BEAR] = SpecialistAgent(
            role=AgentRole.BEAR,
            target_regime=MarketRegime.RISK_OFF,
            state_dim=state_dim,
            config=base_config,
            device=device,
        )

        # Sideways agent (ROTATION, ranges)
        self.specialists[AgentRole.SIDEWAYS] = SpecialistAgent(
            role=AgentRole.SIDEWAYS,
            target_regime=MarketRegime.ROTATION,
            state_dim=state_dim,
            config=base_config,
            device=device,
        )

        # Volatility agent (PANIC, high volatility)
        self.specialists[AgentRole.VOLATILITY] = SpecialistAgent(
            role=AgentRole.VOLATILITY,
            target_regime=MarketRegime.PANIC,
            state_dim=state_dim,
            config=base_config,
            device=device,
        )

        # Meta-agent for dynamic weighting
        if config.enable_dynamic_weighting:
            self.meta_agent = MetaAgent(
                state_dim=state_dim,
                num_specialists=config.num_specialists,
            ).to(device)

            self.meta_optimizer = torch.optim.Adam(
                self.meta_agent.parameters(),
                lr=config.meta_learning_rate,
            )
        else:
            self.meta_agent = None

        logger.info(
            "agent_ensemble_initialized",
            num_specialists=len(self.specialists),
            dynamic_weighting=config.enable_dynamic_weighting,
            ensemble_mode=config.ensemble_mode,
        )

    def select_action(
        self,
        state: TradingState,
        current_regime: Optional[MarketRegime] = None,
        regime_confidence: float = 1.0,
        deterministic: bool = False,
    ) -> Tuple[TradingAction, float, Dict]:
        """
        Select action using ensemble of specialists.

        Args:
            state: Current trading state
            current_regime: Detected market regime
            regime_confidence: Confidence in regime detection
            deterministic: Whether to use deterministic actions

        Returns:
            (action, confidence, metadata)
        """
        # Get specialist predictions
        specialist_actions = {}
        specialist_confidences = {}

        for role, specialist in self.specialists.items():
            action, confidence, _ = specialist.select_action(state, deterministic)
            specialist_actions[role] = action
            specialist_confidences[role] = confidence

        # Determine specialist weights
        if self.config.enable_dynamic_weighting and self.meta_agent is not None:
            weights = self._get_dynamic_weights(state)
        else:
            weights = self._get_fixed_weights(current_regime, regime_confidence)

        # Combine specialist predictions
        if self.config.ensemble_mode == "weighted":
            final_action, final_confidence = self._weighted_combination(
                specialist_actions,
                specialist_confidences,
                weights,
            )
        elif self.config.ensemble_mode == "voting":
            final_action, final_confidence = self._voting_combination(
                specialist_actions,
                specialist_confidences,
                weights,
            )
        else:  # "best"
            final_action, final_confidence = self._best_specialist(
                specialist_actions,
                specialist_confidences,
                weights,
            )

        metadata = {
            "specialist_actions": {role.value: action.value for role, action in specialist_actions.items()},
            "specialist_weights": {role.value: w for role, w in zip(self.specialists.keys(), weights)},
            "ensemble_mode": self.config.ensemble_mode,
        }

        return final_action, final_confidence, metadata

    def _get_dynamic_weights(self, state: TradingState) -> List[float]:
        """Compute weights using meta-agent."""
        # Extract state features
        state_features = self._state_to_tensor(state)

        # Extract regime features (volatility, trend, momentum, etc.)
        regime_features = self._extract_regime_features(state)

        # Get weights from meta-agent
        with torch.no_grad():
            weights = self.meta_agent(
                state_features,
                regime_features,
                temperature=self.config.temperature,
            )

        return weights.cpu().numpy().tolist()

    def _get_fixed_weights(
        self,
        current_regime: Optional[MarketRegime],
        regime_confidence: float,
    ) -> List[float]:
        """Compute fixed weights based on regime."""
        if current_regime is None or regime_confidence < self.config.min_regime_confidence:
            # Uniform weights if regime unclear
            return [1.0 / self.config.num_specialists] * self.config.num_specialists

        # Map regime to specialist
        regime_to_role = {
            MarketRegime.RISK_ON: AgentRole.BULL,
            MarketRegime.RISK_OFF: AgentRole.BEAR,
            MarketRegime.ROTATION: AgentRole.SIDEWAYS,
            MarketRegime.PANIC: AgentRole.VOLATILITY,
        }

        target_role = regime_to_role.get(current_regime, AgentRole.BULL)

        # Assign high weight to specialist, low to others
        weights = []
        for role in self.specialists.keys():
            if role == target_role:
                weights.append(regime_confidence)
            else:
                weights.append((1.0 - regime_confidence) / (self.config.num_specialists - 1))

        return weights

    def _weighted_combination(
        self,
        actions: Dict[AgentRole, TradingAction],
        confidences: Dict[AgentRole, float],
        weights: List[float],
    ) -> Tuple[TradingAction, float]:
        """Combine using weighted average."""
        # Convert actions to scores
        action_scores = {
            TradingAction.LONG: 0.0,
            TradingAction.SHORT: 0.0,
            TradingAction.HOLD: 0.0,
        }

        for (role, action), weight in zip(actions.items(), weights):
            confidence = confidences[role]
            action_scores[action] += weight * confidence

        # Select highest scoring action
        best_action = max(action_scores.items(), key=lambda x: x[1])
        return best_action[0], best_action[1]

    def _voting_combination(
        self,
        actions: Dict[AgentRole, TradingAction],
        confidences: Dict[AgentRole, float],
        weights: List[float],
    ) -> Tuple[TradingAction, float]:
        """Combine using weighted voting."""
        votes = {
            TradingAction.LONG: 0.0,
            TradingAction.SHORT: 0.0,
            TradingAction.HOLD: 0.0,
        }

        for (role, action), weight in zip(actions.items(), weights):
            votes[action] += weight

        # Select most voted action
        best_action = max(votes.items(), key=lambda x: x[1])
        avg_confidence = sum(confidences.values()) / len(confidences)
        return best_action[0], avg_confidence

    def _best_specialist(
        self,
        actions: Dict[AgentRole, TradingAction],
        confidences: Dict[AgentRole, float],
        weights: List[float],
    ) -> Tuple[TradingAction, float]:
        """Select action from best specialist."""
        # Find specialist with highest weight
        best_idx = weights.index(max(weights))
        best_role = list(self.specialists.keys())[best_idx]

        return actions[best_role], confidences[best_role]

    def _state_to_tensor(self, state: TradingState) -> torch.Tensor:
        """Convert state to tensor."""
        # Simplified - would extract full feature vector
        features = []
        if hasattr(state, "features"):
            features = state.features
        return torch.tensor(features, dtype=torch.float32).to(self.device)

    def _extract_regime_features(self, state: TradingState) -> torch.Tensor:
        """Extract regime-related features from state."""
        # Extract features that indicate regime
        regime_features = [
            0.0,  # volatility
            0.0,  # trend_strength
            0.0,  # momentum
            0.0,  # volume_trend
            0.0,  # correlation
            0.0,  # rsi
            0.0,  # atr_zscore
            0.0,  # fear_index
            0.0,  # spread
            0.0,  # liquidity
        ]

        return torch.tensor(regime_features, dtype=torch.float32).to(self.device)

    def get_ensemble_stats(self) -> Dict[str, any]:
        """Get ensemble statistics."""
        stats = {
            "num_specialists": len(self.specialists),
            "ensemble_mode": self.config.ensemble_mode,
            "specialists": {},
        }

        for role, specialist in self.specialists.items():
            stats["specialists"][role.value] = specialist.get_stats()

        return stats

    def save(self, path: str) -> None:
        """Save ensemble."""
        checkpoint = {
            "config": self.config,
            "specialists": {
                role.value: specialist.agent.policy.state_dict()
                for role, specialist in self.specialists.items()
            },
        }

        if self.meta_agent is not None:
            checkpoint["meta_agent"] = self.meta_agent.state_dict()
            checkpoint["meta_optimizer"] = self.meta_optimizer.state_dict()

        torch.save(checkpoint, path)
        logger.info("ensemble_saved", path=path)

    def load(self, path: str) -> None:
        """Load ensemble."""
        checkpoint = torch.load(path, map_location=self.device)

        for role, specialist in self.specialists.items():
            specialist.agent.policy.load_state_dict(checkpoint["specialists"][role.value])

        if self.meta_agent is not None and "meta_agent" in checkpoint:
            self.meta_agent.load_state_dict(checkpoint["meta_agent"])
            self.meta_optimizer.load_state_dict(checkpoint["meta_optimizer"])

        logger.info("ensemble_loaded", path=path)
