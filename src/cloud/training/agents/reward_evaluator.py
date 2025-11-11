"""
Regime-Aware Reward Evaluator

From reinforcement learning: trial and reward concepts.
After every forecast, simulate if trade was right or wrong.
Assign reward signal (+1, 0, -1) and store it.
Mechanic can train reinforcement model on that reward data later.

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RewardSignal:
    """Reward signal from trade simulation."""
    timestamp: datetime
    symbol: str
    prediction: float
    actual: float
    reward: float  # +1, 0, or -1
    regime: str
    confidence: float
    metadata: Dict[str, any]


class RegimeAwareRewardEvaluator:
    """
    Regime-aware reward evaluator for simulated trades.
    
    Usage:
        evaluator = RegimeAwareRewardEvaluator()
        
        # Evaluate forecast
        reward = evaluator.evaluate(
            prediction=0.05,  # Predicted return
            actual=0.03,      # Actual return
            regime="trending",
            confidence=0.8
        )
        
        # Store reward for RL training
        evaluator.store_reward(reward, symbol="BTC/USDT")
    """
    
    def __init__(
        self,
        reward_threshold: float = 0.01,  # 1% threshold for reward
        directional_reward: bool = True
    ):
        """
        Initialize reward evaluator.
        
        Args:
            reward_threshold: Minimum prediction error to assign reward
            directional_reward: Whether to reward correct direction (default: True)
        """
        self.reward_threshold = reward_threshold
        self.directional_reward = directional_reward
        self.reward_history: List[RewardSignal] = []
        
        logger.info(
            "reward_evaluator_initialized",
            reward_threshold=reward_threshold,
            directional_reward=directional_reward
        )
    
    def evaluate(
        self,
        prediction: float,
        actual: float,
        regime: str,
        confidence: float = 1.0,
        symbol: str = "unknown"
    ) -> RewardSignal:
        """
        Evaluate forecast and assign reward signal.
        
        Args:
            prediction: Predicted return/value
            actual: Actual return/value
            regime: Market regime
            confidence: Prediction confidence (0-1)
            symbol: Trading symbol
        
        Returns:
            RewardSignal with reward assignment
        """
        # Calculate error
        error = abs(prediction - actual)
        error_pct = error / (abs(actual) + 1e-8)
        
        # Determine reward based on regime
        if regime.lower() in ["trending", "volatile"]:
            reward = self._evaluate_trending_regime(
                prediction, actual, error, error_pct, confidence
            )
        elif regime.lower() in ["ranging", "mixed"]:
            reward = self._evaluate_ranging_regime(
                prediction, actual, error, error_pct, confidence
            )
        else:
            reward = self._evaluate_default(prediction, actual, error, error_pct)
        
        signal = RewardSignal(
            timestamp=datetime.now(),
            symbol=symbol,
            prediction=prediction,
            actual=actual,
            reward=reward,
            regime=regime,
            confidence=confidence,
            metadata={
                "error": error,
                "error_pct": error_pct,
                "directional_correct": self._check_direction(prediction, actual)
            }
        )
        
        self.reward_history.append(signal)
        
        logger.debug(
            "reward_evaluated",
            symbol=symbol,
            reward=reward,
            error_pct=error_pct,
            regime=regime
        )
        
        return signal
    
    def _evaluate_trending_regime(
        self,
        prediction: float,
        actual: float,
        error: float,
        error_pct: float,
        confidence: float
    ) -> float:
        """Evaluate reward for trending regime (focus on direction)."""
        # In trending markets, direction matters more than magnitude
        direction_correct = self._check_direction(prediction, actual)
        
        if direction_correct:
            # Reward for correct direction, scaled by confidence
            if error_pct < self.reward_threshold:
                return 1.0 * confidence  # Strong reward
            elif error_pct < self.reward_threshold * 2:
                return 0.5 * confidence  # Moderate reward
            else:
                return 0.0  # Correct direction but poor magnitude
        else:
            # Penalize wrong direction
            if error_pct > self.reward_threshold * 2:
                return -1.0  # Strong penalty
            else:
                return -0.5  # Moderate penalty
    
    def _evaluate_ranging_regime(
        self,
        prediction: float,
        actual: float,
        error: float,
        error_pct: float,
        confidence: float
    ) -> float:
        """Evaluate reward for ranging regime (focus on magnitude)."""
        # In ranging markets, magnitude matters more
        if error_pct < self.reward_threshold:
            return 1.0 * confidence  # Strong reward
        elif error_pct < self.reward_threshold * 2:
            return 0.5 * confidence  # Moderate reward
        elif error_pct < self.reward_threshold * 3:
            return 0.0  # Neutral
        else:
            return -0.5  # Penalty for large error
    
    def _evaluate_default(
        self,
        prediction: float,
        actual: float,
        error: float,
        error_pct: float
    ) -> float:
        """Default reward evaluation."""
        if error_pct < self.reward_threshold:
            return 1.0
        elif error_pct < self.reward_threshold * 2:
            return 0.0
        else:
            return -1.0
    
    def _check_direction(self, prediction: float, actual: float) -> bool:
        """Check if prediction direction matches actual."""
        return (prediction > 0) == (actual > 0)
    
    def store_reward(self, reward: RewardSignal) -> None:
        """Store reward signal (for later RL training)."""
        self.reward_history.append(reward)
        
        # Keep only last N rewards
        max_history = 10000
        if len(self.reward_history) > max_history:
            self.reward_history = self.reward_history[-max_history:]
        
        logger.debug("reward_stored", reward=reward.reward, symbol=reward.symbol)
    
    def get_reward_statistics(self, regime: Optional[str] = None) -> Dict[str, float]:
        """Get reward statistics."""
        rewards = self.reward_history
        if regime:
            rewards = [r for r in rewards if r.regime == regime]
        
        if not rewards:
            return {
                "mean_reward": 0.0,
                "total_rewards": 0,
                "positive_rewards": 0,
                "negative_rewards": 0
            }
        
        reward_values = [r.reward for r in rewards]
        
        return {
            "mean_reward": float(np.mean(reward_values)),
            "std_reward": float(np.std(reward_values)),
            "total_rewards": len(rewards),
            "positive_rewards": sum(1 for r in reward_values if r > 0),
            "negative_rewards": sum(1 for r in reward_values if r < 0),
            "neutral_rewards": sum(1 for r in reward_values if r == 0)
        }
    
    def get_rewards_for_training(self, limit: int = 1000) -> List[RewardSignal]:
        """Get recent rewards for RL training."""
        return self.reward_history[-limit:]

