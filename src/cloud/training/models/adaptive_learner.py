"""
Adaptive Learning Rate - Phase 3

Dynamically adjusts learning rates based on performance and market conditions:
1. Increases learning rate when performance is good (exploit)
2. Decreases learning rate when performance is poor (explore)
3. Adjusts based on regime changes
4. Implements momentum and decay

Based on adaptive optimization algorithms like Adam, RMSprop.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class LearningRateState:
    """State of adaptive learning rate."""

    current_rate: float
    base_rate: float
    momentum: float
    recent_performance: List[float]
    recent_regime: str
    adaptation_count: int


class AdaptiveLearner:
    """
    Adaptive learning rate manager.

    Automatically adjusts learning rates based on:
    1. Recent performance (win rate, profit)
    2. Performance stability (variance)
    3. Market regime changes
    4. Time since last update
    """

    def __init__(
        self,
        base_learning_rate: float = 0.05,
        min_learning_rate: float = 0.01,
        max_learning_rate: float = 0.20,
        momentum: float = 0.9,
        performance_window: int = 20,
    ):
        """
        Initialize adaptive learner.

        Args:
            base_learning_rate: Starting learning rate
            min_learning_rate: Minimum allowed learning rate
            max_learning_rate: Maximum allowed learning rate
            momentum: Momentum factor (0-1)
            performance_window: Window for performance evaluation
        """
        self.base_rate = base_learning_rate
        self.min_rate = min_learning_rate
        self.max_rate = max_learning_rate
        self.momentum_factor = momentum
        self.performance_window = performance_window

        # State tracking
        self.current_rate = base_learning_rate
        self.momentum_term = 0.0
        self.recent_performance: List[float] = []
        self.recent_regime = "unknown"
        self.adaptation_count = 0

        # Performance statistics
        self.total_adaptations = 0
        self.rate_increases = 0
        self.rate_decreases = 0

        logger.info(
            "adaptive_learner_initialized",
            base_rate=base_learning_rate,
            min_rate=min_learning_rate,
            max_rate=max_learning_rate,
        )

    def adapt(
        self,
        performance_signal: float,
        current_regime: str,
        regime_changed: bool = False,
    ) -> float:
        """
        Adapt learning rate based on performance and regime.

        Args:
            performance_signal: Recent performance (0-1, where 1 = perfect)
            current_regime: Current market regime
            regime_changed: Whether regime just changed

        Returns:
            New learning rate
        """
        # Store performance
        self.recent_performance.append(performance_signal)
        if len(self.recent_performance) > self.performance_window:
            self.recent_performance = self.recent_performance[-self.performance_window:]

        # If regime changed, reset momentum and be conservative
        if regime_changed and current_regime != self.recent_regime:
            self.momentum_term = 0.0
            self.current_rate = self.base_rate * 0.5  # Halve rate on regime change
            self.recent_regime = current_regime

            logger.info(
                "learning_rate_reset_regime_change",
                new_regime=current_regime,
                new_rate=self.current_rate,
            )

            return self.current_rate

        # Calculate performance statistics
        if len(self.recent_performance) < 5:
            # Not enough data yet
            return self.current_rate

        avg_performance = np.mean(self.recent_performance)
        performance_stability = 1.0 - np.std(self.recent_performance)  # Higher = more stable

        # Determine adaptation direction
        # Good performance + stable = increase rate (exploit)
        # Poor performance = decrease rate (explore more)
        if avg_performance > 0.65 and performance_stability > 0.7:
            # Doing well and stable → increase learning rate (exploit)
            adjustment = 1.1
            self.rate_increases += 1
        elif avg_performance > 0.55:
            # Doing okay → maintain
            adjustment = 1.0
        else:
            # Doing poorly → decrease learning rate (explore)
            adjustment = 0.9
            self.rate_decreases += 1

        # Apply momentum
        gradient = (adjustment - 1.0) * self.base_rate
        self.momentum_term = self.momentum_factor * self.momentum_term + (1 - self.momentum_factor) * gradient

        # Update rate with momentum
        new_rate = self.current_rate + self.momentum_term

        # Clip to bounds
        new_rate = np.clip(new_rate, self.min_rate, self.max_rate)

        # Track
        if new_rate != self.current_rate:
            direction = "increased" if new_rate > self.current_rate else "decreased"
            logger.debug(
                "learning_rate_adapted",
                old_rate=self.current_rate,
                new_rate=new_rate,
                direction=direction,
                avg_performance=avg_performance,
                stability=performance_stability,
            )

        self.current_rate = float(new_rate)
        self.adaptation_count += 1
        self.total_adaptations += 1

        return self.current_rate

    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.current_rate

    def reset(self) -> None:
        """Reset to base learning rate."""
        self.current_rate = self.base_rate
        self.momentum_term = 0.0
        self.recent_performance.clear()

        logger.info("learning_rate_reset", base_rate=self.base_rate)

    def get_stats(self) -> Dict[str, float]:
        """Get learning rate statistics."""
        if self.recent_performance:
            avg_performance = np.mean(self.recent_performance)
            performance_trend = (
                self.recent_performance[-1] - self.recent_performance[0]
            ) if len(self.recent_performance) > 1 else 0.0
        else:
            avg_performance = 0.5
            performance_trend = 0.0

        return {
            "current_rate": self.current_rate,
            "base_rate": self.base_rate,
            "momentum": self.momentum_term,
            "avg_performance": avg_performance,
            "performance_trend": performance_trend,
            "total_adaptations": self.total_adaptations,
            "rate_increases": self.rate_increases,
            "rate_decreases": self.rate_decreases,
        }

    def get_state(self) -> Dict:
        """Get state for persistence."""
        return {
            "current_rate": self.current_rate,
            "momentum_term": self.momentum_term,
            "recent_performance": list(self.recent_performance),
            "recent_regime": self.recent_regime,
            "adaptation_count": self.adaptation_count,
            "total_adaptations": self.total_adaptations,
            "rate_increases": self.rate_increases,
            "rate_decreases": self.rate_decreases,
        }

    def load_state(self, state: Dict) -> None:
        """Load state from persistence."""
        self.current_rate = state.get("current_rate", self.base_rate)
        self.momentum_term = state.get("momentum_term", 0.0)
        self.recent_performance = list(state.get("recent_performance", []))
        self.recent_regime = state.get("recent_regime", "unknown")
        self.adaptation_count = state.get("adaptation_count", 0)
        self.total_adaptations = state.get("total_adaptations", 0)
        self.rate_increases = state.get("rate_increases", 0)
        self.rate_decreases = state.get("rate_decreases", 0)

        logger.info(
            "adaptive_learner_state_loaded",
            current_rate=self.current_rate,
            total_adaptations=self.total_adaptations,
        )


class MultiTimeframeLearner:
    """
    Manages learning rates across multiple timeframes.

    Different timeframes may need different learning rates:
    - Short-term (1min-15min): Higher learning rate, faster adaptation
    - Medium-term (1h-4h): Moderate learning rate
    - Long-term (1d+): Lower learning rate, more stable
    """

    def __init__(self):
        """Initialize multi-timeframe learner."""
        self.learners = {
            "short_term": AdaptiveLearner(base_learning_rate=0.10, min_learning_rate=0.05, max_learning_rate=0.25),
            "medium_term": AdaptiveLearner(base_learning_rate=0.05, min_learning_rate=0.02, max_learning_rate=0.15),
            "long_term": AdaptiveLearner(base_learning_rate=0.02, min_learning_rate=0.01, max_learning_rate=0.08),
        }

        logger.info("multi_timeframe_learner_initialized", timeframes=list(self.learners.keys()))

    def adapt(self, timeframe: str, performance_signal: float, current_regime: str, regime_changed: bool = False) -> float:
        """Adapt learning rate for specific timeframe."""
        if timeframe not in self.learners:
            logger.warning("unknown_timeframe", timeframe=timeframe)
            return 0.05  # Default

        return self.learners[timeframe].adapt(performance_signal, current_regime, regime_changed)

    def get_learning_rate(self, timeframe: str) -> float:
        """Get learning rate for specific timeframe."""
        if timeframe not in self.learners:
            return 0.05

        return self.learners[timeframe].get_learning_rate()

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get stats for all timeframes."""
        return {
            timeframe: learner.get_stats()
            for timeframe, learner in self.learners.items()
        }

    def get_state(self) -> Dict:
        """Get state for all timeframes."""
        return {
            timeframe: learner.get_state()
            for timeframe, learner in self.learners.items()
        }

    def load_state(self, state: Dict) -> None:
        """Load state for all timeframes."""
        for timeframe, learner_state in state.items():
            if timeframe in self.learners:
                self.learners[timeframe].load_state(learner_state)
