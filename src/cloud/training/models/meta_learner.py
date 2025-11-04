"""
Meta-Learning Controller - Phase 4

The system that learns how to learn better!

Meta-learning capabilities:
1. Learns which hyperparameters work best in which regimes
2. Auto-tunes all system parameters
3. Discovers optimal feature combinations
4. Self-diagnoses performance issues
5. Adapts entire system strategy based on meta-performance

This is the "brain" that controls all other learning systems.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class HyperparameterConfig:
    """Configuration for a set of hyperparameters."""

    config_id: str
    parameters: Dict[str, float]
    performance_history: List[float] = field(default_factory=list)
    regimes_tested: List[str] = field(default_factory=list)
    avg_performance: float = 0.5
    sample_count: int = 0


@dataclass
class MetaPerformanceMetrics:
    """Meta-level performance metrics."""

    overall_win_rate: float
    win_rate_by_regime: Dict[str, float]
    sharpe_ratio: float
    max_drawdown: float
    learning_efficiency: float  # How fast system improves
    adaptation_speed: float  # How fast system adapts to regime changes
    feature_stability: float  # How stable feature importance is
    hyperparameter_optimality: float  # How close to optimal hyperparameters


class MetaLearner:
    """
    Meta-learning controller that learns how the system should learn.

    This is higher-order learning - learning about learning itself.

    Key responsibilities:
    1. Track performance of different hyperparameter configurations
    2. Identify best hyperparameters for each regime
    3. Auto-tune the entire system
    4. Detect when system is stuck and trigger exploration
    5. Learn meta-patterns across all trading sessions
    """

    def __init__(
        self,
        exploration_rate: float = 0.1,  # 10% of time try new configs
        meta_learning_rate: float = 0.01,  # Very slow, careful meta-learning
    ):
        """
        Initialize meta-learner.

        Args:
            exploration_rate: Fraction of time to explore new configurations
            meta_learning_rate: Learning rate for meta-parameters
        """
        self.exploration_rate = exploration_rate
        self.meta_learning_rate = meta_learning_rate

        # Library of hyperparameter configurations
        self.configs: Dict[str, HyperparameterConfig] = {}

        # Current active configuration
        self.current_config_id: Optional[str] = None

        # Meta-performance tracking
        self.session_performances: List[float] = []
        self.regime_performances: Dict[str, List[float]] = {}

        # Meta-patterns discovered
        self.meta_patterns: Dict[str, any] = {
            "best_regime_configs": {},  # Best config for each regime
            "feature_regime_mapping": {},  # Which features work in which regimes
            "optimal_learning_rates": {},  # Optimal rates by regime
            "confidence_thresholds": {},  # Optimal thresholds by regime
        }

        # Self-diagnostics
        self.performance_degradation_detected = False
        self.stuck_counter = 0

        logger.info(
            "meta_learner_initialized",
            exploration_rate=exploration_rate,
            meta_learning_rate=meta_learning_rate,
        )

    def register_config(
        self, config_id: str, parameters: Dict[str, float]
    ) -> None:
        """
        Register a new hyperparameter configuration.

        Args:
            config_id: Unique identifier
            parameters: Dictionary of hyperparameters
        """
        if config_id not in self.configs:
            self.configs[config_id] = HyperparameterConfig(
                config_id=config_id,
                parameters=parameters,
            )
            logger.info("config_registered", config_id=config_id, num_configs=len(self.configs))

    def select_config(
        self, current_regime: str, force_exploration: bool = False
    ) -> Tuple[str, Dict[str, float]]:
        """
        Select which hyperparameter configuration to use.

        Uses epsilon-greedy: exploit best known config vs explore new ones.

        Args:
            current_regime: Current market regime
            force_exploration: Force exploration regardless of epsilon

        Returns:
            (config_id, parameters) tuple
        """
        if not self.configs:
            raise ValueError("No configurations registered")

        # Decide: explore or exploit?
        should_explore = (
            force_exploration
            or np.random.random() < self.exploration_rate
            or self.stuck_counter > 10  # Stuck, force exploration
        )

        if should_explore:
            # Exploration: Try random config or create new one
            config_id = np.random.choice(list(self.configs.keys()))
            logger.debug("meta_exploration", config_id=config_id, regime=current_regime)

        else:
            # Exploitation: Use best config for this regime
            best_config = self._get_best_config_for_regime(current_regime)
            config_id = best_config.config_id if best_config else list(self.configs.keys())[0]
            logger.debug("meta_exploitation", config_id=config_id, regime=current_regime)

        self.current_config_id = config_id
        return (config_id, self.configs[config_id].parameters.copy())

    def _get_best_config_for_regime(
        self, regime: str
    ) -> Optional[HyperparameterConfig]:
        """Get best performing config for a specific regime."""
        # Filter configs that have been tested in this regime
        regime_configs = [
            config
            for config in self.configs.values()
            if regime in config.regimes_tested and config.sample_count > 5
        ]

        if not regime_configs:
            return None

        # Return config with highest average performance in this regime
        best_config = max(regime_configs, key=lambda c: c.avg_performance)
        return best_config

    def update_config_performance(
        self,
        config_id: str,
        performance: float,
        regime: str,
    ) -> None:
        """
        Update performance of a configuration.

        Args:
            config_id: Configuration identifier
            performance: Performance metric (0-1, where 1 = perfect)
            regime: Regime during this performance
        """
        if config_id not in self.configs:
            logger.warning("unknown_config", config_id=config_id)
            return

        config = self.configs[config_id]

        # Update performance history
        config.performance_history.append(performance)
        if len(config.performance_history) > 100:
            config.performance_history = config.performance_history[-100:]

        # Track regime
        if regime not in config.regimes_tested:
            config.regimes_tested.append(regime)

        # Update average with EMA
        if config.sample_count == 0:
            config.avg_performance = performance
        else:
            alpha = 0.1  # Fast adaptation to new performance
            config.avg_performance = (
                1 - alpha
            ) * config.avg_performance + alpha * performance

        config.sample_count += 1

        # Track regime performance globally
        if regime not in self.regime_performances:
            self.regime_performances[regime] = []
        self.regime_performances[regime].append(performance)

        # Check if stuck (performance not improving)
        if len(config.performance_history) >= 20:
            recent = config.performance_history[-20:]
            if np.std(recent) < 0.05 and np.mean(recent) < 0.55:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0

        logger.debug(
            "config_performance_updated",
            config_id=config_id,
            performance=performance,
            avg_performance=config.avg_performance,
            regime=regime,
        )

    def learn_meta_patterns(self) -> None:
        """
        Learn meta-patterns from all configurations and performances.

        This is where meta-learning happens - learning about learning.
        """
        if not self.configs:
            return

        # 1. Learn best config for each regime
        for regime in set(
            regime
            for config in self.configs.values()
            for regime in config.regimes_tested
        ):
            best_config = self._get_best_config_for_regime(regime)
            if best_config:
                self.meta_patterns["best_regime_configs"][regime] = {
                    "config_id": best_config.config_id,
                    "avg_performance": best_config.avg_performance,
                    "parameters": best_config.parameters.copy(),
                }

        # 2. Calculate meta-performance metrics
        all_performances = [
            perf
            for config in self.configs.values()
            for perf in config.performance_history
        ]

        if all_performances:
            overall_win_rate = np.mean(all_performances)

            # Learning efficiency: Is performance improving over time?
            if len(all_performances) > 50:
                early = np.mean(all_performances[:25])
                recent = np.mean(all_performances[-25:])
                learning_efficiency = max(0.0, (recent - early) / max(0.01, early))
            else:
                learning_efficiency = 0.0

            # Check for performance degradation
            if len(all_performances) > 30:
                recent_trend = np.polyfit(range(30), all_performances[-30:], 1)[0]
                if recent_trend < -0.01:  # Declining performance
                    self.performance_degradation_detected = True
                    logger.warning(
                        "performance_degradation_detected",
                        recent_trend=recent_trend,
                    )

            logger.info(
                "meta_patterns_learned",
                overall_win_rate=overall_win_rate,
                learning_efficiency=learning_efficiency,
                num_regimes=len(self.meta_patterns["best_regime_configs"]),
            )

    def suggest_hyperparameter_adjustment(
        self, parameter_name: str, current_value: float, regime: str
    ) -> float:
        """
        Suggest adjustment to a hyperparameter based on meta-learning.

        Args:
            parameter_name: Name of hyperparameter
            current_value: Current value
            regime: Current regime

        Returns:
            Suggested new value
        """
        # Check if we have a known-good config for this regime
        if regime in self.meta_patterns["best_regime_configs"]:
            best_config = self.meta_patterns["best_regime_configs"][regime]
            if parameter_name in best_config["parameters"]:
                suggested_value = best_config["parameters"][parameter_name]

                # Move toward suggested value slowly
                adjustment = (
                    suggested_value - current_value
                ) * self.meta_learning_rate

                new_value = current_value + adjustment

                logger.debug(
                    "hyperparameter_adjustment_suggested",
                    parameter=parameter_name,
                    current=current_value,
                    suggested=new_value,
                    regime=regime,
                )

                return new_value

        # No suggestion, keep current
        return current_value

    def should_trigger_system_reset(self) -> Tuple[bool, str]:
        """
        Determine if system should be reset due to poor performance.

        Returns:
            (should_reset, reason) tuple
        """
        # Check if stuck for too long
        if self.stuck_counter > 20:
            return (True, "System stuck - performance not improving for 20+ iterations")

        # Check if performance degrading
        if self.performance_degradation_detected:
            recent_configs = [
                c
                for c in self.configs.values()
                if len(c.performance_history) > 10
            ]

            if recent_configs:
                avg_recent = np.mean([c.avg_performance for c in recent_configs])
                if avg_recent < 0.45:  # Below 45% - really poor
                    return (True, f"Severe performance degradation - avg {avg_recent:.1%}")

        return (False, "Performance acceptable")

    def get_meta_performance_summary(self) -> Dict:
        """Get summary of meta-learning performance."""
        all_performances = [
            perf
            for config in self.configs.values()
            for perf in config.performance_history
        ]

        if not all_performances:
            return {"status": "no_data"}

        return {
            "num_configs": len(self.configs),
            "total_evaluations": len(all_performances),
            "overall_win_rate": np.mean(all_performances),
            "best_config_performance": max(c.avg_performance for c in self.configs.values()),
            "num_regimes_learned": len(self.meta_patterns["best_regime_configs"]),
            "stuck_counter": self.stuck_counter,
            "performance_degradation": self.performance_degradation_detected,
        }

    def get_state(self) -> Dict:
        """Get state for persistence."""
        return {
            "configs": {
                cid: {
                    "parameters": config.parameters,
                    "performance_history": config.performance_history[-50:],  # Keep recent
                    "regimes_tested": config.regimes_tested,
                    "avg_performance": config.avg_performance,
                    "sample_count": config.sample_count,
                }
                for cid, config in self.configs.items()
            },
            "meta_patterns": self.meta_patterns,
            "stuck_counter": self.stuck_counter,
            "performance_degradation_detected": self.performance_degradation_detected,
        }

    def load_state(self, state: Dict) -> None:
        """Load state from persistence."""
        # Load configs
        self.configs = {}
        for cid, config_data in state.get("configs", {}).items():
            self.configs[cid] = HyperparameterConfig(
                config_id=cid,
                parameters=config_data["parameters"],
                performance_history=config_data["performance_history"],
                regimes_tested=config_data["regimes_tested"],
                avg_performance=config_data["avg_performance"],
                sample_count=config_data["sample_count"],
            )

        self.meta_patterns = state.get("meta_patterns", {})
        self.stuck_counter = state.get("stuck_counter", 0)
        self.performance_degradation_detected = state.get(
            "performance_degradation_detected", False
        )

        logger.info(
            "meta_learner_state_loaded",
            num_configs=len(self.configs),
            num_regimes=len(self.meta_patterns.get("best_regime_configs", {})),
        )
