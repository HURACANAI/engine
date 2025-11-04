"""
Feature Importance Learning Module

Learns which features are most predictive of trading outcomes over time.
Uses online learning with EMA to adapt to changing market conditions.

Based on Revuelto bot's feature importance system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger()


@dataclass
class FeatureImportance:
    """Importance metrics for a single feature."""

    feature_name: str
    win_correlation: float = 0.0  # Correlation with winning trades
    profit_correlation: float = 0.0  # Correlation with profit amount
    loss_correlation: float = 0.0  # Correlation with losing trades
    importance_score: float = 0.0  # Overall importance (0-1)
    sample_count: int = 0  # Number of samples seen
    last_updated_timestamp: Optional[str] = None


@dataclass
class FeatureImportanceResult:
    """Result of feature importance analysis."""

    top_win_features: List[Tuple[str, float]]  # [(feature, importance), ...]
    top_loss_features: List[Tuple[str, float]]
    top_profit_features: List[Tuple[str, float]]
    feature_weights: Dict[str, float]  # Normalized weights for all features
    total_samples: int
    stats: Dict[str, float] = field(default_factory=dict)


class FeatureImportanceLearner:
    """
    Learns feature importance using online correlation tracking.

    Key innovations from Revuelto:
    1. EMA-based correlation updates (adapt to regime changes)
    2. Separate tracking for wins, losses, and profit magnitude
    3. Sample-size weighted importance scores
    4. Automatic feature weight normalization
    """

    def __init__(
        self,
        ema_alpha: float = 0.05,
        min_samples_for_confidence: int = 30,
        top_k_features: int = 10,
    ):
        """
        Initialize the feature importance learner.

        Args:
            ema_alpha: Learning rate for EMA updates (0.05 = ~20-trade memory)
            min_samples_for_confidence: Minimum samples before trusting importance
            top_k_features: Number of top features to track
        """
        self.ema_alpha = ema_alpha
        self.min_samples = min_samples_for_confidence
        self.top_k = top_k_features

        # Feature importance tracking
        self.feature_importance: Dict[str, FeatureImportance] = {}

        # Running statistics for normalization
        self.feature_means: Dict[str, float] = {}
        self.feature_stds: Dict[str, float] = {}

        # Global counters
        self.total_samples = 0
        self.total_wins = 0
        self.total_losses = 0

        logger.info(
            "feature_importance_learner_initialized",
            ema_alpha=ema_alpha,
            min_samples=min_samples_for_confidence,
            top_k=top_k_features,
        )

    def update(
        self,
        features: Dict[str, float],
        is_winner: bool,
        profit_bps: float,
        timestamp: Optional[str] = None,
    ) -> None:
        """
        Update feature importance based on trade outcome.

        Args:
            features: Feature values from the trade
            is_winner: Whether trade was profitable
            profit_bps: Profit/loss in basis points
            timestamp: Trade timestamp for tracking
        """
        self.total_samples += 1

        if is_winner:
            self.total_wins += 1
        else:
            self.total_losses += 1

        # Update each feature's importance
        for feature_name, feature_value in features.items():
            if not isinstance(feature_value, (int, float)):
                continue

            if np.isnan(feature_value) or np.isinf(feature_value):
                continue

            # Initialize if new feature
            if feature_name not in self.feature_importance:
                self.feature_importance[feature_name] = FeatureImportance(
                    feature_name=feature_name
                )
                self.feature_means[feature_name] = 0.0
                self.feature_stds[feature_name] = 1.0

            importance = self.feature_importance[feature_name]

            # Update running mean and std (for normalization)
            old_mean = self.feature_means[feature_name]
            self.feature_means[feature_name] = (
                1 - self.ema_alpha
            ) * old_mean + self.ema_alpha * feature_value

            # Simple std estimate
            if importance.sample_count > 1:
                variance = (feature_value - old_mean) ** 2
                old_std = self.feature_stds[feature_name]
                self.feature_stds[feature_name] = np.sqrt(
                    (1 - self.ema_alpha) * (old_std**2) + self.ema_alpha * variance
                )

            # Normalize feature value
            normalized_value = self._normalize_feature(feature_name, feature_value)

            # Update correlations using EMA
            # Win correlation: positive if feature high on wins
            win_signal = 1.0 if is_winner else -1.0
            correlation_update = normalized_value * win_signal

            importance.win_correlation = (
                1 - self.ema_alpha
            ) * importance.win_correlation + self.ema_alpha * correlation_update

            # Loss correlation: positive if feature high on losses
            loss_signal = -1.0 if is_winner else 1.0
            loss_update = normalized_value * loss_signal

            importance.loss_correlation = (
                1 - self.ema_alpha
            ) * importance.loss_correlation + self.ema_alpha * loss_update

            # Profit correlation: correlation with profit magnitude
            # Normalize profit to [-1, 1] range (rough heuristic)
            normalized_profit = np.tanh(profit_bps / 100.0)
            profit_update = normalized_value * normalized_profit

            importance.profit_correlation = (
                1 - self.ema_alpha
            ) * importance.profit_correlation + self.ema_alpha * profit_update

            # Calculate overall importance score
            importance.importance_score = self._calculate_importance_score(importance)

            # Update metadata
            importance.sample_count += 1
            importance.last_updated_timestamp = timestamp

    def _normalize_feature(self, feature_name: str, value: float) -> float:
        """Normalize feature value to ~[-1, 1] range using running stats."""
        mean = self.feature_means.get(feature_name, 0.0)
        std = self.feature_stds.get(feature_name, 1.0)

        if std < 1e-6:
            return 0.0

        # Z-score, clipped to [-3, 3] then scaled to [-1, 1]
        z_score = (value - mean) / std
        z_score = np.clip(z_score, -3.0, 3.0)

        return z_score / 3.0

    def _calculate_importance_score(self, importance: FeatureImportance) -> float:
        """
        Calculate overall importance score for a feature.

        Higher score = more predictive of outcomes.
        """
        # Base score from correlations (use absolute values)
        win_contrib = abs(importance.win_correlation)
        loss_contrib = abs(importance.loss_correlation)
        profit_contrib = abs(importance.profit_correlation)

        # Weighted combination (profit correlation most important)
        base_score = 0.3 * win_contrib + 0.2 * loss_contrib + 0.5 * profit_contrib

        # Apply sample size discount (sigmoid)
        if importance.sample_count < self.min_samples:
            confidence = 1.0 - np.exp(-importance.sample_count / (self.min_samples / 3))
        else:
            confidence = 1.0

        return base_score * confidence

    def get_feature_importance(self) -> FeatureImportanceResult:
        """
        Get current feature importance rankings.

        Returns:
            FeatureImportanceResult with top features and weights
        """
        if not self.feature_importance:
            return FeatureImportanceResult(
                top_win_features=[],
                top_loss_features=[],
                top_profit_features=[],
                feature_weights={},
                total_samples=0,
            )

        # Sort by different criteria
        by_win_corr = sorted(
            self.feature_importance.values(),
            key=lambda x: x.win_correlation,
            reverse=True,
        )

        by_loss_corr = sorted(
            self.feature_importance.values(),
            key=lambda x: x.loss_correlation,
            reverse=True,
        )

        by_profit_corr = sorted(
            self.feature_importance.values(),
            key=lambda x: abs(x.profit_correlation),
            reverse=True,
        )

        # Extract top K
        top_win_features = [
            (f.feature_name, f.win_correlation) for f in by_win_corr[: self.top_k]
        ]

        top_loss_features = [
            (f.feature_name, f.loss_correlation) for f in by_loss_corr[: self.top_k]
        ]

        top_profit_features = [
            (f.feature_name, f.profit_correlation) for f in by_profit_corr[: self.top_k]
        ]

        # Calculate normalized feature weights
        feature_weights = self._calculate_feature_weights()

        # Calculate stats
        avg_importance = np.mean([f.importance_score for f in self.feature_importance.values()])
        max_importance = max([f.importance_score for f in self.feature_importance.values()])

        stats = {
            "total_features": len(self.feature_importance),
            "avg_importance": avg_importance,
            "max_importance": max_importance,
            "win_rate": self.total_wins / self.total_samples if self.total_samples > 0 else 0.0,
        }

        return FeatureImportanceResult(
            top_win_features=top_win_features,
            top_loss_features=top_loss_features,
            top_profit_features=top_profit_features,
            feature_weights=feature_weights,
            total_samples=self.total_samples,
            stats=stats,
        )

    def _calculate_feature_weights(self) -> Dict[str, float]:
        """Calculate normalized weights for all features."""
        weights = {}

        # Get all importance scores
        scores = {
            name: imp.importance_score
            for name, imp in self.feature_importance.items()
        }

        if not scores:
            return {}

        # Normalize to sum to 1.0
        total_score = sum(scores.values())

        if total_score < 1e-6:
            # All features equally weighted if no signal
            uniform_weight = 1.0 / len(scores)
            return {name: uniform_weight for name in scores.keys()}

        for name, score in scores.items():
            weights[name] = score / total_score

        return weights

    def get_weighted_features(
        self, features: Dict[str, float], top_k: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Return features weighted by their importance.

        Args:
            features: Raw feature dict
            top_k: If specified, only return top K most important features

        Returns:
            Dictionary of weighted features
        """
        weights = self._calculate_feature_weights()

        weighted = {}
        for name, value in features.items():
            if name in weights:
                weighted[name] = value * weights[name]
            else:
                # New feature - give it average weight
                avg_weight = 1.0 / len(self.feature_importance) if self.feature_importance else 1.0
                weighted[name] = value * avg_weight

        if top_k is not None:
            # Sort by absolute weighted value and take top K
            sorted_features = sorted(
                weighted.items(), key=lambda x: abs(x[1]), reverse=True
            )
            return dict(sorted_features[:top_k])

        return weighted

    def get_feature_stats(self, feature_name: str) -> Optional[Dict[str, float]]:
        """Get detailed statistics for a specific feature."""
        if feature_name not in self.feature_importance:
            return None

        importance = self.feature_importance[feature_name]

        return {
            "win_correlation": importance.win_correlation,
            "loss_correlation": importance.loss_correlation,
            "profit_correlation": importance.profit_correlation,
            "importance_score": importance.importance_score,
            "sample_count": importance.sample_count,
            "mean": self.feature_means.get(feature_name, 0.0),
            "std": self.feature_stds.get(feature_name, 1.0),
        }

    def reset(self) -> None:
        """Reset all learned importance (useful for regime changes)."""
        self.feature_importance.clear()
        self.feature_means.clear()
        self.feature_stds.clear()
        self.total_samples = 0
        self.total_wins = 0
        self.total_losses = 0

        logger.info("feature_importance_reset")

    def get_state(self) -> Dict:
        """Get current state for persistence."""
        return {
            "feature_importance": {
                name: {
                    "win_correlation": imp.win_correlation,
                    "loss_correlation": imp.loss_correlation,
                    "profit_correlation": imp.profit_correlation,
                    "importance_score": imp.importance_score,
                    "sample_count": imp.sample_count,
                }
                for name, imp in self.feature_importance.items()
            },
            "feature_means": self.feature_means.copy(),
            "feature_stds": self.feature_stds.copy(),
            "total_samples": self.total_samples,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
        }

    def load_state(self, state: Dict) -> None:
        """Load state from persistence."""
        self.feature_importance = {}
        for name, data in state.get("feature_importance", {}).items():
            self.feature_importance[name] = FeatureImportance(
                feature_name=name,
                win_correlation=data["win_correlation"],
                loss_correlation=data["loss_correlation"],
                profit_correlation=data["profit_correlation"],
                importance_score=data["importance_score"],
                sample_count=data["sample_count"],
            )

        self.feature_means = state.get("feature_means", {})
        self.feature_stds = state.get("feature_stds", {})
        self.total_samples = state.get("total_samples", 0)
        self.total_wins = state.get("total_wins", 0)
        self.total_losses = state.get("total_losses", 0)

        logger.info(
            "feature_importance_state_loaded",
            total_samples=self.total_samples,
            num_features=len(self.feature_importance),
        )
