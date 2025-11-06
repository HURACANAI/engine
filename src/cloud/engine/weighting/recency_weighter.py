"""
Recency Weighting System

Implements exponential decay weighting for time-series data.
Recent samples get higher weight, old samples get lower weight.

Why this matters:
- Scalp mode: Microstructure changes fast (halflife ~10 days)
- Regime mode: Need longer memory (halflife ~60 days)
- Wrong weighting = model trains on stale patterns
"""

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


class RecencyWeighter:
    """
    Calculate recency weights for training samples.

    Usage:
        weighter = RecencyWeighter(halflife_days=10)
        weights = weighter.calculate_weights(data)

        # Use in model training
        model.fit(X, y, sample_weight=weights)
    """

    def __init__(
        self,
        halflife_days: float = 10.0,
        min_weight: float = 0.01
    ):
        """
        Initialize recency weighter.

        Args:
            halflife_days: Days for weight to decay to 50%
            min_weight: Minimum weight (prevents zero weights)
        """
        self.halflife = halflife_days
        self.min_weight = min_weight

        # Calculate decay constant: λ = ln(2) / halflife
        self.lambda_decay = np.log(2) / halflife_days

        logger.info(
            "recency_weighter_initialized",
            halflife_days=halflife_days,
            lambda_decay=self.lambda_decay,
            min_weight=min_weight
        )

    def calculate_weights(
        self,
        data: pl.DataFrame,
        timestamp_column: str = 'timestamp'
    ) -> np.ndarray:
        """
        Calculate weights for a dataframe.

        Args:
            data: DataFrame with timestamp column
            timestamp_column: Name of timestamp column

        Returns:
            NumPy array of weights (same length as data)
        """
        if len(data) == 0:
            return np.array([])

        # Get latest timestamp
        latest_time = data[timestamp_column].max()

        # Calculate days ago for each row
        days_ago = (
            (latest_time - data[timestamp_column])
            .dt.total_seconds() / (24 * 3600)
        ).to_numpy()

        # Apply exponential decay: weight = exp(-λ × days_ago)
        weights = np.exp(-self.lambda_decay * days_ago)

        # Apply minimum weight floor
        weights = np.maximum(weights, self.min_weight)

        # Normalize so weights sum to N (preserves sample count)
        weights = weights * len(weights) / weights.sum()

        logger.debug(
            "weights_calculated",
            num_samples=len(weights),
            weight_range=(weights.min(), weights.max()),
            avg_weight=weights.mean()
        )

        return weights

    def calculate_weights_from_labels(
        self,
        labeled_trades: list,
    ) -> np.ndarray:
        """
        Calculate weights from labeled trades.

        Args:
            labeled_trades: List of LabeledTrade objects

        Returns:
            NumPy array of weights
        """
        if not labeled_trades:
            return np.array([])

        # Get timestamps
        timestamps = np.array([t.entry_time for t in labeled_trades])

        # Get latest
        latest_time = max(timestamps)

        # Calculate days ago
        days_ago = np.array([
            (latest_time - t).total_seconds() / (24 * 3600)
            for t in timestamps
        ])

        # Apply decay
        weights = np.exp(-self.lambda_decay * days_ago)
        weights = np.maximum(weights, self.min_weight)

        # Normalize
        weights = weights * len(weights) / weights.sum()

        return weights

    def get_weight_at_age(self, days_ago: float) -> float:
        """
        Get weight for a specific age.

        Useful for understanding decay curve.

        Example:
            weighter = RecencyWeighter(halflife_days=10)
            weight_10d = weighter.get_weight_at_age(10)  # Returns ~0.5
        """
        weight = np.exp(-self.lambda_decay * days_ago)
        return max(weight, self.min_weight)

    def plot_decay_curve(self, max_days: int = 90):
        """
        Generate text-based visualization of decay curve.

        Args:
            max_days: Maximum days to plot
        """
        print(f"\n{'='*50}")
        print(f"RECENCY WEIGHT DECAY CURVE")
        print(f"Halflife: {self.halflife} days")
        print(f"{'='*50}\n")

        for days in [0, 5, 10, 20, 30, 60, 90]:
            if days > max_days:
                break

            weight = self.get_weight_at_age(days)
            bar_length = int(weight * 50)
            bar = '█' * bar_length

            print(f"{days:3d} days ago: {bar} {weight:.3f}")

        print()

    def get_effective_sample_size(self, weights: np.ndarray) -> float:
        """
        Calculate effective sample size after weighting.

        Formula: ESS = (Σw)² / Σ(w²)

        This tells you how many "effective" samples you have after weighting.

        Example:
            If you have 1000 samples but ESS=500, it's like having 500
            equally-weighted samples due to the recency decay.
        """
        if len(weights) == 0:
            return 0.0

        sum_weights = np.sum(weights)
        sum_weights_squared = np.sum(weights ** 2)

        ess = (sum_weights ** 2) / sum_weights_squared

        return ess


def create_mode_specific_weighter(mode: str) -> RecencyWeighter:
    """
    Create weighter with mode-appropriate halflife.

    Args:
        mode: 'scalp', 'confirm', 'regime', or 'risk'

    Returns:
        Configured RecencyWeighter
    """
    halflife_map = {
        'scalp': 10.0,      # Fast microstructure decay
        'confirm': 20.0,    # Medium decay
        'regime': 60.0,     # Slow decay for regimes
        'risk': 120.0,      # Very slow for risk/correlation
    }

    halflife = halflife_map.get(mode, 20.0)

    logger.info(
        "mode_specific_weighter_created",
        mode=mode,
        halflife_days=halflife
    )

    return RecencyWeighter(halflife_days=halflife)
