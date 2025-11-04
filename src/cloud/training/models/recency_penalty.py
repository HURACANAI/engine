"""
Recency Penalty Module

Applies time-based decay to pattern similarities so recent patterns are more important.
Uses exponential decay: similarity_adjusted = similarity * exp(-age_days / half_life)

Based on Revuelto bot's recency penalty system.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class RecencyPenaltyConfig:
    """Configuration for recency penalties."""

    half_life_days: float = 30.0  # Pattern relevance halves every 30 days
    min_similarity_after_decay: float = 0.3  # Drop patterns below this after decay
    max_age_days: float = 365.0  # Ignore patterns older than 1 year
    enabled: bool = True  # Can disable for A/B testing


class RecencyPenalty:
    """
    Applies exponential time decay to pattern similarities.

    Key principle: Recent market behavior is more relevant than old behavior.

    Decay formula: adjusted_similarity = similarity * exp(-age_days / half_life)

    Example with 30-day half-life:
    - Pattern from 0 days ago: 100% weight (multiply by 1.0)
    - Pattern from 30 days ago: 50% weight (multiply by 0.5)
    - Pattern from 60 days ago: 25% weight (multiply by 0.25)
    - Pattern from 90 days ago: 12.5% weight (multiply by 0.125)
    """

    def __init__(self, config: Optional[RecencyPenaltyConfig] = None):
        """
        Initialize recency penalty calculator.

        Args:
            config: RecencyPenaltyConfig or None for defaults
        """
        self.config = config or RecencyPenaltyConfig()

        logger.info(
            "recency_penalty_initialized",
            half_life_days=self.config.half_life_days,
            max_age_days=self.config.max_age_days,
            enabled=self.config.enabled,
        )

    def apply_decay(
        self,
        similarity: float,
        pattern_timestamp: datetime,
        current_timestamp: datetime,
    ) -> float:
        """
        Apply time decay to a single similarity score.

        Args:
            similarity: Original similarity score (0-1)
            pattern_timestamp: When the pattern occurred
            current_timestamp: Current time

        Returns:
            Adjusted similarity score (0-1)
        """
        if not self.config.enabled:
            return similarity

        # Calculate age in days
        age_days = (current_timestamp - pattern_timestamp).total_seconds() / 86400.0

        # Drop patterns that are too old
        if age_days > self.config.max_age_days:
            return 0.0

        # Apply exponential decay
        decay_factor = np.exp(-age_days / self.config.half_life_days)
        adjusted_similarity = similarity * decay_factor

        # Drop patterns below minimum threshold after decay
        if adjusted_similarity < self.config.min_similarity_after_decay:
            return 0.0

        return adjusted_similarity

    def apply_decay_batch(
        self,
        similarities: np.ndarray,
        pattern_timestamps: List[datetime],
        current_timestamp: datetime,
    ) -> np.ndarray:
        """
        Apply time decay to multiple similarity scores at once.

        Args:
            similarities: Array of similarity scores (0-1)
            pattern_timestamps: List of timestamps for each pattern
            current_timestamp: Current time

        Returns:
            Array of adjusted similarity scores
        """
        if not self.config.enabled:
            return similarities

        # Calculate ages in days
        ages_days = np.array([
            (current_timestamp - ts).total_seconds() / 86400.0
            for ts in pattern_timestamps
        ])

        # Apply exponential decay
        decay_factors = np.exp(-ages_days / self.config.half_life_days)
        adjusted_similarities = similarities * decay_factors

        # Drop patterns that are too old
        adjusted_similarities[ages_days > self.config.max_age_days] = 0.0

        # Drop patterns below minimum threshold after decay
        adjusted_similarities[adjusted_similarities < self.config.min_similarity_after_decay] = 0.0

        return adjusted_similarities

    def calculate_effective_sample_size(
        self,
        pattern_timestamps: List[datetime],
        current_timestamp: datetime,
    ) -> float:
        """
        Calculate effective sample size after applying decay.

        This is useful for confidence calculations - if you have 100 patterns
        but they're all old, the effective sample size might only be 20.

        Args:
            pattern_timestamps: List of timestamps for each pattern
            current_timestamp: Current time

        Returns:
            Effective sample size (weighted count)
        """
        if not self.config.enabled or not pattern_timestamps:
            return float(len(pattern_timestamps))

        # Calculate decay weight for each pattern
        ages_days = np.array([
            (current_timestamp - ts).total_seconds() / 86400.0
            for ts in pattern_timestamps
        ])

        decay_factors = np.exp(-ages_days / self.config.half_life_days)

        # Drop patterns that are too old
        decay_factors[ages_days > self.config.max_age_days] = 0.0

        # Sum up the weights
        effective_size = np.sum(decay_factors)

        return float(effective_size)

    def get_decay_factor(
        self,
        pattern_timestamp: datetime,
        current_timestamp: datetime,
    ) -> float:
        """
        Get the decay factor for a pattern (without applying to similarity).

        Args:
            pattern_timestamp: When the pattern occurred
            current_timestamp: Current time

        Returns:
            Decay factor (0-1), where 1 = no decay, 0 = completely decayed
        """
        if not self.config.enabled:
            return 1.0

        age_days = (current_timestamp - pattern_timestamp).total_seconds() / 86400.0

        if age_days > self.config.max_age_days:
            return 0.0

        decay_factor = np.exp(-age_days / self.config.half_life_days)

        return float(decay_factor)

    def filter_patterns(
        self,
        patterns: List[dict],
        current_timestamp: datetime,
        timestamp_key: str = "timestamp",
    ) -> List[dict]:
        """
        Filter out patterns that are too old.

        Args:
            patterns: List of pattern dictionaries
            current_timestamp: Current time
            timestamp_key: Key in pattern dict containing timestamp

        Returns:
            Filtered list of patterns
        """
        if not self.config.enabled:
            return patterns

        filtered = []
        for pattern in patterns:
            pattern_ts = pattern.get(timestamp_key)
            if pattern_ts is None:
                continue

            # Parse timestamp if it's a string
            if isinstance(pattern_ts, str):
                try:
                    pattern_ts = datetime.fromisoformat(pattern_ts)
                except ValueError:
                    continue

            age_days = (current_timestamp - pattern_ts).total_seconds() / 86400.0

            if age_days <= self.config.max_age_days:
                filtered.append(pattern)

        removed = len(patterns) - len(filtered)
        if removed > 0:
            logger.debug(
                "old_patterns_filtered",
                total_patterns=len(patterns),
                filtered_out=removed,
                remaining=len(filtered),
            )

        return filtered


def calculate_time_weighted_win_rate(
    pattern_outcomes: List[bool],
    pattern_timestamps: List[datetime],
    current_timestamp: datetime,
    half_life_days: float = 30.0,
) -> float:
    """
    Calculate win rate with time-based weighting.

    Args:
        pattern_outcomes: List of boolean win/loss outcomes
        pattern_timestamps: Timestamps for each outcome
        current_timestamp: Current time
        half_life_days: Decay half-life in days

    Returns:
        Time-weighted win rate (0-1)
    """
    if not pattern_outcomes or len(pattern_outcomes) != len(pattern_timestamps):
        return 0.5  # Default to 50% if no data

    # Calculate decay weights
    ages_days = np.array([
        (current_timestamp - ts).total_seconds() / 86400.0
        for ts in pattern_timestamps
    ])

    weights = np.exp(-ages_days / half_life_days)

    # Calculate weighted win rate
    outcomes_array = np.array([1.0 if outcome else 0.0 for outcome in pattern_outcomes])
    weighted_wins = np.sum(outcomes_array * weights)
    total_weight = np.sum(weights)

    if total_weight < 1e-6:
        return 0.5

    return float(weighted_wins / total_weight)
