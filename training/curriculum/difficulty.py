"""
Sample Difficulty Calculation

Determines difficulty of training samples for curriculum learning.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd


def calculate_sample_difficulty(
    features: pd.DataFrame,
    labels: pd.Series,
    method: str = "combined"
) -> np.ndarray:
    """
    Calculate difficulty score for each sample

    Args:
        features: Feature DataFrame
        labels: Label series
        method: Difficulty metric ("volatility", "label_rarity", "combined")

    Returns:
        Array of difficulty scores [0-1] (higher = harder)

    Example:
        difficulty = calculate_sample_difficulty(X, y, method="combined")
        easy_samples = difficulty < 0.3
        hard_samples = difficulty > 0.7
    """
    if method == "volatility":
        return _difficulty_by_volatility(features)
    elif method == "label_rarity":
        return _difficulty_by_label_rarity(labels)
    elif method == "feature_complexity":
        return _difficulty_by_feature_complexity(features)
    elif method == "combined":
        return _difficulty_combined(features, labels)
    else:
        raise ValueError(f"Unknown difficulty method: {method}")


def _difficulty_by_volatility(features: pd.DataFrame) -> np.ndarray:
    """
    Difficulty based on price volatility

    High volatility = harder to predict
    """
    # Assume 'close' column exists
    if 'close' not in features.columns:
        # Use first numeric column
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return np.zeros(len(features))
        price_col = numeric_cols[0]
    else:
        price_col = 'close'

    # Calculate rolling volatility (returns std)
    returns = features[price_col].pct_change()
    volatility = returns.rolling(window=20, min_periods=1).std()

    # Normalize to [0, 1]
    vol_min = volatility.min()
    vol_max = volatility.max()

    if vol_max == vol_min:
        return np.zeros(len(features))

    difficulty = (volatility - vol_min) / (vol_max - vol_min)

    return difficulty.fillna(0).values


def _difficulty_by_label_rarity(labels: pd.Series) -> np.ndarray:
    """
    Difficulty based on label rarity

    Rare labels = harder to learn
    """
    # Count label frequencies
    label_counts = labels.value_counts()

    # Map each label to its frequency
    label_freq = labels.map(label_counts)

    # Difficulty = 1 / frequency (normalized)
    difficulty = 1.0 / label_freq

    # Normalize to [0, 1]
    diff_min = difficulty.min()
    diff_max = difficulty.max()

    if diff_max == diff_min:
        return np.zeros(len(labels))

    difficulty = (difficulty - diff_min) / (diff_max - diff_min)

    return difficulty.values


def _difficulty_by_feature_complexity(features: pd.DataFrame) -> np.ndarray:
    """
    Difficulty based on feature complexity

    More extreme feature values = harder
    """
    # Calculate z-scores for all numeric features
    numeric_features = features.select_dtypes(include=[np.number])

    if numeric_features.shape[1] == 0:
        return np.zeros(len(features))

    # Standardize features
    standardized = (numeric_features - numeric_features.mean()) / numeric_features.std()

    # Difficulty = mean absolute z-score
    difficulty = standardized.abs().mean(axis=1)

    # Normalize to [0, 1]
    diff_min = difficulty.min()
    diff_max = difficulty.max()

    if diff_max == diff_min:
        return np.zeros(len(features))

    difficulty = (difficulty - diff_min) / (diff_max - diff_min)

    return difficulty.values


def _difficulty_combined(features: pd.DataFrame, labels: pd.Series) -> np.ndarray:
    """
    Combined difficulty metric

    Averages multiple difficulty measures
    """
    diff_vol = _difficulty_by_volatility(features)
    diff_rarity = _difficulty_by_label_rarity(labels)
    diff_complexity = _difficulty_by_feature_complexity(features)

    # Weighted average
    difficulty = (
        0.4 * diff_vol +
        0.3 * diff_rarity +
        0.3 * diff_complexity
    )

    return difficulty


def rank_by_difficulty(
    features: pd.DataFrame,
    labels: pd.Series,
    method: str = "combined"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rank samples by difficulty

    Args:
        features: Feature DataFrame
        labels: Label series
        method: Difficulty metric

    Returns:
        (difficulty_scores, sorted_indices)

    Example:
        difficulty, indices = rank_by_difficulty(X, y)

        # Train on easiest 30% first
        easy_indices = indices[:int(len(indices) * 0.3)]
        X_easy = X.iloc[easy_indices]
        y_easy = y.iloc[easy_indices]
    """
    difficulty = calculate_sample_difficulty(features, labels, method)

    # Sort indices by difficulty (ascending = easy to hard)
    sorted_indices = np.argsort(difficulty)

    return difficulty, sorted_indices


def create_difficulty_bins(
    difficulty: np.ndarray,
    num_bins: int = 3
) -> np.ndarray:
    """
    Create difficulty bins (e.g., easy, medium, hard)

    Args:
        difficulty: Difficulty scores
        num_bins: Number of bins (default 3 = easy/medium/hard)

    Returns:
        Array of bin labels [0, num_bins-1]

    Example:
        bins = create_difficulty_bins(difficulty, num_bins=3)
        easy_mask = bins == 0
        medium_mask = bins == 1
        hard_mask = bins == 2
    """
    # Use quantile-based binning for balanced bins
    bins = pd.qcut(
        difficulty,
        q=num_bins,
        labels=False,
        duplicates='drop'
    )

    return bins


def get_regime_specific_difficulty(
    features: pd.DataFrame,
    labels: pd.Series,
    regimes: pd.Series
) -> dict:
    """
    Calculate difficulty separately per regime

    Args:
        features: Feature DataFrame
        labels: Label series
        regimes: Regime labels for each sample

    Returns:
        Dict of {regime: difficulty_scores}

    Example:
        regime_difficulty = get_regime_specific_difficulty(X, y, regimes)

        # Get easy samples in trending regime
        trending_difficulty = regime_difficulty['trending']
        trending_easy = trending_difficulty < 0.3
    """
    regime_difficulty = {}

    for regime in regimes.unique():
        regime_mask = regimes == regime

        regime_features = features[regime_mask]
        regime_labels = labels[regime_mask]

        difficulty = calculate_sample_difficulty(
            regime_features,
            regime_labels,
            method="combined"
        )

        regime_difficulty[regime] = difficulty

    return regime_difficulty
