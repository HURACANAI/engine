"""
Diversity Measurement

Measures diversity between engine predictions for ensemble quality.
"""

from typing import Dict, List

import numpy as np


def calculate_diversity(
    predictions: List[np.ndarray],
    method: str = "disagreement"
) -> float:
    """
    Calculate diversity between predictions

    Args:
        predictions: List of prediction arrays from different engines
        method: Diversity metric ("disagreement", "correlation", "kl_divergence")

    Returns:
        Diversity score [0-1] (higher = more diverse)

    Example:
        pred1 = np.array([1, 1, -1, 1])
        pred2 = np.array([1, -1, -1, 1])
        pred3 = np.array([-1, -1, 1, 1])

        diversity = calculate_diversity([pred1, pred2, pred3])
        # Returns: ~0.5 (moderate diversity)
    """
    if len(predictions) < 2:
        return 0.0

    if method == "disagreement":
        return _calculate_disagreement_diversity(predictions)
    elif method == "correlation":
        return _calculate_correlation_diversity(predictions)
    else:
        return _calculate_disagreement_diversity(predictions)


def _calculate_disagreement_diversity(
    predictions: List[np.ndarray]
) -> float:
    """
    Calculate diversity as average pairwise disagreement

    Args:
        predictions: List of prediction arrays

    Returns:
        Diversity [0-1]
    """
    num_engines = len(predictions)
    num_samples = len(predictions[0])

    # Calculate pairwise disagreement
    total_disagreement = 0.0
    num_pairs = 0

    for i in range(num_engines):
        for j in range(i + 1, num_engines):
            # Disagreement rate between engine i and j
            disagreement = (predictions[i] != predictions[j]).sum() / num_samples
            total_disagreement += disagreement
            num_pairs += 1

    if num_pairs == 0:
        return 0.0

    avg_disagreement = total_disagreement / num_pairs

    return avg_disagreement


def _calculate_correlation_diversity(
    predictions: List[np.ndarray]
) -> float:
    """
    Calculate diversity as 1 - average correlation

    Args:
        predictions: List of prediction arrays

    Returns:
        Diversity [0-1]
    """
    num_engines = len(predictions)

    # Calculate pairwise correlations
    correlations = []

    for i in range(num_engines):
        for j in range(i + 1, num_engines):
            corr = np.corrcoef(predictions[i], predictions[j])[0, 1]
            correlations.append(abs(corr))

    if len(correlations) == 0:
        return 0.0

    avg_correlation = np.mean(correlations)

    # Diversity = 1 - correlation
    diversity = 1.0 - avg_correlation

    return max(0.0, min(1.0, diversity))


def measure_prediction_diversity(
    predictions_dict: Dict[str, int]
) -> float:
    """
    Measure diversity in a single prediction round

    Args:
        predictions_dict: Dict of {engine_name: signal} where signal in {-1, 0, 1}

    Returns:
        Diversity score [0-1]

    Example:
        predictions = {
            "engine1": 1,
            "engine2": 1,
            "engine3": -1,
            "engine4": 0
        }

        diversity = measure_prediction_diversity(predictions)
        # Returns: ~0.67 (3 different predictions out of 4)
    """
    if len(predictions_dict) == 0:
        return 0.0

    # Count unique predictions
    unique_predictions = len(set(predictions_dict.values()))

    # Max possible unique = number of engines
    max_unique = len(predictions_dict)

    if max_unique == 1:
        return 0.0

    # Diversity as fraction of unique predictions
    diversity = (unique_predictions - 1) / (max_unique - 1)

    return diversity


def calculate_ensemble_quality(
    predictions: List[np.ndarray],
    accuracies: List[float]
) -> float:
    """
    Calculate ensemble quality score

    Combines individual accuracy with diversity.

    Args:
        predictions: List of prediction arrays
        accuracies: List of individual engine accuracies

    Returns:
        Ensemble quality [0-1]

    Example:
        pred1 = np.array([1, 1, -1])
        pred2 = np.array([1, -1, -1])

        quality = calculate_ensemble_quality(
            predictions=[pred1, pred2],
            accuracies=[0.7, 0.65]
        )
    """
    if len(predictions) == 0 or len(accuracies) == 0:
        return 0.0

    # Average individual accuracy
    avg_accuracy = np.mean(accuracies)

    # Diversity
    diversity = calculate_diversity(predictions, method="disagreement")

    # Ensemble quality = accuracy * (1 + diversity)
    # High accuracy + high diversity = best ensemble
    quality = avg_accuracy * (1 + diversity)

    # Normalize to [0, 1]
    quality = min(1.0, quality / 2.0)

    return quality
