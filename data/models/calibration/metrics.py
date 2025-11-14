"""
Calibration Metrics

Metrics for evaluating probability calibration quality.
"""

from typing import Tuple

import numpy as np
from sklearn.calibration import calibration_curve as sklearn_calibration_curve


def calculate_brier_score(
    probabilities: np.ndarray,
    actual_outcomes: np.ndarray
) -> float:
    """
    Calculate Brier Score (mean squared error of probabilities)

    Lower is better. Perfect calibration = 0.0.

    Args:
        probabilities: Predicted probabilities [0-1]
        actual_outcomes: Actual binary outcomes {0, 1}

    Returns:
        Brier score (float)

    Interpretation:
    - < 0.1: Excellent calibration
    - 0.1-0.2: Good calibration
    - 0.2-0.3: Moderate calibration
    - > 0.3: Poor calibration

    Example:
        brier = calculate_brier_score(
            probabilities=model.predict_proba(X)[:, 1],
            actual_outcomes=y
        )
    """
    probabilities = np.asarray(probabilities)
    actual_outcomes = np.asarray(actual_outcomes)

    # Brier score = mean((p - y)^2)
    brier = np.mean((probabilities - actual_outcomes) ** 2)

    return float(brier)


def calculate_ece(
    probabilities: np.ndarray,
    actual_outcomes: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error (ECE)

    Measures average difference between confidence and accuracy across bins.

    Args:
        probabilities: Predicted probabilities [0-1]
        actual_outcomes: Actual binary outcomes {0, 1}
        n_bins: Number of bins for binning probabilities

    Returns:
        ECE (float, 0-1)

    Interpretation:
    - < 0.05: Well calibrated
    - 0.05-0.10: Moderately calibrated
    - > 0.10: Poorly calibrated

    Example:
        ece = calculate_ece(
            probabilities=model.predict_proba(X)[:, 1],
            actual_outcomes=y,
            n_bins=10
        )
    """
    probabilities = np.asarray(probabilities)
    actual_outcomes = np.asarray(actual_outcomes)

    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probabilities, bin_edges[:-1], right=False) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0

    for bin_idx in range(n_bins):
        bin_mask = bin_indices == bin_idx
        bin_size = bin_mask.sum()

        if bin_size == 0:
            continue

        # Average confidence in this bin
        bin_confidence = probabilities[bin_mask].mean()

        # Actual accuracy in this bin
        bin_accuracy = actual_outcomes[bin_mask].mean()

        # Weighted contribution to ECE
        ece += (bin_size / len(probabilities)) * abs(bin_confidence - bin_accuracy)

    return float(ece)


def calculate_calibration_curve(
    probabilities: np.ndarray,
    actual_outcomes: np.ndarray,
    n_bins: int = 10,
    strategy: str = 'uniform'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate calibration curve (reliability diagram data)

    Args:
        probabilities: Predicted probabilities [0-1]
        actual_outcomes: Actual binary outcomes {0, 1}
        n_bins: Number of bins
        strategy: 'uniform' or 'quantile' binning

    Returns:
        (mean_predicted_probs, fraction_positives) for each bin

    Example:
        mean_pred, frac_pos = calculate_calibration_curve(
            probabilities=model.predict_proba(X)[:, 1],
            actual_outcomes=y
        )

        # Plot
        import matplotlib.pyplot as plt
        plt.plot(mean_pred, frac_pos, marker='o')
        plt.plot([0, 1], [0, 1], 'k--')  # Perfect calibration line
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
    """
    probabilities = np.asarray(probabilities)
    actual_outcomes = np.asarray(actual_outcomes)

    # Use sklearn's calibration curve
    fraction_of_positives, mean_predicted_value = sklearn_calibration_curve(
        actual_outcomes,
        probabilities,
        n_bins=n_bins,
        strategy=strategy
    )

    return mean_predicted_value, fraction_of_positives


def calculate_calibration_by_regime(
    probabilities: np.ndarray,
    actual_outcomes: np.ndarray,
    regimes: np.ndarray
) -> dict:
    """
    Calculate calibration metrics separately for each regime

    Args:
        probabilities: Predicted probabilities
        actual_outcomes: Actual outcomes
        regimes: Regime labels for each sample

    Returns:
        Dict of {regime: {"brier": float, "ece": float}}

    Example:
        regime_calibration = calculate_calibration_by_regime(
            probabilities=probs,
            actual_outcomes=y,
            regimes=regime_labels
        )

        for regime, metrics in regime_calibration.items():
            print(f"{regime}: Brier={metrics['brier']:.3f}")
    """
    results = {}

    unique_regimes = np.unique(regimes)

    for regime in unique_regimes:
        regime_mask = regimes == regime

        regime_probs = probabilities[regime_mask]
        regime_outcomes = actual_outcomes[regime_mask]

        if len(regime_probs) < 10:
            # Not enough samples for reliable calibration
            continue

        brier = calculate_brier_score(regime_probs, regime_outcomes)
        ece = calculate_ece(regime_probs, regime_outcomes)

        results[regime] = {
            "brier": float(brier),
            "ece": float(ece),
            "num_samples": int(len(regime_probs))
        }

    return results
