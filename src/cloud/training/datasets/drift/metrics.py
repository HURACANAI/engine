"""
Drift Detection Metrics

Statistical metrics for detecting distribution drift.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats
import structlog

logger = structlog.get_logger(__name__)


def calculate_psi(
    expected: pd.Series,
    actual: pd.Series,
    buckets: int = 10
) -> float:
    """
    Calculate Population Stability Index (PSI)

    PSI measures the change in distribution between two datasets.

    Interpretation:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.2: Moderate change (investigate)
    - PSI >= 0.2: Significant change (critical drift)

    Args:
        expected: Reference distribution (historical data)
        actual: Current distribution (new data)
        buckets: Number of buckets for binning

    Returns:
        PSI value (float)

    Example:
        psi = calculate_psi(
            expected=historical_df['close'],
            actual=new_df['close'],
            buckets=10
        )
    """
    # Remove NaN values
    expected = expected.dropna()
    actual = actual.dropna()

    if len(expected) == 0 or len(actual) == 0:
        logger.warning("empty_data_for_psi")
        return 0.0

    # Create bins based on expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)  # Remove duplicates

    if len(breakpoints) < 3:
        logger.warning("insufficient_unique_values_for_psi", unique_values=len(breakpoints))
        return 0.0

    # Calculate percentage in each bucket for expected
    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    expected_percents = expected_counts / len(expected)

    # Calculate percentage in each bucket for actual
    actual_counts, _ = np.histogram(actual, bins=breakpoints)
    actual_percents = actual_counts / len(actual)

    # Avoid division by zero and log(0)
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

    # Calculate PSI
    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))

    return float(psi)


def calculate_ks_statistic(
    expected: pd.Series,
    actual: pd.Series
) -> Tuple[float, float]:
    """
    Calculate Kolmogorov-Smirnov statistic

    KS test measures maximum difference between cumulative distributions.

    Args:
        expected: Reference distribution
        actual: Current distribution

    Returns:
        (ks_statistic, p_value)

    Interpretation:
    - p_value < 0.05: Distributions are significantly different
    - ks_statistic: Maximum distance (0-1, higher = more different)

    Example:
        ks_stat, p_value = calculate_ks_statistic(
            expected=historical_df['volume'],
            actual=new_df['volume']
        )
    """
    # Remove NaN values
    expected = expected.dropna()
    actual = actual.dropna()

    if len(expected) == 0 or len(actual) == 0:
        logger.warning("empty_data_for_ks_test")
        return 0.0, 1.0

    # Perform KS test
    ks_statistic, p_value = stats.ks_2samp(expected, actual)

    return float(ks_statistic), float(p_value)


def calculate_population_coverage(
    expected: pd.Series,
    actual: pd.Series,
    tolerance: float = 0.05
) -> float:
    """
    Calculate population coverage

    Measures what percentage of expected range is covered by actual data.

    Args:
        expected: Reference distribution
        actual: Current distribution
        tolerance: Range expansion tolerance (5% by default)

    Returns:
        Coverage ratio (0-1)

    Example:
        coverage = calculate_population_coverage(
            expected=historical_df['close'],
            actual=new_df['close']
        )
    """
    # Remove NaN values
    expected = expected.dropna()
    actual = actual.dropna()

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Calculate ranges
    exp_min, exp_max = expected.min(), expected.max()
    act_min, act_max = actual.min(), actual.max()

    # Expand expected range by tolerance
    exp_range = exp_max - exp_min
    exp_min_tol = exp_min - exp_range * tolerance
    exp_max_tol = exp_max + exp_range * tolerance

    # Check if actual is within expected range (with tolerance)
    if act_min >= exp_min_tol and act_max <= exp_max_tol:
        coverage = 1.0
    else:
        # Calculate partial coverage
        overlap_min = max(act_min, exp_min_tol)
        overlap_max = min(act_max, exp_max_tol)
        overlap_range = max(0, overlap_max - overlap_min)

        actual_range = act_max - act_min
        coverage = overlap_range / actual_range if actual_range > 0 else 0.0

    return float(coverage)


def calculate_missingness_drift(
    expected: pd.DataFrame,
    actual: pd.DataFrame
) -> dict:
    """
    Calculate drift in missing value patterns

    Args:
        expected: Reference DataFrame
        actual: Current DataFrame

    Returns:
        Dict of {column: (expected_missing_pct, actual_missing_pct, diff)}

    Example:
        missingness = calculate_missingness_drift(
            expected=historical_df,
            actual=new_df
        )
    """
    results = {}

    for col in expected.columns:
        if col not in actual.columns:
            continue

        expected_missing_pct = expected[col].isna().sum() / len(expected)
        actual_missing_pct = actual[col].isna().sum() / len(actual)
        diff = actual_missing_pct - expected_missing_pct

        results[col] = {
            "expected_missing_pct": float(expected_missing_pct),
            "actual_missing_pct": float(actual_missing_pct),
            "diff": float(diff)
        }

    return results


def calculate_correlation_drift(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    features: list
) -> float:
    """
    Calculate drift in correlation structure

    Measures change in feature correlations between datasets.

    Args:
        expected: Reference DataFrame
        actual: Current DataFrame
        features: List of feature columns to check

    Returns:
        Average absolute correlation difference

    Example:
        corr_drift = calculate_correlation_drift(
            expected=historical_df,
            actual=new_df,
            features=["close", "volume", "volatility"]
        )
    """
    # Filter to common features
    common_features = [f for f in features if f in expected.columns and f in actual.columns]

    if len(common_features) < 2:
        logger.warning("insufficient_features_for_correlation_drift")
        return 0.0

    # Calculate correlation matrices
    expected_corr = expected[common_features].corr()
    actual_corr = actual[common_features].corr()

    # Calculate average absolute difference
    diff_matrix = np.abs(expected_corr - actual_corr)

    # Get upper triangle (avoid counting twice)
    upper_triangle = np.triu_indices_from(diff_matrix, k=1)
    avg_drift = diff_matrix.values[upper_triangle].mean()

    return float(avg_drift)
