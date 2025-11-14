"""
Regime Warning Indicators

Statistical indicators for detecting early regime shifts.
"""

from typing import Tuple

import numpy as np
import pandas as pd


def detect_volatility_shift(
    returns: np.ndarray,
    window: int = 20,
    threshold_std: float = 2.0
) -> Tuple[bool, float]:
    """
    Detect volatility regime shift using rolling volatility

    Args:
        returns: Return series
        window: Rolling window size
        threshold_std: Standard deviation threshold for shift

    Returns:
        (shift_detected, current_vol_zscore)

    Example:
        shift, zscore = detect_volatility_shift(returns, window=20)
        if shift:
            print(f"Volatility shift detected: {zscore:.2f} std")
    """
    if len(returns) < window * 2:
        return False, 0.0

    returns = np.asarray(returns)

    # Calculate rolling volatility
    rolling_vol = pd.Series(returns).rolling(window).std()

    if len(rolling_vol) < window:
        return False, 0.0

    # Current volatility
    current_vol = rolling_vol.iloc[-1]

    # Historical volatility (excluding recent window)
    historical_vol = rolling_vol.iloc[:-window]

    if len(historical_vol) == 0:
        return False, 0.0

    # Z-score of current volatility
    vol_mean = historical_vol.mean()
    vol_std = historical_vol.std()

    if vol_std == 0:
        return False, 0.0

    vol_zscore = (current_vol - vol_mean) / vol_std

    # Shift detected if zscore exceeds threshold
    shift_detected = abs(vol_zscore) > threshold_std

    return shift_detected, vol_zscore


def detect_volume_anomaly(
    volumes: np.ndarray,
    window: int = 20,
    threshold_std: float = 3.0
) -> Tuple[bool, float]:
    """
    Detect volume anomalies (spikes or drops)

    Args:
        volumes: Volume series
        window: Rolling window size
        threshold_std: Standard deviation threshold

    Returns:
        (anomaly_detected, volume_zscore)

    Example:
        anomaly, zscore = detect_volume_anomaly(volumes)
        if anomaly:
            print(f"Volume anomaly: {zscore:.2f} std")
    """
    if len(volumes) < window * 2:
        return False, 0.0

    volumes = np.asarray(volumes)

    # Current volume
    current_volume = volumes[-1]

    # Historical volume (excluding recent)
    historical_volume = volumes[:-window]

    if len(historical_volume) == 0:
        return False, 0.0

    # Z-score
    vol_mean = historical_volume.mean()
    vol_std = historical_volume.std()

    if vol_std == 0:
        return False, 0.0

    volume_zscore = (current_volume - vol_mean) / vol_std

    # Anomaly if zscore exceeds threshold
    anomaly_detected = abs(volume_zscore) > threshold_std

    return anomaly_detected, volume_zscore


def detect_correlation_breakdown(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    window: int = 50,
    threshold_change: float = 0.3
) -> Tuple[bool, float]:
    """
    Detect correlation breakdown between two assets

    Useful for detecting regime shifts in crypto (e.g., BTC-ETH correlation)

    Args:
        returns_a: Returns for asset A
        returns_b: Returns for asset B
        window: Rolling correlation window
        threshold_change: Minimum correlation change to trigger

    Returns:
        (breakdown_detected, correlation_change)

    Example:
        breakdown, change = detect_correlation_breakdown(
            btc_returns, eth_returns
        )
        if breakdown:
            print(f"Correlation breakdown: {change:.2f} change")
    """
    if len(returns_a) < window * 2 or len(returns_b) < window * 2:
        return False, 0.0

    returns_a = np.asarray(returns_a)
    returns_b = np.asarray(returns_b)

    # Align lengths
    min_len = min(len(returns_a), len(returns_b))
    returns_a = returns_a[-min_len:]
    returns_b = returns_b[-min_len:]

    # Rolling correlation
    rolling_corr = pd.Series(returns_a).rolling(window).corr(pd.Series(returns_b))

    if len(rolling_corr) < window * 2:
        return False, 0.0

    # Current correlation
    current_corr = rolling_corr.iloc[-1]

    # Historical correlation (excluding recent)
    historical_corr = rolling_corr.iloc[:-window].mean()

    # Change in correlation
    corr_change = abs(current_corr - historical_corr)

    # Breakdown if change exceeds threshold
    breakdown_detected = corr_change > threshold_change

    return breakdown_detected, corr_change


def calculate_cusum(
    data: np.ndarray,
    target_mean: float,
    threshold: float = 5.0
) -> Tuple[bool, float]:
    """
    CUSUM (Cumulative Sum) change detection

    Detects shifts in mean value.

    Args:
        data: Time series data
        target_mean: Expected mean
        threshold: CUSUM threshold for shift detection

    Returns:
        (shift_detected, cusum_value)

    Example:
        shift, cusum = calculate_cusum(returns, target_mean=0)
        if shift:
            print(f"Mean shift detected: CUSUM={cusum:.2f}")
    """
    if len(data) == 0:
        return False, 0.0

    data = np.asarray(data)

    # Calculate deviations from target
    deviations = data - target_mean

    # Cumulative sum
    cusum = np.cumsum(deviations)

    # Current CUSUM value (normalized by std)
    current_cusum = cusum[-1]
    data_std = data.std()

    if data_std > 0:
        normalized_cusum = abs(current_cusum) / data_std
    else:
        normalized_cusum = 0.0

    # Shift if normalized CUSUM exceeds threshold
    shift_detected = normalized_cusum > threshold

    return shift_detected, normalized_cusum


def calculate_pages_test(
    data: np.ndarray,
    window: int = 20,
    threshold: float = 3.0
) -> Tuple[bool, float]:
    """
    Page's Test for detecting distributional changes

    Args:
        data: Time series data
        window: Window size for comparison
        threshold: Test statistic threshold

    Returns:
        (change_detected, test_statistic)

    Example:
        change, stat = calculate_pages_test(returns)
        if change:
            print(f"Distribution change: stat={stat:.2f}")
    """
    if len(data) < window * 2:
        return False, 0.0

    data = np.asarray(data)

    # Split into recent and historical
    recent = data[-window:]
    historical = data[:-window]

    # Calculate means
    recent_mean = recent.mean()
    historical_mean = historical.mean()
    historical_std = historical.std()

    if historical_std == 0:
        return False, 0.0

    # Test statistic (z-score of mean difference)
    test_stat = abs(recent_mean - historical_mean) / (historical_std / np.sqrt(window))

    # Change detected if test stat exceeds threshold
    change_detected = test_stat > threshold

    return change_detected, test_stat


def detect_return_distribution_shift(
    returns: np.ndarray,
    window: int = 50,
    threshold_ks: float = 0.3
) -> Tuple[bool, float]:
    """
    Detect shift in return distribution using Kolmogorov-Smirnov test

    Args:
        returns: Return series
        window: Window size
        threshold_ks: KS statistic threshold

    Returns:
        (shift_detected, ks_statistic)
    """
    if len(returns) < window * 2:
        return False, 0.0

    from scipy.stats import ks_2samp

    returns = np.asarray(returns)

    # Split into recent and historical
    recent = returns[-window:]
    historical = returns[:-window]

    # KS test
    ks_stat, p_value = ks_2samp(recent, historical)

    # Shift if KS stat exceeds threshold
    shift_detected = ks_stat > threshold_ks

    return shift_detected, ks_stat
