"""
Leakage Detection Checks

Individual check functions for different types of data leakage.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


def check_feature_windowing(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    feature_cols: Optional[List[str]] = None,
    max_future_rows: int = 0
) -> List[str]:
    """
    Check if features use future data in their windowing

    Detects when features at time T use data from T+1 or later.

    Args:
        df: DataFrame with features and timestamps
        timestamp_col: Name of timestamp column
        feature_cols: List of feature columns to check (None = all numeric)
        max_future_rows: Maximum allowed future rows (0 = strict)

    Returns:
        List of error messages (empty = no leakage)

    Example:
        # Bad: Feature at T=0 uses data from T=1
        df.loc[0, 'moving_avg'] = df.loc[0:2, 'close'].mean()  # LEAKAGE!

        # Good: Feature at T=0 uses only data from T=0 and earlier
        df.loc[0, 'moving_avg'] = df.loc[0, 'close']  # OK
    """
    errors = []

    if timestamp_col not in df.columns:
        errors.append(f"Timestamp column '{timestamp_col}' not found")
        return errors

    # Check if timestamps are sorted
    if not df[timestamp_col].is_monotonic_increasing:
        errors.append("Timestamps are not monotonically increasing - potential ordering leakage")

    # Check for feature columns
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=['number']).columns.tolist()
        if timestamp_col in feature_cols:
            feature_cols.remove(timestamp_col)

    # Check for NaN patterns that suggest future peeking
    for col in feature_cols:
        # If feature has NaN at the end but not at start, might be reverse fill
        first_valid = df[col].first_valid_index()
        last_valid = df[col].last_valid_index()

        if first_valid is not None and last_valid is not None:
            if first_valid > max_future_rows:
                errors.append(
                    f"Feature '{col}' has NaN in first {first_valid} rows - "
                    f"possible future fill or windowing error"
                )

    return errors


def check_scaling_leakage(
    train_data: pd.DataFrame | np.ndarray,
    test_data: pd.DataFrame | np.ndarray,
    scaler_params: Optional[dict] = None
) -> List[str]:
    """
    Check if scaler was fit on both train and test data

    Scaling leakage occurs when StandardScaler, MinMaxScaler, etc. are fit
    on the entire dataset instead of just the training set.

    Args:
        train_data: Training data
        test_data: Test data
        scaler_params: Optional scaler parameters (mean, std, min, max)

    Returns:
        List of error messages (empty = no leakage)

    Example:
        # Bad: Scaler fit on train+test
        scaler.fit(pd.concat([train, test]))  # LEAKAGE!

        # Good: Scaler fit only on train
        scaler.fit(train)  # OK
    """
    errors = []

    # Convert to numpy if DataFrame
    if isinstance(train_data, pd.DataFrame):
        train_data = train_data.values
    if isinstance(test_data, pd.DataFrame):
        test_data = test_data.values

    # Check if test data has values outside train range (hint of separate scaling)
    train_min = train_data.min(axis=0)
    train_max = train_data.max(axis=0)

    test_min = test_data.min(axis=0)
    test_max = test_data.max(axis=0)

    # If test data is perfectly within [-1, 1] or [0, 1], might be scaled separately
    test_in_unit_range = np.all((test_min >= -1.01) & (test_max <= 1.01))
    test_in_zero_one = np.all((test_min >= -0.01) & (test_max <= 1.01))

    train_in_unit_range = np.all((train_min >= -1.01) & (train_max <= 1.01))

    if test_in_unit_range or test_in_zero_one:
        if not train_in_unit_range:
            # Test is scaled but train isn't - possible separate scaling
            errors.append(
                "Test data appears scaled to unit range but train doesn't - "
                "possible scaling leakage or incorrect scaling order"
            )

    # If scaler params provided, check if they match train stats
    if scaler_params:
        if 'mean' in scaler_params:
            expected_mean = train_data.mean(axis=0)
            provided_mean = np.array(scaler_params['mean'])

            if not np.allclose(expected_mean, provided_mean, rtol=0.01):
                errors.append(
                    "Scaler mean doesn't match training data mean - "
                    "possible fit on train+test instead of train only"
                )

        if 'std' in scaler_params:
            expected_std = train_data.std(axis=0)
            provided_std = np.array(scaler_params['std'])

            if not np.allclose(expected_std, provided_std, rtol=0.01):
                errors.append(
                    "Scaler std doesn't match training data std - "
                    "possible fit on train+test instead of train only"
                )

    return errors


def check_label_alignment(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    label_col: str = 'label',
    min_forward_periods: int = 1
) -> List[str]:
    """
    Check if labels are properly aligned with forward returns

    Ensures labels at time T use returns from T+1 onwards, not T.

    Args:
        features_df: DataFrame with features and timestamps
        labels_df: DataFrame with labels and timestamps
        timestamp_col: Name of timestamp column
        label_col: Name of label column
        min_forward_periods: Minimum forward periods for labels

    Returns:
        List of error messages (empty = no leakage)

    Example:
        # Bad: Label at T uses return from T to T
        labels[t] = returns[t]  # LEAKAGE!

        # Good: Label at T uses return from T+1 to T+5
        labels[t] = returns[t+1:t+6].sum()  # OK
    """
    errors = []

    if timestamp_col not in features_df.columns:
        errors.append(f"Timestamp column '{timestamp_col}' not found in features")
        return errors

    if timestamp_col not in labels_df.columns:
        errors.append(f"Timestamp column '{timestamp_col}' not found in labels")
        return errors

    # Merge on timestamp to align
    merged = features_df.merge(
        labels_df[[timestamp_col, label_col]],
        on=timestamp_col,
        how='inner',
        suffixes=('_feat', '_label')
    )

    if len(merged) == 0:
        errors.append("No overlapping timestamps between features and labels")
        return errors

    # Check if first N rows have labels (they shouldn't if labels are forward-looking)
    first_n_rows = min(min_forward_periods, len(merged))
    first_labels = merged[label_col].iloc[:first_n_rows]

    if not first_labels.isna().all():
        num_non_null = first_labels.notna().sum()
        errors.append(
            f"First {min_forward_periods} rows have {num_non_null} non-null labels - "
            f"labels should be NaN for first {min_forward_periods} rows if forward-looking"
        )

    # Check if last N rows have labels (they might be NaN if forward-looking)
    last_n_rows = min(min_forward_periods, len(merged))
    last_labels = merged[label_col].iloc[-last_n_rows:]

    non_null_at_end = last_labels.notna().sum()
    if non_null_at_end == last_n_rows:
        errors.append(
            f"Last {min_forward_periods} rows all have labels - "
            f"possible look-ahead bias (should have some NaN if forward returns)"
        )

    return errors


def check_data_cutoff(
    train_timestamps: pd.Series,
    test_timestamps: pd.Series
) -> List[str]:
    """
    Check if train and test data have proper temporal cutoff

    Ensures no test data appears before train data (time leakage).

    Args:
        train_timestamps: Training set timestamps
        test_timestamps: Test set timestamps

    Returns:
        List of error messages (empty = no leakage)

    Example:
        # Bad: Some test data is before some train data
        train: [2020-01-01, 2020-03-01]  # Gap!
        test:  [2020-02-01, 2020-04-01]  # LEAKAGE!

        # Good: All test data is after all train data
        train: [2020-01-01 to 2020-02-28]
        test:  [2020-03-01 to 2020-04-30]  # OK
    """
    errors = []

    if len(train_timestamps) == 0 or len(test_timestamps) == 0:
        errors.append("Empty train or test timestamps")
        return errors

    train_max = train_timestamps.max()
    test_min = test_timestamps.min()

    # Test should start after train ends
    if test_min <= train_max:
        overlap_count = (test_timestamps <= train_max).sum()
        errors.append(
            f"Test data starts before train data ends - "
            f"{overlap_count} test samples overlap with train period. "
            f"Train max: {train_max}, Test min: {test_min}"
        )

    # Check for any test samples in train period
    train_min = train_timestamps.min()

    test_in_train_period = ((test_timestamps >= train_min) & (test_timestamps <= train_max)).sum()

    if test_in_train_period > 0:
        errors.append(
            f"Found {test_in_train_period} test samples within training period - "
            f"clear temporal leakage!"
        )

    return errors


def check_time_ordering(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    group_col: Optional[str] = None
) -> List[str]:
    """
    Check if data is properly time-ordered

    Args:
        df: DataFrame to check
        timestamp_col: Name of timestamp column
        group_col: Optional grouping column (e.g., 'symbol')

    Returns:
        List of error messages (empty = no issues)
    """
    errors = []

    if timestamp_col not in df.columns:
        errors.append(f"Timestamp column '{timestamp_col}' not found")
        return errors

    if group_col and group_col in df.columns:
        # Check ordering within each group
        for group_name, group_df in df.groupby(group_col):
            if not group_df[timestamp_col].is_monotonic_increasing:
                errors.append(
                    f"Timestamps not monotonically increasing for {group_col}='{group_name}'"
                )
    else:
        # Check overall ordering
        if not df[timestamp_col].is_monotonic_increasing:
            errors.append("Timestamps not monotonically increasing")

            # Find where ordering breaks
            diffs = df[timestamp_col].diff()
            negative_diffs = diffs < pd.Timedelta(0)

            if negative_diffs.any():
                first_break = negative_diffs.idxmax()
                errors.append(
                    f"First ordering break at index {first_break}: "
                    f"{df.loc[first_break-1, timestamp_col]} -> {df.loc[first_break, timestamp_col]}"
                )

    return errors
