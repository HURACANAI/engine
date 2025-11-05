"""
Purged Walk-Forward Cross-Validation

Prevents leakage in financial time series by:
1. Purging train samples that overlap with test
2. Adding embargo periods
3. Combinatorial splits for robust OOS estimates

Based on: "Advances in Financial Machine Learning" by Marcos López de Prado
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PurgedFold:
    """A single purged fold."""

    train_indices: np.ndarray
    test_indices: np.ndarray
    purged_count: int  # Number of samples purged
    embargo_count: int  # Number of samples embargoed


class CombinatorialPurgedKFold:
    """
    Combinatorial Purged K-Fold Cross-Validation.

    Key Features:
    1. **Purging**: Remove train samples that overlap with test period
    2. **Embargo**: Remove train samples immediately after test period
    3. **Combinatorial**: Test multiple combinations of test folds

    Why This Matters:
    - Standard K-fold leaks information in financial data
    - Overlapping labels (triple-barrier) span multiple bars
    - This prevents looking into the future

    Example:
        # Standard K-fold (WRONG - leaks!)
        Train: [0-799]  Test: [800-999]
        ↑ Train sample at bar 795 has label from bar 850 (in test!)

        # Purged K-fold (CORRECT)
        Train: [0-740]  Purged: [741-799]  Test: [800-999]  Embargo: [1000-1020]
        ↑ No train sample uses info from test period

    Usage:
        cv = CombinatorialPurgedKFold(
            n_splits=5,
            n_test_splits=2,
            embargo_pct=0.01,
        )

        # Get trade event times (when labels are known)
        pred_times = pd.Series(
            index=range(len(X)),
            data=[bar_to_timestamp(i) for i in range(len(X))]
        )

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, pred_times)):
            logger.info(f"Fold {fold_idx}: train={len(train_idx)}, test={len(test_idx)}")

            # Train
            model.fit(X[train_idx], y[train_idx])

            # Test
            y_pred = model.predict(X[test_idx])
            oos_score = metric(y[test_idx], y_pred)

        # BLOCK deployment if OOS too low
        if np.mean(oos_scores) < threshold:
            raise ValueError("OOS performance insufficient - model overfitting!")
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        embargo_pct: float = 0.01,
    ):
        """
        Initialize purged CV.

        Args:
            n_splits: Number of splits (K)
            n_test_splits: Number of splits used for testing in each combination
            embargo_pct: Percentage of samples to embargo after test
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_pct = embargo_pct

        logger.info(
            "purged_cv_initialized",
            n_splits=n_splits,
            n_test_splits=n_test_splits,
            embargo_pct=embargo_pct,
        )

    def split(
        self,
        X: np.ndarray,
        pred_times: pd.Series,
        eval_times: pd.Series = None,
    ):
        """
        Generate purged train/test splits.

        Args:
            X: Feature matrix
            pred_times: Series with index=sample_idx, values=timestamp when prediction made
            eval_times: Series with index=sample_idx, values=timestamp when outcome known
                       (if None, assumes eval_time = pred_time + avg_holding_period)

        Yields:
            (train_indices, test_indices) tuples
        """
        if eval_times is None:
            # Estimate eval times (when label is known)
            # Assume average holding period of 50 bars
            avg_hold = 50
            eval_times = pred_times + pd.Timedelta(bars=avg_hold)

        n_samples = len(X)

        # Generate test sets
        test_splits = np.array_split(np.arange(n_samples), self.n_splits)

        # Combinatorial: test on multiple combinations
        from itertools import combinations

        test_combinations = list(combinations(range(self.n_splits), self.n_test_splits))

        logger.info(
            "generating_folds",
            n_samples=n_samples,
            n_test_combinations=len(test_combinations),
        )

        for combo_idx, test_split_indices in enumerate(test_combinations):
            # Combine test splits
            test_indices = np.concatenate([test_splits[i] for i in test_split_indices])

            # Get test time range
            test_times = pred_times.iloc[test_indices]
            t0_test = test_times.min()
            t1_test = test_times.max()

            # Get eval time range (when test outcomes known)
            test_eval_times = eval_times.iloc[test_indices]
            t1_eval = test_eval_times.max()

            # Embargo: add period after eval
            embargo_samples = int(n_samples * self.embargo_pct)
            t1_embargo = t1_eval + pd.Timedelta(bars=embargo_samples)

            # Train indices: all samples NOT in test
            train_indices = np.setdiff1d(np.arange(n_samples), test_indices)

            # PURGE: Remove train samples that overlap with test
            # A train sample at time t is removed if its eval_time overlaps [t0_test, t1_test]
            purge_mask = np.zeros(len(train_indices), dtype=bool)

            for i, train_idx in enumerate(train_indices):
                train_pred_time = pred_times.iloc[train_idx]
                train_eval_time = eval_times.iloc[train_idx]

                # Check if this train sample's outcome period overlaps with test
                if train_pred_time <= t1_test and train_eval_time >= t0_test:
                    purge_mask[i] = True

            # EMBARGO: Remove train samples immediately after test
            embargo_mask = np.zeros(len(train_indices), dtype=bool)

            for i, train_idx in enumerate(train_indices):
                train_pred_time = pred_times.iloc[train_idx]

                # Check if in embargo period
                if t1_eval <= train_pred_time <= t1_embargo:
                    embargo_mask[i] = True

            # Remove purged and embargoed samples
            keep_mask = ~(purge_mask | embargo_mask)
            train_indices_clean = train_indices[keep_mask]

            purged_count = purge_mask.sum()
            embargo_count = embargo_mask.sum()

            logger.info(
                "fold_generated",
                combo=combo_idx,
                train=len(train_indices_clean),
                test=len(test_indices),
                purged=purged_count,
                embargoed=embargo_count,
            )

            yield train_indices_clean, test_indices


def require_oos_pass(
    oos_scores: List[float],
    threshold: float,
    metric_name: str = "score",
) -> None:
    """
    Require OOS performance to pass threshold before deployment.

    Args:
        oos_scores: List of OOS scores from CV folds
        threshold: Minimum acceptable OOS score
        metric_name: Name of metric

    Raises:
        ValueError: If OOS performance insufficient
    """
    mean_oos = np.mean(oos_scores)
    std_oos = np.std(oos_scores)
    min_oos = np.min(oos_scores)

    logger.info(
        "oos_performance_check",
        mean=mean_oos,
        std=std_oos,
        min=min_oos,
        threshold=threshold,
    )

    if mean_oos < threshold:
        raise ValueError(
            f"OOS {metric_name} insufficient: {mean_oos:.4f} < {threshold:.4f}. "
            f"Model is likely overfitting. Do not deploy!"
        )

    if min_oos < threshold * 0.8:  # Any fold < 80% of threshold
        logger.warning(
            "oos_fold_concern",
            min_oos=min_oos,
            threshold_80pct=threshold * 0.8,
        )

    logger.info(
        "oos_pass",
        mean=mean_oos,
        passes_threshold=True,
    )
