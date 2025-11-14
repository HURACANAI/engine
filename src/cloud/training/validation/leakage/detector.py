"""
Leakage Detector

Unified interface for all leakage detection checks.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional

import pandas as pd
import numpy as np
import structlog

from .checks import (
    check_feature_windowing,
    check_scaling_leakage,
    check_label_alignment,
    check_data_cutoff,
    check_time_ordering
)

logger = structlog.get_logger(__name__)


class LeakageType(str, Enum):
    """Types of data leakage"""
    FEATURE_WINDOWING = "feature_windowing"
    SCALING = "scaling"
    LABEL_ALIGNMENT = "label_alignment"
    DATA_CUTOFF = "data_cutoff"
    TIME_ORDERING = "time_ordering"


@dataclass
class LeakageIssue:
    """Single leakage issue"""
    leakage_type: LeakageType
    message: str
    severity: str  # "error", "warning"
    details: Optional[dict] = None


class LeakageDetector:
    """
    Leakage Detector

    Runs all leakage detection checks and reports issues.

    Example:
        detector = LeakageDetector(fail_fast=False)

        issues = detector.check_all(
            features_df=features,
            labels_df=labels,
            train_indices=train_idx,
            test_indices=test_idx
        )

        if issues:
            for issue in issues:
                logger.error("leakage_detected", issue=issue.message)

            if detector.has_critical_issues(issues):
                raise DataLeakageError("Critical leakage detected!")
    """

    def __init__(self, fail_fast: bool = False):
        """
        Initialize detector

        Args:
            fail_fast: If True, stop on first issue
        """
        self.fail_fast = fail_fast

    def check_all(
        self,
        features_df: pd.DataFrame,
        labels_df: Optional[pd.DataFrame] = None,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        scaler_params: Optional[dict] = None,
        timestamp_col: str = 'timestamp',
        label_col: str = 'label',
        group_col: Optional[str] = None
    ) -> List[LeakageIssue]:
        """
        Run all leakage checks

        Args:
            features_df: DataFrame with features
            labels_df: Optional DataFrame with labels
            train_indices: Optional training indices
            test_indices: Optional test indices
            scaler_params: Optional scaler parameters
            timestamp_col: Name of timestamp column
            label_col: Name of label column
            group_col: Optional grouping column (e.g., 'symbol')

        Returns:
            List of LeakageIssue objects
        """
        logger.info(
            "running_leakage_checks",
            num_features=len(features_df),
            has_labels=labels_df is not None,
            has_train_test_split=train_indices is not None
        )

        issues = []

        # Check 1: Time ordering
        time_errors = check_time_ordering(
            features_df,
            timestamp_col=timestamp_col,
            group_col=group_col
        )

        for error in time_errors:
            issues.append(LeakageIssue(
                leakage_type=LeakageType.TIME_ORDERING,
                message=error,
                severity="error"
            ))

        if self.fail_fast and issues:
            return issues

        # Check 2: Feature windowing
        feature_errors = check_feature_windowing(
            features_df,
            timestamp_col=timestamp_col
        )

        for error in feature_errors:
            issues.append(LeakageIssue(
                leakage_type=LeakageType.FEATURE_WINDOWING,
                message=error,
                severity="warning"  # Might be intentional
            ))

        if self.fail_fast and issues:
            return issues

        # Check 3: Label alignment (if labels provided)
        if labels_df is not None:
            label_errors = check_label_alignment(
                features_df,
                labels_df,
                timestamp_col=timestamp_col,
                label_col=label_col
            )

            for error in label_errors:
                issues.append(LeakageIssue(
                    leakage_type=LeakageType.LABEL_ALIGNMENT,
                    message=error,
                    severity="error"
                ))

            if self.fail_fast and issues:
                return issues

        # Check 4: Data cutoff (if train/test split provided)
        if train_indices is not None and test_indices is not None:
            if timestamp_col in features_df.columns:
                train_timestamps = features_df.iloc[train_indices][timestamp_col]
                test_timestamps = features_df.iloc[test_indices][timestamp_col]

                cutoff_errors = check_data_cutoff(train_timestamps, test_timestamps)

                for error in cutoff_errors:
                    issues.append(LeakageIssue(
                        leakage_type=LeakageType.DATA_CUTOFF,
                        message=error,
                        severity="error"
                    ))

                if self.fail_fast and issues:
                    return issues

        # Check 5: Scaling leakage (if scaler params provided or train/test split)
        if train_indices is not None and test_indices is not None:
            # Get numeric columns only
            numeric_cols = features_df.select_dtypes(include=['number']).columns.tolist()
            if timestamp_col in numeric_cols:
                numeric_cols.remove(timestamp_col)

            if numeric_cols:
                train_data = features_df.iloc[train_indices][numeric_cols]
                test_data = features_df.iloc[test_indices][numeric_cols]

                scaling_errors = check_scaling_leakage(
                    train_data,
                    test_data,
                    scaler_params=scaler_params
                )

                for error in scaling_errors:
                    issues.append(LeakageIssue(
                        leakage_type=LeakageType.SCALING,
                        message=error,
                        severity="error"
                    ))

        logger.info(
            "leakage_checks_complete",
            total_issues=len(issues),
            errors=sum(1 for i in issues if i.severity == "error"),
            warnings=sum(1 for i in issues if i.severity == "warning")
        )

        return issues

    def has_critical_issues(self, issues: List[LeakageIssue]) -> bool:
        """
        Check if any issues are critical (errors, not warnings)

        Args:
            issues: List of LeakageIssue objects

        Returns:
            True if any critical issues found
        """
        return any(issue.severity == "error" for issue in issues)

    def generate_report(self, issues: List[LeakageIssue]) -> str:
        """
        Generate human-readable report

        Args:
            issues: List of LeakageIssue objects

        Returns:
            Report string
        """
        if not issues:
            return "✅ No leakage detected - all checks passed!"

        report_lines = [
            "=" * 80,
            "DATA LEAKAGE DETECTION REPORT",
            "=" * 80,
            "",
            f"Total Issues: {len(issues)}",
            f"Errors: {sum(1 for i in issues if i.severity == 'error')}",
            f"Warnings: {sum(1 for i in issues if i.severity == 'warning')}",
            "",
            "ISSUES:",
            "-" * 80,
        ]

        # Group by type
        by_type = {}
        for issue in issues:
            if issue.leakage_type not in by_type:
                by_type[issue.leakage_type] = []
            by_type[issue.leakage_type].append(issue)

        for leakage_type, type_issues in by_type.items():
            report_lines.append(f"\n{leakage_type.value.upper()}:")

            for issue in type_issues:
                severity_icon = "❌" if issue.severity == "error" else "⚠️"
                report_lines.append(f"  {severity_icon} {issue.message}")

        report_lines.append("")
        report_lines.append("=" * 80)

        return "\n".join(report_lines)
