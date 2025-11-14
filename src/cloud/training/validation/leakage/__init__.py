"""
Leakage & Look-Ahead Guards

Prevents future data from leaking into training through:
- Feature windowing validation
- Scaling leakage detection
- Label alignment checks
- Data cutoff verification

Critical for model validity - catches bugs that inflate performance.

Usage:
    from src.cloud.training.validation.leakage import LeakageDetector

    detector = LeakageDetector()

    # Check for leakage
    issues = detector.check_all(
        features_df=features,
        labels_df=labels,
        train_indices=train_idx,
        test_indices=test_idx
    )

    if issues:
        for issue in issues:
            print(f"LEAKAGE: {issue.message}")
        raise DataLeakageError("Data leakage detected!")
"""

from .detector import LeakageDetector, LeakageIssue, LeakageType
from .checks import (
    check_feature_windowing,
    check_scaling_leakage,
    check_label_alignment,
    check_data_cutoff,
    check_time_ordering
)

__all__ = [
    "LeakageDetector",
    "LeakageIssue",
    "LeakageType",
    "check_feature_windowing",
    "check_scaling_leakage",
    "check_label_alignment",
    "check_data_cutoff",
    "check_time_ordering",
]
