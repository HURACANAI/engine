"""
Dataset Drift & Quality Monitor

Detects distribution drift, data quality issues, and anomalies in market data.

Key Features:
- PSI (Population Stability Index) for drift detection
- KS (Kolmogorov-Smirnov) tests for distribution changes
- Data quality checks (completeness, sanity, coverage)
- Regime-specific drift tracking
- Critical issue blocking

Usage:
    from src.cloud.training.datasets.drift import DriftMonitor, QualityMonitor

    # Drift detection
    drift_monitor = DriftMonitor()
    drift_report = drift_monitor.check_drift(
        reference_df=historical_data,
        current_df=new_data,
        features=["close", "volume", "volatility"]
    )

    if drift_report.critical_drift:
        raise DataQualityError("Critical drift detected!")

    # Quality checks
    quality_monitor = QualityMonitor()
    quality_report = quality_monitor.check_quality(candles_df)

    if not quality_report.passed:
        raise DataQualityError("Quality check failed!")
"""

from .drift_monitor import DriftMonitor, DriftReport, DriftSeverity
from .quality_monitor import QualityMonitor, QualityReport, QualityIssue
from .metrics import calculate_psi, calculate_ks_statistic

__all__ = [
    "DriftMonitor",
    "DriftReport",
    "DriftSeverity",
    "QualityMonitor",
    "QualityReport",
    "QualityIssue",
    "calculate_psi",
    "calculate_ks_statistic",
]
