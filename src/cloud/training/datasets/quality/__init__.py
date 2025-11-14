"""
Data Quality Monitor

Real-time data quality checks for market data.

Checks:
- Completeness: Missing candles, gaps in timestamps
- Sanity: Price within reasonable bounds, volume spikes
- Consistency: OHLC relationships, monotonic time
- Coverage: All expected symbols present

Usage:
    from src.cloud.training.datasets.quality import QualityMonitor

    monitor = QualityMonitor()
    report = monitor.check_quality(candles_df)

    if not report.passed:
        for issue in report.issues:
            print(f"{issue.severity}: {issue.message}")

    if report.has_critical_issues:
        raise DataQualityError("Critical quality issues!")
"""

from .quality_monitor import QualityMonitor, QualityReport, QualityIssue, IssueSeverity

__all__ = [
    "QualityMonitor",
    "QualityReport",
    "QualityIssue",
    "IssueSeverity",
]
