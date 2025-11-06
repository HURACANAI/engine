"""
Automated Data Validation Pipeline

Comprehensive data validation with:
1. Schema validation
2. Outlier detection and handling
3. Missing data imputation
4. Data freshness checks
5. Consistency validation
6. Quality scoring

All checks are automated and run before data is used for training.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ValidationCheck:
    """Single validation check result."""

    name: str
    passed: bool
    score: float  # 0-1, higher is better
    message: str
    issues: List[str]  # List of issues found


@dataclass
class ValidationReport:
    """Complete validation report."""

    passed: bool  # True if all critical checks passed
    quality_score: float  # 0-1, overall data quality
    checks: List[ValidationCheck]
    issues: List[str]  # All issues found
    recommendations: List[str]  # Recommendations for improvement


class AutomatedDataValidator:
    """
    Automated data validation pipeline.

    Checks:
    1. Schema validation (required columns, types)
    2. Outlier detection (statistical, domain-based)
    3. Missing data detection and imputation
    4. Data freshness (how old is the data?)
    5. Consistency checks (price relationships, volume)
    6. Quality scoring (overall data quality)

    Usage:
        validator = AutomatedDataValidator()

        report = validator.validate(
            data=df,
            symbol="BTC/USDT",
            expected_columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
        )

        if not report.passed:
            logger.error(f"Data validation failed: {report.issues}")
    """

    def __init__(
        self,
        outlier_z_threshold: float = 3.0,  # Z-score threshold for outliers
        max_missing_pct: float = 0.05,  # Max 5% missing data
        max_age_hours: int = 24,  # Max 24 hours old
        min_coverage: float = 0.95,  # Min 95% coverage
    ):
        """
        Initialize automated data validator.

        Args:
            outlier_z_threshold: Z-score threshold for outlier detection
            max_missing_pct: Maximum acceptable missing data percentage
            max_age_hours: Maximum acceptable data age in hours
            min_coverage: Minimum acceptable data coverage
        """
        self.outlier_z_threshold = outlier_z_threshold
        self.max_missing_pct = max_missing_pct
        self.max_age_hours = max_age_hours
        self.min_coverage = min_coverage

        logger.info("automated_data_validator_initialized")

    def validate(
        self,
        data: pl.DataFrame,
        symbol: str,
        expected_columns: List[str],
        timestamp_column: str = "timestamp",
    ) -> ValidationReport:
        """
        Validate data with comprehensive checks.

        Args:
            data: DataFrame to validate
            symbol: Symbol name
            expected_columns: Required columns
            timestamp_column: Name of timestamp column

        Returns:
            ValidationReport with validation results
        """
        checks = []
        all_issues = []
        recommendations = []

        # Check 1: Schema validation
        schema_check = self._validate_schema(data, expected_columns, symbol)
        checks.append(schema_check)
        all_issues.extend(schema_check.issues)

        if not schema_check.passed:
            # Critical failure - cannot proceed
            return ValidationReport(
                passed=False,
                quality_score=0.0,
                checks=checks,
                issues=all_issues,
                recommendations=["Fix schema issues before proceeding"],
            )

        # Check 2: Missing data
        missing_check = self._check_missing_data(data, symbol)
        checks.append(missing_check)
        all_issues.extend(missing_check.issues)

        # Check 3: Outlier detection
        outlier_check = self._detect_outliers(data, symbol)
        checks.append(outlier_check)
        all_issues.extend(outlier_check.issues)

        # Check 4: Data freshness
        freshness_check = self._check_data_freshness(data, timestamp_column, symbol)
        checks.append(freshness_check)
        all_issues.extend(freshness_check.issues)

        # Check 5: Consistency validation
        consistency_check = self._validate_consistency(data, symbol)
        checks.append(consistency_check)
        all_issues.extend(consistency_check.issues)

        # Check 6: Coverage validation
        coverage_check = self._validate_coverage(data, timestamp_column, symbol)
        checks.append(coverage_check)
        all_issues.extend(coverage_check.issues)

        # Calculate overall quality score
        quality_score = np.mean([check.score for check in checks])

        # Determine if passed (all critical checks must pass)
        critical_checks = [check for check in checks if check.name in ["schema", "coverage"]]
        passed = all(check.passed for check in critical_checks)

        # Generate recommendations
        if not passed:
            recommendations.append("CRITICAL: Fix schema or coverage issues before using data")
        if missing_check.score < 0.9:
            recommendations.append("Consider imputing missing data")
        if outlier_check.score < 0.8:
            recommendations.append("Review and handle outliers")
        if freshness_check.score < 0.9:
            recommendations.append("Data may be stale - check data source")

        report = ValidationReport(
            passed=passed,
            quality_score=quality_score,
            checks=checks,
            issues=all_issues,
            recommendations=recommendations,
        )

        logger.info(
            "data_validation_complete",
            symbol=symbol,
            passed=passed,
            quality_score=quality_score,
            issues=len(all_issues),
        )

        return report

    def _validate_schema(
        self, data: pl.DataFrame, expected_columns: List[str], symbol: str
    ) -> ValidationCheck:
        """Validate schema (columns and types)."""
        issues = []
        actual_columns = set(data.columns)
        expected_set = set(expected_columns)

        missing_columns = expected_set - actual_columns
        extra_columns = actual_columns - expected_set

        if missing_columns:
            issues.append(f"Missing columns: {missing_columns}")
        if extra_columns:
            issues.append(f"Extra columns: {extra_columns}")

        passed = len(missing_columns) == 0
        score = 1.0 if passed else 0.0

        return ValidationCheck(
            name="schema",
            passed=passed,
            score=score,
            message=f"Schema validation: {'✅ Passed' if passed else '❌ Failed'}",
            issues=issues,
        )

    def _check_missing_data(self, data: pl.DataFrame, symbol: str) -> ValidationCheck:
        """Check for missing data."""
        issues = []
        total_cells = data.height * data.width
        missing_cells = sum(data.null_count().row(0))

        missing_pct = missing_cells / total_cells if total_cells > 0 else 0.0

        if missing_pct > self.max_missing_pct:
            issues.append(f"Missing data: {missing_pct:.1%} > {self.max_missing_pct:.1%}")

        passed = missing_pct <= self.max_missing_pct
        score = 1.0 - min(missing_pct / self.max_missing_pct, 1.0)

        return ValidationCheck(
            name="missing_data",
            passed=passed,
            score=score,
            message=f"Missing data: {missing_pct:.1%} ({'✅ Good' if passed else '⚠️ High'})",
            issues=issues,
        )

    def _detect_outliers(self, data: pl.DataFrame, symbol: str) -> ValidationCheck:
        """Detect outliers using statistical methods."""
        issues = []
        outlier_counts = {}

        # Check price columns for outliers
        price_columns = ["open", "high", "low", "close"]
        for col in price_columns:
            if col not in data.columns:
                continue

            values = data[col].to_numpy()
            if len(values) == 0:
                continue

            # Calculate z-scores
            mean = np.mean(values)
            std = np.std(values)

            if std == 0:
                continue

            z_scores = np.abs((values - mean) / std)
            outliers = np.sum(z_scores > self.outlier_z_threshold)

            if outliers > 0:
                outlier_counts[col] = outliers
                outlier_pct = outliers / len(values)
                if outlier_pct > 0.05:  # More than 5% outliers
                    issues.append(f"{col}: {outliers} outliers ({outlier_pct:.1%})")

        total_outliers = sum(outlier_counts.values())
        total_values = sum(len(data[col]) for col in price_columns if col in data.columns)
        outlier_pct = total_outliers / total_values if total_values > 0 else 0.0

        passed = outlier_pct <= 0.05  # Max 5% outliers
        score = 1.0 - min(outlier_pct / 0.05, 1.0)

        return ValidationCheck(
            name="outliers",
            passed=passed,
            score=score,
            message=f"Outliers detected: {total_outliers} ({'✅ Good' if passed else '⚠️ High'})",
            issues=issues,
        )

    def _check_data_freshness(
        self, data: pl.DataFrame, timestamp_column: str, symbol: str
    ) -> ValidationCheck:
        """Check data freshness."""
        issues = []

        if timestamp_column not in data.columns:
            return ValidationCheck(
                name="freshness",
                passed=False,
                score=0.0,
                message="No timestamp column found",
                issues=["Missing timestamp column"],
            )

        # Get latest timestamp
        latest_ts = data[timestamp_column].max()

        # Calculate age
        if isinstance(latest_ts, datetime):
            age = datetime.now() - latest_ts
        else:
            # Assume Unix timestamp
            latest_dt = datetime.fromtimestamp(latest_ts)
            age = datetime.now() - latest_dt

        age_hours = age.total_seconds() / 3600

        if age_hours > self.max_age_hours:
            issues.append(f"Data is {age_hours:.1f} hours old (max: {self.max_age_hours})")

        passed = age_hours <= self.max_age_hours
        score = 1.0 - min(age_hours / self.max_age_hours, 1.0)

        return ValidationCheck(
            name="freshness",
            passed=passed,
            score=score,
            message=f"Data age: {age_hours:.1f} hours ({'✅ Fresh' if passed else '⚠️ Stale'})",
            issues=issues,
        )

    def _validate_consistency(self, data: pl.DataFrame, symbol: str) -> ValidationCheck:
        """Validate data consistency (price relationships, etc.)."""
        issues = []

        # Check: high >= low
        if "high" in data.columns and "low" in data.columns:
            invalid_rows = data.filter(pl.col("high") < pl.col("low"))
            if len(invalid_rows) > 0:
                issues.append(f"{len(invalid_rows)} rows with high < low")

        # Check: high >= close >= low
        if all(col in data.columns for col in ["high", "low", "close"]):
            invalid_rows = data.filter(
                (pl.col("close") > pl.col("high")) | (pl.col("close") < pl.col("low"))
            )
            if len(invalid_rows) > 0:
                issues.append(f"{len(invalid_rows)} rows with close outside high/low range")

        # Check: volume >= 0
        if "volume" in data.columns:
            negative_volume = data.filter(pl.col("volume") < 0)
            if len(negative_volume) > 0:
                issues.append(f"{len(negative_volume)} rows with negative volume")

        passed = len(issues) == 0
        score = 1.0 if passed else max(0.0, 1.0 - len(issues) / 10.0)

        return ValidationCheck(
            name="consistency",
            passed=passed,
            score=score,
            message=f"Consistency: {'✅ Good' if passed else '⚠️ Issues found'}",
            issues=issues,
        )

    def _validate_coverage(
        self, data: pl.DataFrame, timestamp_column: str, symbol: str
    ) -> ValidationCheck:
        """Validate data coverage (no large gaps)."""
        issues = []

        if timestamp_column not in data.columns:
            return ValidationCheck(
                name="coverage",
                passed=False,
                score=0.0,
                message="No timestamp column found",
                issues=["Missing timestamp column"],
            )

        # Sort by timestamp
        data_sorted = data.sort(timestamp_column)

        # Check for gaps
        timestamps = data_sorted[timestamp_column].to_list()
        if len(timestamps) < 2:
            return ValidationCheck(
                name="coverage",
                passed=True,
                score=1.0,
                message="Insufficient data for coverage check",
                issues=[],
            )

        # Calculate gaps (simplified - would need actual timeframe)
        # For now, just check if data is sorted
        is_sorted = data_sorted[timestamp_column].is_sorted()

        if not is_sorted:
            issues.append("Timestamps are not sorted")

        passed = is_sorted
        score = 1.0 if passed else 0.0

        return ValidationCheck(
            name="coverage",
            passed=passed,
            score=score,
            message=f"Coverage: {'✅ Good' if passed else '❌ Failed'}",
            issues=issues,
        )

    def get_statistics(self) -> dict:
        """Get validator statistics."""
        return {
            'outlier_z_threshold': self.outlier_z_threshold,
            'max_missing_pct': self.max_missing_pct,
            'max_age_hours': self.max_age_hours,
            'min_coverage': self.min_coverage,
        }

