"""
Quality Monitor

Real-time data quality checks for market data.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional

import pandas as pd
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


class IssueSeverity(str, Enum):
    """Quality issue severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class QualityIssue:
    """Single data quality issue"""
    severity: IssueSeverity
    check_name: str
    message: str
    affected_rows: Optional[int] = None
    details: Optional[dict] = None


@dataclass
class QualityReport:
    """
    Complete data quality report

    Contains all quality check results and pass/fail status.
    """
    timestamp: datetime
    passed: bool
    has_critical_issues: bool
    issues: List[QualityIssue]

    # Summary stats
    total_checks: int
    checks_passed: int
    checks_failed: int

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "passed": self.passed,
            "has_critical_issues": self.has_critical_issues,
            "issues": [
                {
                    "severity": issue.severity.value,
                    "check": issue.check_name,
                    "message": issue.message,
                    "affected_rows": issue.affected_rows
                }
                for issue in self.issues
            ],
            "summary": {
                "total_checks": self.total_checks,
                "checks_passed": self.checks_passed,
                "checks_failed": self.checks_failed
            }
        }


class QualityMonitor:
    """
    Data Quality Monitor

    Performs real-time quality checks on market data.

    Example:
        monitor = QualityMonitor(
            max_missing_pct=0.05,
            max_price_change_pct=0.10,
            max_volume_spike_multiplier=100
        )

        report = monitor.check_quality(candles_df)

        if not report.passed:
            logger.error("quality_check_failed", issues=report.issues)

        if report.has_critical_issues:
            raise DataQualityError("Critical issues detected!")
    """

    def __init__(
        self,
        max_missing_pct: float = 0.05,
        max_price_change_pct: float = 0.10,
        max_volume_spike_multiplier: float = 100.0,
        min_coverage: float = 0.95,
        max_gap_minutes: int = 30,
        block_on_critical: bool = True
    ):
        """
        Initialize quality monitor

        Args:
            max_missing_pct: Maximum allowed missing data (5% default)
            max_price_change_pct: Maximum price change between candles (10% default)
            max_volume_spike_multiplier: Max volume spike vs median (100x default)
            min_coverage: Minimum data coverage required (95% default)
            max_gap_minutes: Maximum allowed gap in timestamps (30 min default)
            block_on_critical: If True, critical issues block training
        """
        self.max_missing_pct = max_missing_pct
        self.max_price_change_pct = max_price_change_pct
        self.max_volume_spike_multiplier = max_volume_spike_multiplier
        self.min_coverage = min_coverage
        self.max_gap_minutes = max_gap_minutes
        self.block_on_critical = block_on_critical

    def check_quality(
        self,
        df: pd.DataFrame | pl.DataFrame,
        expected_symbols: Optional[List[str]] = None
    ) -> QualityReport:
        """
        Run all quality checks

        Args:
            df: DataFrame with market data (candles)
            expected_symbols: List of expected symbols (None = skip check)

        Returns:
            QualityReport
        """
        # Convert to pandas if polars
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        logger.info(
            "running_quality_checks",
            num_rows=len(df),
            num_cols=len(df.columns)
        )

        issues = []
        checks_run = 0
        checks_passed = 0

        # Check 1: Completeness (missing data)
        checks_run += 1
        completeness_issues = self._check_completeness(df)
        if completeness_issues:
            issues.extend(completeness_issues)
        else:
            checks_passed += 1

        # Check 2: Timestamp gaps
        checks_run += 1
        gap_issues = self._check_timestamp_gaps(df)
        if gap_issues:
            issues.extend(gap_issues)
        else:
            checks_passed += 1

        # Check 3: Price sanity
        checks_run += 1
        price_issues = self._check_price_sanity(df)
        if price_issues:
            issues.extend(price_issues)
        else:
            checks_passed += 1

        # Check 4: Volume spikes
        checks_run += 1
        volume_issues = self._check_volume_spikes(df)
        if volume_issues:
            issues.extend(volume_issues)
        else:
            checks_passed += 1

        # Check 5: OHLC consistency
        checks_run += 1
        ohlc_issues = self._check_ohlc_consistency(df)
        if ohlc_issues:
            issues.extend(ohlc_issues)
        else:
            checks_passed += 1

        # Check 6: Symbol coverage (if expected symbols provided)
        if expected_symbols:
            checks_run += 1
            coverage_issues = self._check_symbol_coverage(df, expected_symbols)
            if coverage_issues:
                issues.extend(coverage_issues)
            else:
                checks_passed += 1

        # Determine overall status
        has_critical = any(issue.severity == IssueSeverity.CRITICAL for issue in issues)
        has_errors = any(issue.severity == IssueSeverity.ERROR for issue in issues)

        passed = not has_critical and not has_errors

        report = QualityReport(
            timestamp=datetime.utcnow(),
            passed=passed,
            has_critical_issues=has_critical,
            issues=issues,
            total_checks=checks_run,
            checks_passed=checks_passed,
            checks_failed=len(issues)
        )

        logger.info(
            "quality_check_complete",
            passed=passed,
            has_critical=has_critical,
            total_issues=len(issues),
            checks_passed=checks_passed,
            checks_failed=len(issues)
        )

        return report

    def _check_completeness(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check for missing data"""
        issues = []

        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = missing_count / len(df)

            if missing_pct > self.max_missing_pct:
                severity = IssueSeverity.CRITICAL if missing_pct > 0.2 else IssueSeverity.ERROR

                issues.append(QualityIssue(
                    severity=severity,
                    check_name="completeness",
                    message=f"Column '{col}' has {missing_pct*100:.1f}% missing data",
                    affected_rows=missing_count,
                    details={"missing_pct": missing_pct}
                ))

        return issues

    def _check_timestamp_gaps(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check for gaps in timestamps"""
        issues = []

        if 'timestamp' not in df.columns:
            return issues

        # Sort by timestamp
        df = df.sort_values('timestamp')

        # Calculate time differences
        time_diffs = df['timestamp'].diff()

        # Find large gaps
        max_gap_td = timedelta(minutes=self.max_gap_minutes)
        large_gaps = time_diffs > max_gap_td

        if large_gaps.any():
            num_gaps = large_gaps.sum()
            max_gap = time_diffs.max()

            severity = IssueSeverity.CRITICAL if num_gaps > 10 else IssueSeverity.WARNING

            issues.append(QualityIssue(
                severity=severity,
                check_name="timestamp_gaps",
                message=f"Found {num_gaps} timestamp gaps > {self.max_gap_minutes} minutes",
                affected_rows=num_gaps,
                details={"max_gap_minutes": max_gap.total_seconds() / 60}
            ))

        return issues

    def _check_price_sanity(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check for unreasonable price changes"""
        issues = []

        price_cols = ['open', 'high', 'low', 'close']
        available_price_cols = [col for col in price_cols if col in df.columns]

        if not available_price_cols:
            return issues

        # Check for negative or zero prices
        for col in available_price_cols:
            invalid_prices = (df[col] <= 0).sum()

            if invalid_prices > 0:
                issues.append(QualityIssue(
                    severity=IssueSeverity.CRITICAL,
                    check_name="price_sanity",
                    message=f"Found {invalid_prices} non-positive prices in '{col}'",
                    affected_rows=invalid_prices
                ))

        # Check for extreme price changes (if we have close prices)
        if 'close' in df.columns and 'symbol' in df.columns:
            for symbol in df['symbol'].unique():
                symbol_df = df[df['symbol'] == symbol].sort_values('timestamp')

                if len(symbol_df) < 2:
                    continue

                price_changes = symbol_df['close'].pct_change().abs()
                extreme_changes = price_changes > self.max_price_change_pct

                if extreme_changes.any():
                    num_extreme = extreme_changes.sum()
                    max_change = price_changes.max()

                    severity = IssueSeverity.ERROR if num_extreme > 5 else IssueSeverity.WARNING

                    issues.append(QualityIssue(
                        severity=severity,
                        check_name="price_sanity",
                        message=f"Symbol '{symbol}' has {num_extreme} price changes > {self.max_price_change_pct*100:.0f}%",
                        affected_rows=num_extreme,
                        details={"max_change_pct": max_change * 100, "symbol": symbol}
                    ))

        return issues

    def _check_volume_spikes(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check for unreasonable volume spikes"""
        issues = []

        if 'volume' not in df.columns:
            return issues

        if 'symbol' in df.columns:
            # Check per symbol
            for symbol in df['symbol'].unique():
                symbol_df = df[df['symbol'] == symbol]

                median_volume = symbol_df['volume'].median()
                if median_volume == 0:
                    continue

                volume_ratio = symbol_df['volume'] / median_volume
                spikes = volume_ratio > self.max_volume_spike_multiplier

                if spikes.any():
                    num_spikes = spikes.sum()
                    max_ratio = volume_ratio.max()

                    issues.append(QualityIssue(
                        severity=IssueSeverity.WARNING,
                        check_name="volume_spikes",
                        message=f"Symbol '{symbol}' has {num_spikes} volume spikes > {self.max_volume_spike_multiplier}x median",
                        affected_rows=num_spikes,
                        details={"max_ratio": float(max_ratio), "symbol": symbol}
                    ))
        else:
            # Check overall
            median_volume = df['volume'].median()
            if median_volume > 0:
                volume_ratio = df['volume'] / median_volume
                spikes = volume_ratio > self.max_volume_spike_multiplier

                if spikes.any():
                    num_spikes = spikes.sum()

                    issues.append(QualityIssue(
                        severity=IssueSeverity.WARNING,
                        check_name="volume_spikes",
                        message=f"Found {num_spikes} volume spikes > {self.max_volume_spike_multiplier}x median",
                        affected_rows=num_spikes
                    ))

        return issues

    def _check_ohlc_consistency(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check OHLC price consistency"""
        issues = []

        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return issues

        # High should be >= Low
        invalid_hl = df['high'] < df['low']
        if invalid_hl.any():
            issues.append(QualityIssue(
                severity=IssueSeverity.CRITICAL,
                check_name="ohlc_consistency",
                message=f"Found {invalid_hl.sum()} candles where High < Low",
                affected_rows=invalid_hl.sum()
            ))

        # High should be >= Open and Close
        invalid_h_o = df['high'] < df['open']
        invalid_h_c = df['high'] < df['close']

        if invalid_h_o.any() or invalid_h_c.any():
            total_invalid = (invalid_h_o | invalid_h_c).sum()

            issues.append(QualityIssue(
                severity=IssueSeverity.CRITICAL,
                check_name="ohlc_consistency",
                message=f"Found {total_invalid} candles where High < Open or Close",
                affected_rows=total_invalid
            ))

        # Low should be <= Open and Close
        invalid_l_o = df['low'] > df['open']
        invalid_l_c = df['low'] > df['close']

        if invalid_l_o.any() or invalid_l_c.any():
            total_invalid = (invalid_l_o | invalid_l_c).sum()

            issues.append(QualityIssue(
                severity=IssueSeverity.CRITICAL,
                check_name="ohlc_consistency",
                message=f"Found {total_invalid} candles where Low > Open or Close",
                affected_rows=total_invalid
            ))

        return issues

    def _check_symbol_coverage(
        self,
        df: pd.DataFrame,
        expected_symbols: List[str]
    ) -> List[QualityIssue]:
        """Check if all expected symbols are present"""
        issues = []

        if 'symbol' not in df.columns:
            return issues

        actual_symbols = set(df['symbol'].unique())
        expected_symbols_set = set(expected_symbols)

        missing_symbols = expected_symbols_set - actual_symbols

        if missing_symbols:
            coverage = len(actual_symbols) / len(expected_symbols_set)

            severity = IssueSeverity.CRITICAL if coverage < self.min_coverage else IssueSeverity.ERROR

            issues.append(QualityIssue(
                severity=severity,
                check_name="symbol_coverage",
                message=f"Missing {len(missing_symbols)} symbols: {sorted(missing_symbols)[:5]}...",
                details={
                    "missing_symbols": sorted(missing_symbols),
                    "coverage": coverage
                }
            ))

        return issues
