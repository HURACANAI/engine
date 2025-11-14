"""
Drift Monitor

Detects distribution drift in market data using PSI, KS tests, and other metrics.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd
import polars as pl
import structlog

from .metrics import (
    calculate_psi,
    calculate_ks_statistic,
    calculate_population_coverage,
    calculate_missingness_drift,
    calculate_correlation_drift
)

logger = structlog.get_logger(__name__)


class DriftSeverity(str, Enum):
    """Drift severity levels"""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FeatureDrift:
    """Drift metrics for a single feature"""
    feature_name: str
    psi: float
    ks_statistic: float
    ks_p_value: float
    coverage: float
    severity: DriftSeverity
    details: Dict[str, float]


@dataclass
class DriftReport:
    """
    Complete drift detection report

    Contains drift metrics for all features and overall assessment.
    """
    timestamp: datetime
    overall_severity: DriftSeverity
    critical_drift: bool  # Blocks training if True
    feature_drifts: List[FeatureDrift]
    missingness_drift: Dict[str, dict]
    correlation_drift: float

    # Summary stats
    num_features_checked: int
    num_features_drifted: int
    max_psi: float
    max_ks_statistic: float

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_severity": self.overall_severity.value,
            "critical_drift": self.critical_drift,
            "feature_drifts": [
                {
                    "feature": fd.feature_name,
                    "psi": fd.psi,
                    "ks_statistic": fd.ks_statistic,
                    "ks_p_value": fd.ks_p_value,
                    "coverage": fd.coverage,
                    "severity": fd.severity.value
                }
                for fd in self.feature_drifts
            ],
            "missingness_drift": self.missingness_drift,
            "correlation_drift": self.correlation_drift,
            "summary": {
                "num_features_checked": self.num_features_checked,
                "num_features_drifted": self.num_features_drifted,
                "max_psi": self.max_psi,
                "max_ks_statistic": self.max_ks_statistic
            }
        }


class DriftMonitor:
    """
    Dataset Drift Monitor

    Detects distribution drift using multiple statistical tests.

    Thresholds:
    - PSI < 0.1: No drift
    - 0.1 <= PSI < 0.2: Moderate drift (investigate)
    - PSI >= 0.2: Significant drift (critical)

    - KS p-value < 0.05: Significant difference
    - Coverage < 0.8: Population shift

    Example:
        monitor = DriftMonitor(
            psi_threshold_moderate=0.1,
            psi_threshold_critical=0.2
        )

        report = monitor.check_drift(
            reference_df=historical_data,
            current_df=new_data,
            features=["close", "volume", "volatility"]
        )

        if report.critical_drift:
            raise DataQualityError("Critical drift detected!")
    """

    def __init__(
        self,
        psi_threshold_moderate: float = 0.1,
        psi_threshold_critical: float = 0.2,
        ks_p_value_threshold: float = 0.05,
        coverage_threshold: float = 0.8,
        block_on_critical: bool = True
    ):
        """
        Initialize drift monitor

        Args:
            psi_threshold_moderate: PSI threshold for moderate drift
            psi_threshold_critical: PSI threshold for critical drift
            ks_p_value_threshold: KS test p-value threshold
            coverage_threshold: Minimum population coverage
            block_on_critical: If True, critical drift blocks training
        """
        self.psi_threshold_moderate = psi_threshold_moderate
        self.psi_threshold_critical = psi_threshold_critical
        self.ks_p_value_threshold = ks_p_value_threshold
        self.coverage_threshold = coverage_threshold
        self.block_on_critical = block_on_critical

    def _assess_severity(
        self,
        psi: float,
        ks_p_value: float,
        coverage: float
    ) -> DriftSeverity:
        """
        Assess drift severity based on metrics

        Args:
            psi: Population Stability Index
            ks_p_value: KS test p-value
            coverage: Population coverage

        Returns:
            DriftSeverity
        """
        # Critical drift conditions
        if psi >= self.psi_threshold_critical:
            return DriftSeverity.CRITICAL

        if coverage < 0.5:  # Severe population shift
            return DriftSeverity.CRITICAL

        # High drift conditions
        if psi >= self.psi_threshold_moderate and ks_p_value < self.ks_p_value_threshold:
            return DriftSeverity.HIGH

        if coverage < self.coverage_threshold:
            return DriftSeverity.HIGH

        # Moderate drift
        if psi >= self.psi_threshold_moderate:
            return DriftSeverity.MODERATE

        if ks_p_value < self.ks_p_value_threshold:
            return DriftSeverity.MODERATE

        # Low or no drift
        if psi >= 0.05:
            return DriftSeverity.LOW

        return DriftSeverity.NONE

    def check_drift(
        self,
        reference_df: pd.DataFrame | pl.DataFrame,
        current_df: pd.DataFrame | pl.DataFrame,
        features: Optional[List[str]] = None
    ) -> DriftReport:
        """
        Check for drift between reference and current data

        Args:
            reference_df: Reference (historical) data
            current_df: Current (new) data
            features: List of features to check (None = all numeric)

        Returns:
            DriftReport

        Raises:
            DataQualityError: If critical drift detected and block_on_critical=True
        """
        # Convert to pandas if polars
        if isinstance(reference_df, pl.DataFrame):
            reference_df = reference_df.to_pandas()
        if isinstance(current_df, pl.DataFrame):
            current_df = current_df.to_pandas()

        # Determine features to check
        if features is None:
            features = reference_df.select_dtypes(include=['number']).columns.tolist()

        logger.info(
            "checking_drift",
            num_features=len(features),
            reference_rows=len(reference_df),
            current_rows=len(current_df)
        )

        # Check drift for each feature
        feature_drifts = []

        for feature in features:
            if feature not in reference_df.columns or feature not in current_df.columns:
                logger.warning("feature_missing_skipping", feature=feature)
                continue

            # Calculate metrics
            psi = calculate_psi(reference_df[feature], current_df[feature])
            ks_stat, ks_p = calculate_ks_statistic(reference_df[feature], current_df[feature])
            coverage = calculate_population_coverage(reference_df[feature], current_df[feature])

            # Assess severity
            severity = self._assess_severity(psi, ks_p, coverage)

            feature_drift = FeatureDrift(
                feature_name=feature,
                psi=psi,
                ks_statistic=ks_stat,
                ks_p_value=ks_p,
                coverage=coverage,
                severity=severity,
                details={
                    "reference_mean": float(reference_df[feature].mean()),
                    "current_mean": float(current_df[feature].mean()),
                    "reference_std": float(reference_df[feature].std()),
                    "current_std": float(current_df[feature].std())
                }
            )

            feature_drifts.append(feature_drift)

            if severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
                logger.warning(
                    "drift_detected",
                    feature=feature,
                    severity=severity.value,
                    psi=psi,
                    ks_statistic=ks_stat,
                    coverage=coverage
                )

        # Calculate missingness drift
        missingness_drift = calculate_missingness_drift(reference_df, current_df)

        # Calculate correlation drift
        correlation_drift = calculate_correlation_drift(reference_df, current_df, features)

        # Determine overall severity
        severities = [fd.severity for fd in feature_drifts]
        if DriftSeverity.CRITICAL in severities:
            overall_severity = DriftSeverity.CRITICAL
        elif DriftSeverity.HIGH in severities:
            overall_severity = DriftSeverity.HIGH
        elif DriftSeverity.MODERATE in severities:
            overall_severity = DriftSeverity.MODERATE
        elif DriftSeverity.LOW in severities:
            overall_severity = DriftSeverity.LOW
        else:
            overall_severity = DriftSeverity.NONE

        # Critical drift check
        critical_drift = overall_severity == DriftSeverity.CRITICAL and self.block_on_critical

        # Summary stats
        num_drifted = sum(1 for fd in feature_drifts if fd.severity != DriftSeverity.NONE)
        max_psi = max([fd.psi for fd in feature_drifts]) if feature_drifts else 0.0
        max_ks = max([fd.ks_statistic for fd in feature_drifts]) if feature_drifts else 0.0

        report = DriftReport(
            timestamp=datetime.utcnow(),
            overall_severity=overall_severity,
            critical_drift=critical_drift,
            feature_drifts=feature_drifts,
            missingness_drift=missingness_drift,
            correlation_drift=correlation_drift,
            num_features_checked=len(feature_drifts),
            num_features_drifted=num_drifted,
            max_psi=max_psi,
            max_ks_statistic=max_ks
        )

        logger.info(
            "drift_check_complete",
            overall_severity=overall_severity.value,
            critical_drift=critical_drift,
            num_drifted=num_drifted,
            max_psi=max_psi
        )

        return report
