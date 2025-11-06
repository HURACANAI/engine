"""
Drift Detector

Detects distribution shifts using statistical tests:
1. KS Test (Kolmogorov-Smirnov) - Distribution changes
2. PSI (Population Stability Index) - Feature drift
3. Chi-Square - Categorical changes
4. Custom thresholds - Domain-specific drift

When drift exceeds thresholds:
- WARNING: Monitor closely
- CRITICAL: Trigger retrain
- SEVERE: Pause trading
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import polars as pl
import structlog
from scipy import stats

logger = structlog.get_logger(__name__)


class DriftSeverity(Enum):
    """Drift severity levels."""
    NONE = "none"
    WARNING = "warning"
    CRITICAL = "critical"
    SEVERE = "severe"


@dataclass
class DriftMetrics:
    """Metrics from drift detection."""

    # KS test results (continuous distributions)
    ks_statistic: float
    ks_pvalue: float
    ks_drifted: bool

    # PSI (Population Stability Index)
    psi_score: float
    psi_drifted: bool

    # Label distribution drift
    label_dist_current: Dict[str, float]
    label_dist_reference: Dict[str, float]
    label_drift_score: float
    label_drifted: bool

    # Cost drift
    cost_current_mean: float
    cost_reference_mean: float
    cost_drift_pct: float
    cost_drifted: bool

    # Overall
    overall_severity: DriftSeverity
    timestamp: datetime


class DriftDetector:
    """
    Statistical drift detection for trading data.

    Usage:
        detector = DriftDetector(
            ks_threshold=0.05,
            psi_threshold=0.1,
            label_drift_threshold=0.1,
            cost_drift_threshold=0.2
        )

        # Compare current vs reference
        metrics = detector.detect(
            current_data=recent_candles,
            reference_data=baseline_candles,
            current_labels=recent_labels,
            reference_labels=baseline_labels
        )

        if metrics.overall_severity == DriftSeverity.CRITICAL:
            trigger_retrain()
        elif metrics.overall_severity == DriftSeverity.SEVERE:
            pause_trading()
    """

    def __init__(
        self,
        ks_threshold: float = 0.05,       # KS test p-value
        psi_threshold: float = 0.1,       # PSI threshold (0.1 = 10% drift)
        label_drift_threshold: float = 0.1,  # 10% change in label distribution
        cost_drift_threshold: float = 0.2    # 20% change in costs
    ):
        """
        Initialize drift detector.

        Args:
            ks_threshold: KS test p-value threshold (lower = more sensitive)
            psi_threshold: PSI threshold (higher = less sensitive)
            label_drift_threshold: Label distribution change threshold
            cost_drift_threshold: Cost change threshold
        """
        self.ks_threshold = ks_threshold
        self.psi_threshold = psi_threshold
        self.label_drift_threshold = label_drift_threshold
        self.cost_drift_threshold = cost_drift_threshold

        logger.info(
            "drift_detector_initialized",
            ks_threshold=ks_threshold,
            psi_threshold=psi_threshold,
            label_drift_threshold=label_drift_threshold,
            cost_drift_threshold=cost_drift_threshold
        )

    def detect(
        self,
        current_data: pl.DataFrame,
        reference_data: pl.DataFrame,
        current_labels: Optional[List] = None,
        reference_labels: Optional[List] = None
    ) -> DriftMetrics:
        """
        Detect drift between current and reference data.

        Args:
            current_data: Recent candle data
            reference_data: Baseline/reference candle data
            current_labels: Recent labels (optional)
            reference_labels: Reference labels (optional)

        Returns:
            DriftMetrics with detection results
        """
        logger.info(
            "drift_detection_start",
            current_rows=len(current_data),
            reference_rows=len(reference_data)
        )

        # 1. KS Test on returns distribution
        ks_stat, ks_pval, ks_drift = self._ks_test_returns(
            current_data, reference_data
        )

        # 2. PSI on price distribution
        psi_score, psi_drift = self._psi_test_prices(
            current_data, reference_data
        )

        # 3. Label distribution drift
        label_dist_current = {}
        label_dist_ref = {}
        label_drift_score = 0.0
        label_drift = False

        if current_labels and reference_labels:
            label_dist_current, label_dist_ref, label_drift_score, label_drift = \
                self._label_distribution_drift(current_labels, reference_labels)

        # 4. Cost drift
        cost_current = 0.0
        cost_ref = 0.0
        cost_drift_pct = 0.0
        cost_drift = False

        if current_labels and reference_labels:
            cost_current, cost_ref, cost_drift_pct, cost_drift = \
                self._cost_drift(current_labels, reference_labels)

        # 5. Determine overall severity
        severity = self._determine_severity(
            ks_drift, psi_drift, label_drift, cost_drift
        )

        metrics = DriftMetrics(
            ks_statistic=ks_stat,
            ks_pvalue=ks_pval,
            ks_drifted=ks_drift,
            psi_score=psi_score,
            psi_drifted=psi_drift,
            label_dist_current=label_dist_current,
            label_dist_reference=label_dist_ref,
            label_drift_score=label_drift_score,
            label_drifted=label_drift,
            cost_current_mean=cost_current,
            cost_reference_mean=cost_ref,
            cost_drift_pct=cost_drift_pct,
            cost_drifted=cost_drift,
            overall_severity=severity,
            timestamp=datetime.now()
        )

        logger.info(
            "drift_detection_complete",
            ks_drifted=ks_drift,
            psi_drifted=psi_drift,
            label_drifted=label_drift,
            cost_drifted=cost_drift,
            severity=severity.value
        )

        return metrics

    def _ks_test_returns(
        self,
        current_data: pl.DataFrame,
        reference_data: pl.DataFrame
    ) -> tuple[float, float, bool]:
        """
        KS test on returns distribution.

        Returns:
            (statistic, p-value, is_drifted)
        """
        # Calculate returns
        current_returns = current_data.with_columns(
            pl.col('close').pct_change().alias('returns')
        )['returns'].drop_nulls().to_numpy()

        ref_returns = reference_data.with_columns(
            pl.col('close').pct_change().alias('returns')
        )['returns'].drop_nulls().to_numpy()

        # KS test
        ks_stat, ks_pval = stats.ks_2samp(current_returns, ref_returns)

        is_drifted = ks_pval < self.ks_threshold

        logger.debug(
            "ks_test_returns",
            statistic=ks_stat,
            pvalue=ks_pval,
            drifted=is_drifted
        )

        return float(ks_stat), float(ks_pval), is_drifted

    def _psi_test_prices(
        self,
        current_data: pl.DataFrame,
        reference_data: pl.DataFrame
    ) -> tuple[float, bool]:
        """
        PSI (Population Stability Index) on price distribution.

        PSI formula:
        PSI = Σ (actual% - expected%) × ln(actual% / expected%)

        PSI < 0.1: No significant change
        PSI 0.1-0.2: Moderate change
        PSI > 0.2: Significant change

        Returns:
            (psi_score, is_drifted)
        """
        current_prices = current_data['close'].to_numpy()
        ref_prices = reference_data['close'].to_numpy()

        # Create bins based on reference quantiles
        quantiles = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        bins = np.quantile(ref_prices, quantiles)

        # Histogram counts
        current_counts, _ = np.histogram(current_prices, bins=bins)
        ref_counts, _ = np.histogram(ref_prices, bins=bins)

        # Proportions (add small epsilon to avoid log(0))
        eps = 1e-10
        current_props = (current_counts + eps) / (current_counts.sum() + eps * len(bins))
        ref_props = (ref_counts + eps) / (ref_counts.sum() + eps * len(bins))

        # PSI calculation
        psi = np.sum((current_props - ref_props) * np.log(current_props / ref_props))

        is_drifted = psi > self.psi_threshold

        logger.debug(
            "psi_test_prices",
            psi_score=psi,
            drifted=is_drifted
        )

        return float(psi), is_drifted

    def _label_distribution_drift(
        self,
        current_labels: List,
        reference_labels: List
    ) -> tuple[Dict, Dict, float, bool]:
        """
        Detect drift in label distribution.

        Returns:
            (current_dist, ref_dist, drift_score, is_drifted)
        """
        # Get meta-label distributions
        current_profitable = sum(1 for t in current_labels if t.meta_label == 1)
        current_total = len(current_labels)
        current_pct = current_profitable / current_total if current_total > 0 else 0

        ref_profitable = sum(1 for t in reference_labels if t.meta_label == 1)
        ref_total = len(reference_labels)
        ref_pct = ref_profitable / ref_total if ref_total > 0 else 0

        current_dist = {
            'profitable': current_pct,
            'unprofitable': 1 - current_pct
        }

        ref_dist = {
            'profitable': ref_pct,
            'unprofitable': 1 - ref_pct
        }

        # Drift score = absolute difference in profitable %
        drift_score = abs(current_pct - ref_pct)

        is_drifted = drift_score > self.label_drift_threshold

        logger.debug(
            "label_distribution_drift",
            current_profitable_pct=current_pct,
            ref_profitable_pct=ref_pct,
            drift_score=drift_score,
            drifted=is_drifted
        )

        return current_dist, ref_dist, float(drift_score), is_drifted

    def _cost_drift(
        self,
        current_labels: List,
        reference_labels: List
    ) -> tuple[float, float, float, bool]:
        """
        Detect drift in cost structure.

        Returns:
            (current_mean, ref_mean, drift_pct, is_drifted)
        """
        # Average costs
        current_costs = [t.costs_bps for t in current_labels]
        ref_costs = [t.costs_bps for t in reference_labels]

        current_mean = np.mean(current_costs) if current_costs else 0
        ref_mean = np.mean(ref_costs) if ref_costs else 0

        # Drift percentage
        if ref_mean > 0:
            drift_pct = abs(current_mean - ref_mean) / ref_mean
        else:
            drift_pct = 0

        is_drifted = drift_pct > self.cost_drift_threshold

        logger.debug(
            "cost_drift",
            current_mean_bps=current_mean,
            ref_mean_bps=ref_mean,
            drift_pct=drift_pct,
            drifted=is_drifted
        )

        return float(current_mean), float(ref_mean), float(drift_pct), is_drifted

    def _determine_severity(
        self,
        ks_drift: bool,
        psi_drift: bool,
        label_drift: bool,
        cost_drift: bool
    ) -> DriftSeverity:
        """
        Determine overall drift severity.

        Logic:
        - SEVERE: 3+ indicators drifted
        - CRITICAL: 2 indicators drifted
        - WARNING: 1 indicator drifted
        - NONE: No drift
        """
        drift_count = sum([ks_drift, psi_drift, label_drift, cost_drift])

        if drift_count >= 3:
            return DriftSeverity.SEVERE
        elif drift_count == 2:
            return DriftSeverity.CRITICAL
        elif drift_count == 1:
            return DriftSeverity.WARNING
        else:
            return DriftSeverity.NONE


def format_drift_report(metrics: DriftMetrics) -> str:
    """
    Format drift metrics as human-readable report.

    Args:
        metrics: DriftMetrics object

    Returns:
        Formatted string
    """
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("DRIFT DETECTION REPORT")
    lines.append("=" * 70)

    # Overall severity
    severity_emoji = {
        DriftSeverity.NONE: "✅",
        DriftSeverity.WARNING: "⚠️",
        DriftSeverity.CRITICAL: "🔴",
        DriftSeverity.SEVERE: "🚨"
    }
    emoji = severity_emoji.get(metrics.overall_severity, "❓")

    lines.append(f"\n{emoji} Overall Severity: {metrics.overall_severity.value.upper()}")
    lines.append(f"Timestamp: {metrics.timestamp}\n")

    # KS Test
    lines.append("-" * 70)
    lines.append("1. RETURNS DISTRIBUTION (KS Test)")
    lines.append(f"   Statistic: {metrics.ks_statistic:.4f}")
    lines.append(f"   P-value:   {metrics.ks_pvalue:.4f}")
    drift_emoji = "🔴" if metrics.ks_drifted else "✅"
    lines.append(f"   {drift_emoji} Drifted:   {metrics.ks_drifted}")

    # PSI
    lines.append("\n" + "-" * 70)
    lines.append("2. PRICE DISTRIBUTION (PSI)")
    lines.append(f"   PSI Score: {metrics.psi_score:.4f}")
    drift_emoji = "🔴" if metrics.psi_drifted else "✅"
    lines.append(f"   {drift_emoji} Drifted:   {metrics.psi_drifted}")

    # Label Distribution
    if metrics.label_dist_current:
        lines.append("\n" + "-" * 70)
        lines.append("3. LABEL DISTRIBUTION")
        lines.append(f"   Current Profitable:   {metrics.label_dist_current.get('profitable', 0):.1%}")
        lines.append(f"   Reference Profitable: {metrics.label_dist_reference.get('profitable', 0):.1%}")
        lines.append(f"   Drift Score:          {metrics.label_drift_score:.4f}")
        drift_emoji = "🔴" if metrics.label_drifted else "✅"
        lines.append(f"   {drift_emoji} Drifted:              {metrics.label_drifted}")

    # Cost Drift
    if metrics.cost_current_mean > 0:
        lines.append("\n" + "-" * 70)
        lines.append("4. COST STRUCTURE")
        lines.append(f"   Current Mean:   {metrics.cost_current_mean:.2f} bps")
        lines.append(f"   Reference Mean: {metrics.cost_reference_mean:.2f} bps")
        lines.append(f"   Drift:          {metrics.cost_drift_pct:.1%}")
        drift_emoji = "🔴" if metrics.cost_drifted else "✅"
        lines.append(f"   {drift_emoji} Drifted:        {metrics.cost_drifted}")

    # Recommendations
    lines.append("\n" + "=" * 70)
    lines.append("RECOMMENDED ACTIONS")
    lines.append("=" * 70)

    if metrics.overall_severity == DriftSeverity.SEVERE:
        lines.append("🚨 SEVERE DRIFT DETECTED")
        lines.append("   → PAUSE TRADING immediately")
        lines.append("   → Trigger FULL RETRAIN")
        lines.append("   → Investigate market changes")
        lines.append("   → Notify operators")

    elif metrics.overall_severity == DriftSeverity.CRITICAL:
        lines.append("🔴 CRITICAL DRIFT DETECTED")
        lines.append("   → Trigger FULL RETRAIN")
        lines.append("   → Reduce position sizes")
        lines.append("   → Monitor closely")

    elif metrics.overall_severity == DriftSeverity.WARNING:
        lines.append("⚠️  WARNING: Minor drift detected")
        lines.append("   → Monitor for worsening")
        lines.append("   → Consider incremental retrain")
        lines.append("   → Log for analysis")

    else:
        lines.append("✅ No significant drift")
        lines.append("   → Continue normal operations")

    lines.append("=" * 70 + "\n")

    return "\n".join(lines)
