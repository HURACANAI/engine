"""
Calibration Tracker

Tracks calibration quality over time and by regime.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import structlog

from .metrics import (
    calculate_brier_score,
    calculate_ece,
    calculate_calibration_curve,
    calculate_calibration_by_regime
)

logger = structlog.get_logger(__name__)


@dataclass
class CalibrationSnapshot:
    """Single calibration measurement"""
    timestamp: datetime
    symbol: str
    model_id: str
    regime: Optional[str]

    num_samples: int
    brier_score: float
    ece: float

    # Before calibration
    uncalibrated_brier: Optional[float] = None
    uncalibrated_ece: Optional[float] = None

    # Calibration curve data
    mean_predicted: Optional[np.ndarray] = None
    fraction_positive: Optional[np.ndarray] = None


class CalibrationTracker:
    """
    Calibration Tracker

    Tracks model calibration over time and by regime.

    Stores calibration history for:
    - Model gates to check calibration quality
    - Monitoring dashboards
    - Detecting calibration drift

    Example:
        tracker = CalibrationTracker()

        # Record calibration after training
        tracker.record(
            timestamp=datetime.now(),
            symbol="BTC",
            model_id="btc_trend_v47",
            regime="trending",
            probabilities=model.predict_proba(X)[:, 1],
            actual_outcomes=y,
            uncalibrated_probs=uncalibrated_probs
        )

        # Check calibration history
        recent = tracker.get_recent_snapshots(symbol="BTC", limit=10)

        # Detect calibration drift
        is_drifting = tracker.is_calibration_drifting(
            symbol="BTC",
            window_days=7
        )
    """

    def __init__(self):
        """Initialize calibration tracker"""
        self.snapshots: List[CalibrationSnapshot] = []

    def record(
        self,
        timestamp: datetime,
        symbol: str,
        model_id: str,
        probabilities: np.ndarray,
        actual_outcomes: np.ndarray,
        regime: Optional[str] = None,
        uncalibrated_probs: Optional[np.ndarray] = None
    ) -> CalibrationSnapshot:
        """
        Record calibration snapshot

        Args:
            timestamp: Measurement time
            symbol: Trading symbol
            model_id: Model identifier
            probabilities: Calibrated probabilities
            actual_outcomes: Actual binary outcomes
            regime: Optional regime label
            uncalibrated_probs: Optional uncalibrated probabilities (for comparison)

        Returns:
            CalibrationSnapshot
        """
        probabilities = np.asarray(probabilities)
        actual_outcomes = np.asarray(actual_outcomes)

        if len(probabilities) == 0:
            logger.warning(
                "empty_calibration_record",
                symbol=symbol,
                model_id=model_id
            )
            return None

        # Calculate calibration metrics
        brier = calculate_brier_score(probabilities, actual_outcomes)
        ece = calculate_ece(probabilities, actual_outcomes)

        # Calculate calibration curve
        mean_pred, frac_pos = calculate_calibration_curve(
            probabilities,
            actual_outcomes
        )

        # Before calibration metrics (if available)
        uncalibrated_brier = None
        uncalibrated_ece = None

        if uncalibrated_probs is not None:
            uncalibrated_probs = np.asarray(uncalibrated_probs)
            uncalibrated_brier = calculate_brier_score(
                uncalibrated_probs,
                actual_outcomes
            )
            uncalibrated_ece = calculate_ece(
                uncalibrated_probs,
                actual_outcomes
            )

        # Create snapshot
        snapshot = CalibrationSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            model_id=model_id,
            regime=regime,
            num_samples=len(probabilities),
            brier_score=brier,
            ece=ece,
            uncalibrated_brier=uncalibrated_brier,
            uncalibrated_ece=uncalibrated_ece,
            mean_predicted=mean_pred,
            fraction_positive=frac_pos
        )

        self.snapshots.append(snapshot)

        logger.info(
            "calibration_recorded",
            symbol=symbol,
            model_id=model_id,
            regime=regime,
            num_samples=len(probabilities),
            brier=brier,
            ece=ece,
            improvement_brier=(
                uncalibrated_brier - brier
                if uncalibrated_brier is not None
                else None
            )
        )

        return snapshot

    def record_regime_calibration(
        self,
        timestamp: datetime,
        symbol: str,
        model_id: str,
        probabilities: np.ndarray,
        actual_outcomes: np.ndarray,
        regimes: np.ndarray
    ) -> Dict[str, CalibrationSnapshot]:
        """
        Record calibration for each regime separately

        Args:
            timestamp: Measurement time
            symbol: Trading symbol
            model_id: Model identifier
            probabilities: Probabilities
            actual_outcomes: Actual outcomes
            regimes: Regime labels

        Returns:
            Dict of {regime: CalibrationSnapshot}
        """
        regime_snapshots = {}

        regime_metrics = calculate_calibration_by_regime(
            probabilities,
            actual_outcomes,
            regimes
        )

        for regime, metrics in regime_metrics.items():
            # Get data for this regime
            regime_mask = regimes == regime
            regime_probs = probabilities[regime_mask]
            regime_outcomes = actual_outcomes[regime_mask]

            # Record snapshot
            snapshot = self.record(
                timestamp=timestamp,
                symbol=symbol,
                model_id=model_id,
                probabilities=regime_probs,
                actual_outcomes=regime_outcomes,
                regime=regime
            )

            regime_snapshots[regime] = snapshot

        return regime_snapshots

    def get_recent_snapshots(
        self,
        symbol: Optional[str] = None,
        model_id: Optional[str] = None,
        regime: Optional[str] = None,
        limit: int = 100
    ) -> List[CalibrationSnapshot]:
        """
        Get recent calibration snapshots

        Args:
            symbol: Filter by symbol (None = all)
            model_id: Filter by model (None = all)
            regime: Filter by regime (None = all)
            limit: Maximum snapshots to return

        Returns:
            List of CalibrationSnapshot (most recent first)
        """
        filtered = self.snapshots

        if symbol is not None:
            filtered = [s for s in filtered if s.symbol == symbol]

        if model_id is not None:
            filtered = [s for s in filtered if s.model_id == model_id]

        if regime is not None:
            filtered = [s for s in filtered if s.regime == regime]

        # Sort by timestamp (newest first)
        filtered.sort(key=lambda s: s.timestamp, reverse=True)

        return filtered[:limit]

    def is_calibration_drifting(
        self,
        symbol: str,
        window_days: int = 7,
        ece_threshold: float = 0.10,
        brier_threshold: float = 0.25
    ) -> bool:
        """
        Check if calibration is drifting (degrading over time)

        Args:
            symbol: Symbol to check
            window_days: Look at last N days
            ece_threshold: ECE threshold (> = drifting)
            brier_threshold: Brier threshold (> = drifting)

        Returns:
            True if calibration is drifting
        """
        cutoff = datetime.now() - pd.Timedelta(days=window_days)

        recent = [
            s for s in self.snapshots
            if s.symbol == symbol and s.timestamp >= cutoff
        ]

        if len(recent) == 0:
            logger.warning(
                "no_recent_calibration_data",
                symbol=symbol,
                window_days=window_days
            )
            return False

        # Check if recent ECE or Brier exceed thresholds
        avg_ece = np.mean([s.ece for s in recent])
        avg_brier = np.mean([s.brier_score for s in recent])

        is_drifting = (avg_ece > ece_threshold or avg_brier > brier_threshold)

        if is_drifting:
            logger.warning(
                "calibration_drift_detected",
                symbol=symbol,
                avg_ece=avg_ece,
                avg_brier=avg_brier,
                ece_threshold=ece_threshold,
                brier_threshold=brier_threshold
            )

        return is_drifting

    def get_calibration_improvement(
        self,
        symbol: str,
        model_id: str
    ) -> Optional[Dict[str, float]]:
        """
        Get calibration improvement from uncalibrated to calibrated

        Args:
            symbol: Symbol
            model_id: Model ID

        Returns:
            Dict with improvement metrics (None if no data)
        """
        snapshots = self.get_recent_snapshots(
            symbol=symbol,
            model_id=model_id,
            limit=1
        )

        if len(snapshots) == 0:
            return None

        snapshot = snapshots[0]

        if snapshot.uncalibrated_brier is None:
            return None

        return {
            "brier_before": snapshot.uncalibrated_brier,
            "brier_after": snapshot.brier_score,
            "brier_improvement": snapshot.uncalibrated_brier - snapshot.brier_score,
            "ece_before": snapshot.uncalibrated_ece,
            "ece_after": snapshot.ece,
            "ece_improvement": snapshot.uncalibrated_ece - snapshot.ece
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert snapshots to DataFrame for analysis

        Returns:
            DataFrame with calibration history
        """
        if len(self.snapshots) == 0:
            return pd.DataFrame()

        records = []

        for s in self.snapshots:
            records.append({
                "timestamp": s.timestamp,
                "symbol": s.symbol,
                "model_id": s.model_id,
                "regime": s.regime,
                "num_samples": s.num_samples,
                "brier_score": s.brier_score,
                "ece": s.ece,
                "uncalibrated_brier": s.uncalibrated_brier,
                "uncalibrated_ece": s.uncalibrated_ece
            })

        return pd.DataFrame(records)

    def generate_report(
        self,
        symbol: Optional[str] = None,
        window_days: int = 30
    ) -> str:
        """
        Generate calibration report

        Args:
            symbol: Filter by symbol (None = all)
            window_days: Report on last N days

        Returns:
            Human-readable report
        """
        cutoff = datetime.now() - pd.Timedelta(days=window_days)

        snapshots = [
            s for s in self.snapshots
            if s.timestamp >= cutoff
        ]

        if symbol is not None:
            snapshots = [s for s in snapshots if s.symbol == symbol]

        if len(snapshots) == 0:
            return "No calibration data available"

        # Group by symbol
        by_symbol = {}
        for s in snapshots:
            if s.symbol not in by_symbol:
                by_symbol[s.symbol] = []
            by_symbol[s.symbol].append(s)

        report_lines = [
            "=" * 80,
            "CALIBRATION REPORT",
            "=" * 80,
            "",
            f"Period: Last {window_days} days",
            f"Total Snapshots: {len(snapshots)}",
            ""
        ]

        for sym, sym_snapshots in by_symbol.items():
            avg_brier = np.mean([s.brier_score for s in sym_snapshots])
            avg_ece = np.mean([s.ece for s in sym_snapshots])

            # Get improvement (if available)
            with_uncalibrated = [
                s for s in sym_snapshots
                if s.uncalibrated_brier is not None
            ]

            if len(with_uncalibrated) > 0:
                avg_improvement = np.mean([
                    s.uncalibrated_brier - s.brier_score
                    for s in with_uncalibrated
                ])
                improvement_str = f" (↓ {avg_improvement:.3f} improvement)"
            else:
                improvement_str = ""

            report_lines.extend([
                f"{sym}:",
                f"  Snapshots: {len(sym_snapshots)}",
                f"  Avg Brier Score: {avg_brier:.3f}{improvement_str}",
                f"  Avg ECE: {avg_ece:.3f}",
                ""
            ])

            # Calibration status
            if avg_ece < 0.05:
                status = "✅ Well Calibrated"
            elif avg_ece < 0.10:
                status = "⚠️  Moderately Calibrated"
            else:
                status = "❌ Poorly Calibrated"

            report_lines.append(f"  Status: {status}")
            report_lines.append("")

        report_lines.append("=" * 80)

        return "\n".join(report_lines)

    def clear_old_snapshots(self, days_to_keep: int = 90) -> int:
        """
        Clear snapshots older than N days

        Args:
            days_to_keep: Keep snapshots from last N days

        Returns:
            Number of snapshots removed
        """
        cutoff = datetime.now() - pd.Timedelta(days=days_to_keep)

        original_count = len(self.snapshots)

        self.snapshots = [
            s for s in self.snapshots
            if s.timestamp >= cutoff
        ]

        removed_count = original_count - len(self.snapshots)

        if removed_count > 0:
            logger.info(
                "old_snapshots_cleared",
                removed=removed_count,
                kept=len(self.snapshots),
                days_to_keep=days_to_keep
            )

        return removed_count
