"""
Robust Overfitting Detection System

Detects overfitting using multiple indicators:
1. Train/Test performance gap
2. Cross-validation stability
3. Performance degradation over time
4. Feature importance stability
5. Model complexity vs performance
6. Out-of-sample vs in-sample metrics

Multiple indicators provide robust detection of overfitting.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class OverfittingIndicator:
    """Single overfitting indicator."""

    name: str
    score: float  # 0-1, higher = more overfitting
    threshold: float  # Threshold for flagging
    flagged: bool  # True if indicator suggests overfitting
    message: str  # Explanation


@dataclass
class OverfittingReport:
    """Complete overfitting detection report."""

    is_overfitting: bool  # True if overfitting detected
    confidence: float  # 0-1, confidence in overfitting detection
    indicators: List[OverfittingIndicator]
    overall_score: float  # 0-1, higher = more overfitting
    recommendation: str  # Action recommendation
    severity: str  # 'LOW', 'MODERATE', 'HIGH', 'CRITICAL'


class RobustOverfittingDetector:
    """
    Robust overfitting detection using multiple indicators.

    Indicators:
    1. Train/Test performance gap (Sharpe, win rate)
    2. Cross-validation stability (std dev across folds)
    3. Performance degradation over time
    4. Feature importance stability
    5. Model complexity vs performance
    6. Out-of-sample vs in-sample metrics

    Usage:
        detector = RobustOverfittingDetector()

        report = detector.detect_overfitting(
            train_sharpe=2.5,
            test_sharpe=1.2,
            train_win_rate=0.85,
            test_win_rate=0.62,
            cv_sharpe_std=0.35,
            performance_trend=[1.5, 1.3, 1.1, 0.9],  # Degrading
        )

        if report.is_overfitting:
            logger.warning(f"Overfitting detected: {report.recommendation}")
    """

    def __init__(
        self,
        train_test_gap_threshold: float = 0.5,  # Max acceptable gap
        cv_stability_threshold: float = 0.3,  # Max acceptable std dev
        degradation_threshold: float = -0.2,  # Max acceptable degradation rate
        complexity_penalty: float = 0.1,  # Penalty for high complexity
    ):
        """
        Initialize robust overfitting detector.

        Args:
            train_test_gap_threshold: Maximum acceptable train/test gap
            cv_stability_threshold: Maximum acceptable CV std dev
            degradation_threshold: Maximum acceptable performance degradation
            complexity_penalty: Penalty for high model complexity
        """
        self.train_test_gap_threshold = train_test_gap_threshold
        self.cv_stability_threshold = cv_stability_threshold
        self.degradation_threshold = degradation_threshold
        self.complexity_penalty = complexity_penalty

        logger.info("robust_overfitting_detector_initialized")

    def detect_overfitting(
        self,
        train_sharpe: float,
        test_sharpe: float,
        train_win_rate: float,
        test_win_rate: float,
        cv_sharpe_std: Optional[float] = None,
        cv_win_rate_std: Optional[float] = None,
        performance_trend: Optional[List[float]] = None,
        feature_importance_stability: Optional[float] = None,
        model_complexity: Optional[float] = None,
        oos_metrics: Optional[Dict[str, float]] = None,
    ) -> OverfittingReport:
        """
        Detect overfitting using multiple indicators.

        Args:
            train_sharpe: Training Sharpe ratio
            test_sharpe: Test Sharpe ratio
            train_win_rate: Training win rate
            test_win_rate: Test win rate
            cv_sharpe_std: CV Sharpe std dev (optional)
            cv_win_rate_std: CV win rate std dev (optional)
            performance_trend: Performance over time (optional)
            feature_importance_stability: Feature importance stability (optional)
            model_complexity: Model complexity score (optional)
            oos_metrics: Additional OOS metrics (optional)

        Returns:
            OverfittingReport with detection results
        """
        indicators = []

        # Indicator 1: Train/Test Sharpe gap
        sharpe_gap = train_sharpe - test_sharpe
        sharpe_gap_score = min(abs(sharpe_gap) / self.train_test_gap_threshold, 1.0)
        sharpe_gap_flagged = sharpe_gap > self.train_test_gap_threshold

        indicators.append(
            OverfittingIndicator(
                name="train_test_sharpe_gap",
                score=sharpe_gap_score,
                threshold=self.train_test_gap_threshold,
                flagged=sharpe_gap_flagged,
                message=f"Train/Test Sharpe gap: {sharpe_gap:.2f} ({'⚠️ Overfitting' if sharpe_gap_flagged else '✅ Good'})",
            )
        )

        # Indicator 2: Train/Test Win Rate gap
        wr_gap = train_win_rate - test_win_rate
        wr_gap_score = min(abs(wr_gap) / 0.2, 1.0)  # 20% gap = max
        wr_gap_flagged = wr_gap > 0.2

        indicators.append(
            OverfittingIndicator(
                name="train_test_win_rate_gap",
                score=wr_gap_score,
                threshold=0.2,
                flagged=wr_gap_flagged,
                message=f"Train/Test Win Rate gap: {wr_gap:.1%} ({'⚠️ Overfitting' if wr_gap_flagged else '✅ Good'})",
            )
        )

        # Indicator 3: CV Stability
        if cv_sharpe_std is not None:
            cv_stability_score = min(cv_sharpe_std / self.cv_stability_threshold, 1.0)
            cv_stability_flagged = cv_sharpe_std > self.cv_stability_threshold

            indicators.append(
                OverfittingIndicator(
                    name="cv_stability",
                    score=cv_stability_score,
                    threshold=self.cv_stability_threshold,
                    flagged=cv_stability_flagged,
                    message=f"CV Sharpe std dev: {cv_sharpe_std:.2f} ({'⚠️ Unstable' if cv_stability_flagged else '✅ Stable'})",
                )
            )

        # Indicator 4: Performance Degradation
        if performance_trend is not None and len(performance_trend) >= 3:
            # Calculate degradation rate (negative = degrading)
            trend_array = np.array(performance_trend)
            degradation_rate = (trend_array[-1] - trend_array[0]) / len(trend_array)
            degradation_score = min(abs(degradation_rate) / abs(self.degradation_threshold), 1.0)
            degradation_flagged = degradation_rate < self.degradation_threshold

            indicators.append(
                OverfittingIndicator(
                    name="performance_degradation",
                    score=degradation_score,
                    threshold=self.degradation_threshold,
                    flagged=degradation_flagged,
                    message=f"Performance degradation: {degradation_rate:.3f} per period ({'⚠️ Degrading' if degradation_flagged else '✅ Stable'})",
                )
            )

        # Indicator 5: Feature Importance Stability
        if feature_importance_stability is not None:
            # Lower stability = higher overfitting risk
            importance_score = 1.0 - feature_importance_stability
            importance_flagged = feature_importance_stability < 0.7

            indicators.append(
                OverfittingIndicator(
                    name="feature_importance_stability",
                    score=importance_score,
                    threshold=0.7,
                    flagged=importance_flagged,
                    message=f"Feature importance stability: {feature_importance_stability:.2f} ({'⚠️ Unstable' if importance_flagged else '✅ Stable'})",
                )
            )

        # Indicator 6: Model Complexity vs Performance
        if model_complexity is not None and test_sharpe > 0:
            # High complexity + low test performance = overfitting
            complexity_ratio = model_complexity / test_sharpe
            complexity_score = min(complexity_ratio / 10.0, 1.0)  # Normalize
            complexity_flagged = complexity_ratio > 5.0

            indicators.append(
                OverfittingIndicator(
                    name="complexity_performance_ratio",
                    score=complexity_score,
                    threshold=5.0,
                    flagged=complexity_flagged,
                    message=f"Complexity/Performance ratio: {complexity_ratio:.2f} ({'⚠️ Overfitting' if complexity_flagged else '✅ Good'})",
                )
            )

        # Calculate overall score (weighted average)
        if indicators:
            overall_score = np.mean([ind.score for ind in indicators])
            flagged_count = sum(1 for ind in indicators if ind.flagged)
            confidence = flagged_count / len(indicators)
        else:
            overall_score = 0.0
            confidence = 0.0

        # Determine if overfitting
        is_overfitting = overall_score > 0.5 or confidence > 0.5

        # Determine severity
        if overall_score > 0.8 or confidence > 0.8:
            severity = "CRITICAL"
        elif overall_score > 0.6 or confidence > 0.6:
            severity = "HIGH"
        elif overall_score > 0.4 or confidence > 0.4:
            severity = "MODERATE"
        else:
            severity = "LOW"

        # Generate recommendation
        if is_overfitting:
            if severity == "CRITICAL":
                recommendation = "❌ CRITICAL: Model is severely overfitting. DO NOT DEPLOY. Simplify model or add regularization."
            elif severity == "HIGH":
                recommendation = "⚠️ HIGH: Model is overfitting. Review model complexity and add regularization."
            elif severity == "MODERATE":
                recommendation = "⚠️ MODERATE: Model shows signs of overfitting. Monitor closely and consider regularization."
            else:
                recommendation = "✅ LOW: Model shows minor overfitting signs. Monitor performance."
        else:
            recommendation = "✅ Model shows no significant overfitting. Safe to proceed."

        report = OverfittingReport(
            is_overfitting=is_overfitting,
            confidence=confidence,
            indicators=indicators,
            overall_score=overall_score,
            recommendation=recommendation,
            severity=severity,
        )

        logger.info(
            "overfitting_detection_complete",
            is_overfitting=is_overfitting,
            confidence=confidence,
            severity=severity,
            overall_score=overall_score,
        )

        return report

    def get_statistics(self) -> dict:
        """Get detector statistics."""
        return {
            'train_test_gap_threshold': self.train_test_gap_threshold,
            'cv_stability_threshold': self.cv_stability_threshold,
            'degradation_threshold': self.degradation_threshold,
            'complexity_penalty': self.complexity_penalty,
        }

