"""
Self-Diagnostic System - Phase 4

The system that monitors its own health and diagnoses problems automatically.

Capabilities:
1. Detects performance anomalies
2. Identifies root causes of poor performance
3. Suggests fixes
4. Monitors all subsystems
5. Generates health reports

This is like having a doctor for your trading system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import structlog

logger = structlog.get_logger()


class HealthStatus(Enum):
    """Overall health status."""

    EXCELLENT = "excellent"  # Everything working optimally
    GOOD = "good"  # Working well, minor issues
    WARNING = "warning"  # Some issues detected
    CRITICAL = "critical"  # Serious problems
    FAILING = "failing"  # System not functioning properly


class DiagnosticCategory(Enum):
    """Categories of diagnostics."""

    PERFORMANCE = "performance"
    DATA_QUALITY = "data_quality"
    LEARNING = "learning"
    RISK_MANAGEMENT = "risk_management"
    EXECUTION = "execution"


@dataclass
class DiagnosticIssue:
    """A single diagnostic issue."""

    category: DiagnosticCategory
    severity: HealthStatus  # How serious
    title: str
    description: str
    detected_at: datetime
    suggested_fix: Optional[str] = None
    auto_fixable: bool = False


@dataclass
class SystemHealthReport:
    """Complete system health report."""

    timestamp: datetime
    overall_status: HealthStatus
    issues: List[DiagnosticIssue]
    subsystem_health: Dict[str, HealthStatus]
    metrics: Dict[str, float]
    recommendations: List[str]


class SelfDiagnostic:
    """
    Self-diagnostic system that monitors and diagnoses the trading system.

    Monitors:
    1. Trading performance metrics
    2. Learning system health
    3. Data quality issues
    4. Risk management violations
    5. Execution problems
    """

    def __init__(
        self,
        performance_threshold: float = 0.50,  # Min acceptable win rate
        learning_efficiency_threshold: float = 0.0,  # Must be improving
        max_drawdown_threshold: float = 0.20,  # Max 20% drawdown
    ):
        """
        Initialize self-diagnostic system.

        Args:
            performance_threshold: Minimum acceptable performance
            learning_efficiency_threshold: Minimum learning improvement
            max_drawdown_threshold: Maximum acceptable drawdown
        """
        self.perf_threshold = performance_threshold
        self.learning_threshold = learning_efficiency_threshold
        self.drawdown_threshold = max_drawdown_threshold

        # Issue tracking
        self.current_issues: List[DiagnosticIssue] = []
        self.issue_history: List[DiagnosticIssue] = []

        # Metrics history
        self.win_rate_history: List[float] = []
        self.drawdown_history: List[float] = []
        self.learning_rate_history: List[float] = []

        logger.info(
            "self_diagnostic_initialized",
            perf_threshold=performance_threshold,
            drawdown_threshold=max_drawdown_threshold,
        )

    def diagnose(
        self,
        win_rate: float,
        current_drawdown: float,
        learning_efficiency: float,
        feature_importance_stability: float,
        num_trades: int,
        data_quality_score: float = 1.0,
    ) -> SystemHealthReport:
        """
        Run complete system diagnosis.

        Args:
            win_rate: Current win rate (0-1)
            current_drawdown: Current drawdown (0-1)
            learning_efficiency: How much system is improving (0-1+)
            feature_importance_stability: Stability of feature importance (0-1)
            num_trades: Number of trades executed
            data_quality_score: Quality of input data (0-1)

        Returns:
            SystemHealthReport with complete diagnosis
        """
        # Store metrics
        self.win_rate_history.append(win_rate)
        self.drawdown_history.append(current_drawdown)
        self.learning_rate_history.append(learning_efficiency)

        # Keep recent history
        if len(self.win_rate_history) > 100:
            self.win_rate_history = self.win_rate_history[-100:]
            self.drawdown_history = self.drawdown_history[-100:]
            self.learning_rate_history = self.learning_rate_history[-100:]

        # Clear previous issues
        self.current_issues = []

        # Run all diagnostic checks
        self._check_performance(win_rate, num_trades)
        self._check_drawdown(current_drawdown)
        self._check_learning(learning_efficiency)
        self._check_feature_stability(feature_importance_stability)
        self._check_data_quality(data_quality_score)
        self._check_trade_volume(num_trades)

        # Determine overall health
        overall_status = self._calculate_overall_health()

        # Generate subsystem health
        subsystem_health = {
            "performance": self._assess_performance_health(win_rate),
            "learning": self._assess_learning_health(learning_efficiency),
            "risk_management": self._assess_risk_health(current_drawdown),
            "data_quality": self._assess_data_health(data_quality_score),
        }

        # Generate recommendations
        recommendations = self._generate_recommendations()

        # Create metrics dict
        metrics = {
            "win_rate": win_rate,
            "current_drawdown": current_drawdown,
            "learning_efficiency": learning_efficiency,
            "feature_stability": feature_importance_stability,
            "num_trades": num_trades,
            "data_quality": data_quality_score,
        }

        report = SystemHealthReport(
            timestamp=datetime.now(),
            overall_status=overall_status,
            issues=self.current_issues.copy(),
            subsystem_health=subsystem_health,
            metrics=metrics,
            recommendations=recommendations,
        )

        # Store issues in history
        self.issue_history.extend(self.current_issues)
        if len(self.issue_history) > 1000:
            self.issue_history = self.issue_history[-1000:]

        logger.info(
            "system_diagnosed",
            overall_status=overall_status.value,
            num_issues=len(self.current_issues),
            win_rate=win_rate,
            drawdown=current_drawdown,
        )

        return report

    def _check_performance(self, win_rate: float, num_trades: int) -> None:
        """Check trading performance."""
        # Check win rate
        if win_rate < self.perf_threshold:
            severity = HealthStatus.CRITICAL if win_rate < 0.45 else HealthStatus.WARNING

            self.current_issues.append(
                DiagnosticIssue(
                    category=DiagnosticCategory.PERFORMANCE,
                    severity=severity,
                    title="Low Win Rate",
                    description=f"Win rate {win_rate:.1%} below threshold {self.perf_threshold:.1%}",
                    detected_at=datetime.now(),
                    suggested_fix="Check confidence threshold, regime detection, or pattern quality",
                    auto_fixable=False,
                )
            )

        # Check for declining performance
        if len(self.win_rate_history) >= 20:
            recent_trend = np.polyfit(range(20), self.win_rate_history[-20:], 1)[0]
            if recent_trend < -0.01:  # Declining
                self.current_issues.append(
                    DiagnosticIssue(
                        category=DiagnosticCategory.PERFORMANCE,
                        severity=HealthStatus.WARNING,
                        title="Declining Performance",
                        description=f"Win rate trending down: {recent_trend:.3f} per period",
                        detected_at=datetime.now(),
                        suggested_fix="May need to retrain or adjust to new market regime",
                        auto_fixable=False,
                    )
                )

    def _check_drawdown(self, current_drawdown: float) -> None:
        """Check drawdown levels."""
        if current_drawdown > self.drawdown_threshold:
            severity = (
                HealthStatus.CRITICAL
                if current_drawdown > 0.25
                else HealthStatus.WARNING
            )

            self.current_issues.append(
                DiagnosticIssue(
                    category=DiagnosticCategory.RISK_MANAGEMENT,
                    severity=severity,
                    title="Excessive Drawdown",
                    description=f"Drawdown {current_drawdown:.1%} exceeds threshold {self.drawdown_threshold:.1%}",
                    detected_at=datetime.now(),
                    suggested_fix="Reduce position sizes or pause trading",
                    auto_fixable=True,  # Can auto-reduce sizes
                )
            )

    def _check_learning(self, learning_efficiency: float) -> None:
        """Check learning system health."""
        if learning_efficiency < self.learning_threshold:
            self.current_issues.append(
                DiagnosticIssue(
                    category=DiagnosticCategory.LEARNING,
                    severity=HealthStatus.WARNING,
                    title="Poor Learning Efficiency",
                    description=f"System not improving: efficiency {learning_efficiency:.2f}",
                    detected_at=datetime.now(),
                    suggested_fix="Increase learning rate or try new features",
                    auto_fixable=True,  # Can adjust learning rates
                )
            )

    def _check_feature_stability(self, stability: float) -> None:
        """Check feature importance stability."""
        if stability < 0.5:
            self.current_issues.append(
                DiagnosticIssue(
                    category=DiagnosticCategory.LEARNING,
                    severity=HealthStatus.WARNING,
                    title="Unstable Feature Importance",
                    description=f"Feature importance changing rapidly: stability {stability:.2f}",
                    detected_at=datetime.now(),
                    suggested_fix="May indicate regime change or data quality issues",
                    auto_fixable=False,
                )
            )

    def _check_data_quality(self, quality_score: float) -> None:
        """Check data quality."""
        if quality_score < 0.8:
            severity = HealthStatus.CRITICAL if quality_score < 0.6 else HealthStatus.WARNING

            self.current_issues.append(
                DiagnosticIssue(
                    category=DiagnosticCategory.DATA_QUALITY,
                    severity=severity,
                    title="Poor Data Quality",
                    description=f"Data quality score {quality_score:.2f} below acceptable",
                    detected_at=datetime.now(),
                    suggested_fix="Check data sources, fill missing values",
                    auto_fixable=False,
                )
            )

    def _check_trade_volume(self, num_trades: int) -> None:
        """Check if trading enough."""
        if num_trades < 10:
            self.current_issues.append(
                DiagnosticIssue(
                    category=DiagnosticCategory.EXECUTION,
                    severity=HealthStatus.WARNING,
                    title="Low Trade Volume",
                    description=f"Only {num_trades} trades executed",
                    detected_at=datetime.now(),
                    suggested_fix="Lower confidence threshold or check entry conditions",
                    auto_fixable=True,
                )
            )

    def _calculate_overall_health(self) -> HealthStatus:
        """Calculate overall system health."""
        if not self.current_issues:
            return HealthStatus.EXCELLENT

        # Check severity of issues
        severities = [issue.severity for issue in self.current_issues]

        if HealthStatus.FAILING in severities:
            return HealthStatus.FAILING
        elif HealthStatus.CRITICAL in severities:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in severities:
            return HealthStatus.WARNING
        else:
            return HealthStatus.GOOD

    def _assess_performance_health(self, win_rate: float) -> HealthStatus:
        """Assess performance subsystem health."""
        if win_rate >= 0.65:
            return HealthStatus.EXCELLENT
        elif win_rate >= 0.55:
            return HealthStatus.GOOD
        elif win_rate >= 0.50:
            return HealthStatus.WARNING
        else:
            return HealthStatus.CRITICAL

    def _assess_learning_health(self, efficiency: float) -> HealthStatus:
        """Assess learning subsystem health."""
        if efficiency > 0.1:
            return HealthStatus.EXCELLENT
        elif efficiency > 0.0:
            return HealthStatus.GOOD
        elif efficiency > -0.05:
            return HealthStatus.WARNING
        else:
            return HealthStatus.CRITICAL

    def _assess_risk_health(self, drawdown: float) -> HealthStatus:
        """Assess risk management health."""
        if drawdown < 0.05:
            return HealthStatus.EXCELLENT
        elif drawdown < 0.10:
            return HealthStatus.GOOD
        elif drawdown < 0.20:
            return HealthStatus.WARNING
        else:
            return HealthStatus.CRITICAL

    def _assess_data_health(self, quality: float) -> HealthStatus:
        """Assess data quality health."""
        if quality >= 0.95:
            return HealthStatus.EXCELLENT
        elif quality >= 0.85:
            return HealthStatus.GOOD
        elif quality >= 0.75:
            return HealthStatus.WARNING
        else:
            return HealthStatus.CRITICAL

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Based on issues
        for issue in self.current_issues:
            if issue.suggested_fix:
                recommendations.append(f"{issue.title}: {issue.suggested_fix}")

        # General recommendations
        if len(self.win_rate_history) >= 10:
            avg_win_rate = np.mean(self.win_rate_history[-10:])
            if avg_win_rate < 0.52:
                recommendations.append(
                    "Consider increasing confidence threshold to filter more trades"
                )

        if len(self.drawdown_history) >= 10:
            max_dd = max(self.drawdown_history[-10:])
            if max_dd > 0.15:
                recommendations.append("Reduce position sizes to manage risk better")

        return recommendations

    def get_issue_summary(self) -> Dict:
        """Get summary of current issues."""
        return {
            "total_issues": len(self.current_issues),
            "by_severity": {
                severity.value: sum(
                    1 for issue in self.current_issues if issue.severity == severity
                )
                for severity in HealthStatus
            },
            "by_category": {
                category.value: sum(
                    1 for issue in self.current_issues if issue.category == category
                )
                for category in DiagnosticCategory
            },
            "auto_fixable": sum(1 for issue in self.current_issues if issue.auto_fixable),
        }
