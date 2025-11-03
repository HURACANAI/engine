"""Pattern health monitoring and degradation detection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
import structlog

from .types import AlertSeverity, HealthAlert, HealthStatus, PatternHealthReport

logger = structlog.get_logger(__name__)


@dataclass
class PatternHealthConfig:
    """Configuration for pattern health monitoring."""
    min_win_rate_threshold: float = 0.45  # 45% minimum
    degradation_threshold_pct: float = 0.15  # 15% decline
    min_trades_for_assessment: int = 20
    comparison_window_days: int = 7
    baseline_window_days: int = 30


class PatternHealthMonitor:
    """
    Monitors health of trading patterns over time.

    Detects when patterns stop working, degrade significantly,
    or show signs of overfitting.
    """

    def __init__(self, dsn: str, config: Optional[PatternHealthConfig] = None):
        self.dsn = dsn
        self.config = config or PatternHealthConfig()
        self._conn: Optional[psycopg2.extensions.connection] = None

    def connect(self) -> None:
        """Establish database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.dsn)

    def close(self) -> None:
        """Close database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()

    def check_pattern_health(self, pattern_id: int) -> Optional[PatternHealthReport]:
        """Check health of a specific pattern."""
        self.connect()

        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get pattern baseline performance
            cur.execute(
                """
                SELECT
                    pattern_name,
                    win_rate as stored_win_rate,
                    total_occurrences
                FROM pattern_library
                WHERE pattern_id = %s
                """,
                (pattern_id,),
            )
            pattern_info = cur.fetchone()

            if not pattern_info:
                return None

            # Get recent performance (last 7 days)
            cur.execute(
                """
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN is_winner THEN 1 ELSE 0 END) as wins,
                    AVG(CASE WHEN is_winner THEN 1.0 ELSE 0.0 END) as win_rate
                FROM trade_memory
                WHERE entry_timestamp >= NOW() - INTERVAL '%s days'
                  AND is_winner IS NOT NULL
                  AND model_version LIKE '%%pattern_' || %s || '%%'
                """,
                (self.config.comparison_window_days, pattern_id),
            )
            recent = cur.fetchone()

        if not recent or recent["total_trades"] < self.config.min_trades_for_assessment:
            # Not enough data to assess
            return None

        baseline_wr = float(pattern_info["stored_win_rate"] or 0.5)
        current_wr = float(recent["win_rate"] or 0.0)
        total_trades = recent["total_trades"]

        # Calculate degradation
        degradation_pct = ((baseline_wr - current_wr) / baseline_wr) if baseline_wr > 0 else 0.0

        # Determine status
        issues = []
        if current_wr < self.config.min_win_rate_threshold:
            status = HealthStatus.CRITICAL
            issues.append(f"Win rate below minimum threshold ({self.config.min_win_rate_threshold:.0%})")
        elif degradation_pct > self.config.degradation_threshold_pct:
            status = HealthStatus.DEGRADED
            issues.append(f"Win rate degraded {degradation_pct:.1%} from baseline")
        else:
            status = HealthStatus.HEALTHY

        return PatternHealthReport(
            pattern_id=pattern_id,
            pattern_name=pattern_info["pattern_name"],
            current_win_rate=current_wr,
            baseline_win_rate=baseline_wr,
            recent_trades=total_trades,
            status=status,
            degradation_pct=degradation_pct,
            issues=issues,
            timestamp=datetime.now(tz=timezone.utc),
        )

    def check_all_patterns(self) -> List[PatternHealthReport]:
        """Check health of all active patterns."""
        self.connect()
        reports = []

        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get all patterns with sufficient usage
            cur.execute(
                """
                SELECT pattern_id
                FROM pattern_library
                WHERE total_occurrences >= %s
                  AND reliability_score > 0  -- Not blacklisted
                ORDER BY total_occurrences DESC
                """,
                (self.config.min_trades_for_assessment,),
            )
            patterns = cur.fetchall()

        for row in patterns:
            report = self.check_pattern_health(row["pattern_id"])
            if report:
                reports.append(report)

        logger.info("pattern_health_check_complete", patterns_checked=len(reports))
        return reports

    def generate_pattern_alerts(self, reports: List[PatternHealthReport]) -> List[HealthAlert]:
        """Generate alerts from pattern health reports."""
        alerts = []

        for report in reports:
            if report.status == HealthStatus.HEALTHY:
                continue

            severity = (
                AlertSeverity.CRITICAL
                if report.status == HealthStatus.CRITICAL
                else AlertSeverity.WARNING
            )

            # Build message
            if report.current_win_rate < self.config.min_win_rate_threshold:
                message = (
                    f"Pattern '{report.pattern_name}' failing\n"
                    f"Win rate: {report.current_win_rate:.1%} (threshold: {self.config.min_win_rate_threshold:.1%})\n"
                    f"Baseline: {report.baseline_win_rate:.1%}\n"
                    f"Recent trades: {report.recent_trades} ({self.config.comparison_window_days} days)"
                )
                suggested_actions = [
                    "Blacklist this pattern immediately",
                    "Investigate market regime changes",
                    "Review recent losing trades for this pattern",
                    "Consider retraining on current market conditions",
                ]
            else:
                message = (
                    f"Pattern '{report.pattern_name}' degrading\n"
                    f"Win rate: {report.current_win_rate:.1%} (was {report.baseline_win_rate:.1%})\n"
                    f"Degradation: {report.degradation_pct:.1%}\n"
                    f"Recent trades: {report.recent_trades}"
                )
                suggested_actions = [
                    "Monitor pattern closely for further decline",
                    "Review if market regime changed",
                    "Consider reducing position size for this pattern",
                    "Retrain if degradation continues",
                ]

            alert = HealthAlert(
                severity=severity,
                title=f"Pattern Health Issue: {report.pattern_name}",
                message=message,
                metric=None,
                suggested_actions=suggested_actions,
                context={
                    "pattern_id": report.pattern_id,
                    "current_win_rate": report.current_win_rate,
                    "baseline_win_rate": report.baseline_win_rate,
                    "degradation_pct": report.degradation_pct,
                    "recent_trades": report.recent_trades,
                },
                timestamp=report.timestamp,
                alert_id=f"pattern_{report.pattern_id}_{report.timestamp.timestamp()}",
            )

            alerts.append(alert)
            logger.warning(
                "pattern_health_alert",
                pattern=report.pattern_name,
                win_rate=report.current_win_rate,
                status=report.status.value,
            )

        return alerts

    def check_for_overfitting(self) -> List[HealthAlert]:
        """
        Detect patterns that perform well in backtest but poorly in live trading.

        This suggests overfitting to historical data.
        """
        self.connect()
        alerts = []

        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Find patterns with high stored win rate but poor recent performance
            cur.execute(
                """
                WITH recent_performance AS (
                    SELECT
                        model_version,
                        AVG(CASE WHEN is_winner THEN 1.0 ELSE 0.0 END) as live_win_rate,
                        COUNT(*) as live_trades
                    FROM trade_memory
                    WHERE entry_timestamp >= NOW() - INTERVAL '14 days'
                      AND is_winner IS NOT NULL
                    GROUP BY model_version
                )
                SELECT
                    pl.pattern_id,
                    pl.pattern_name,
                    pl.win_rate as backtest_win_rate,
                    rp.live_win_rate,
                    rp.live_trades
                FROM pattern_library pl
                JOIN recent_performance rp ON rp.model_version LIKE '%pattern_' || pl.pattern_id || '%'
                WHERE pl.win_rate > 0.6  -- Good backtest performance
                  AND rp.live_win_rate < 0.48  -- Poor live performance
                  AND rp.live_trades >= 30  -- Sufficient sample
                  AND pl.reliability_score > 0  -- Not blacklisted
                """,
            )
            overfitted = cur.fetchall()

        for row in overfitted:
            backtest_wr = float(row["backtest_win_rate"])
            live_wr = float(row["live_win_rate"])
            gap = backtest_wr - live_wr

            alert = HealthAlert(
                severity=AlertSeverity.WARNING,
                title=f"Potential Overfitting: {row['pattern_name']}",
                message=(
                    f"Pattern shows signs of overfitting to historical data\n"
                    f"Backtest win rate: {backtest_wr:.1%}\n"
                    f"Live win rate: {live_wr:.1%}\n"
                    f"Gap: {gap:.1%}\n"
                    f"Live trades: {row['live_trades']}"
                ),
                metric=None,
                suggested_actions=[
                    "Blacklist pattern - likely overfit to historical data",
                    "Review feature engineering for this pattern",
                    "Use more recent training data",
                    "Add regularization to prevent overfitting",
                ],
                context={
                    "pattern_id": row["pattern_id"],
                    "backtest_win_rate": backtest_wr,
                    "live_win_rate": live_wr,
                    "gap": gap,
                },
                timestamp=datetime.now(tz=timezone.utc),
                alert_id=f"overfit_{row['pattern_id']}_{datetime.now(tz=timezone.utc).timestamp()}",
            )

            alerts.append(alert)
            logger.warning("overfitting_detected", pattern=row["pattern_name"], gap=gap)

        return alerts
