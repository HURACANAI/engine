"""Statistical anomaly detection for trading metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
import psycopg2
import psycopg2.errors
from psycopg2.extras import RealDictCursor
import structlog

from .types import AlertSeverity, HealthAlert, HealthMetric, HealthStatus

logger = structlog.get_logger(__name__)


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection."""
    win_rate_stddev_threshold: float = 2.0
    profit_stddev_threshold: float = 2.5
    trade_volume_drop_pct: float = 0.5  # 50% drop
    min_trades_for_analysis: int = 30
    baseline_window_days: int = 7
    comparison_window_hours: int = 24


class StatisticalAnomalyDetector:
    """
    Detects anomalies in trading performance using statistical methods.

    Uses z-scores, rolling averages, and trend analysis to identify
    when metrics deviate significantly from baseline behavior.
    """

    def __init__(self, dsn: str, config: Optional[AnomalyConfig] = None):
        self.dsn = dsn
        self.config = config or AnomalyConfig()
        self._conn: Optional[psycopg2.extensions.connection] = None

    def connect(self) -> None:
        """Establish database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.dsn)

    def close(self) -> None:
        """Close database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()

    def check_win_rate(self) -> List[HealthAlert]:
        """Check for win rate anomalies."""
        alerts = []
        
        try:
            self.connect()
            
            with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get baseline win rate (last 7 days)
                cur.execute(
                    """
                    SELECT
                        AVG(CASE WHEN is_winner THEN 1.0 ELSE 0.0 END) as win_rate,
                        STDDEV(CASE WHEN is_winner THEN 1.0 ELSE 0.0 END) as stddev,
                        COUNT(*) as total_trades
                    FROM trade_memory
                    WHERE entry_timestamp >= NOW() - INTERVAL '%s days'
                      AND entry_timestamp < NOW() - INTERVAL '%s hours'
                      AND is_winner IS NOT NULL
                    """,
                    (self.config.baseline_window_days, self.config.comparison_window_hours),
                )
                baseline = cur.fetchone()

                # Get current win rate (last 24 hours)
                cur.execute(
                    """
                    SELECT
                        AVG(CASE WHEN is_winner THEN 1.0 ELSE 0.0 END) as win_rate,
                        COUNT(*) as total_trades
                    FROM trade_memory
                    WHERE entry_timestamp >= NOW() - INTERVAL '%s hours'
                      AND is_winner IS NOT NULL
                    """,
                    (self.config.comparison_window_hours,),
                )
                current = cur.fetchone()
        except psycopg2.errors.UndefinedTable:
            logger.debug("table_trade_memory_not_exists", message="Table doesn't exist yet - skipping win rate check")
            self._conn.rollback()
            return alerts
        except Exception as e:
            logger.warning("win_rate_check_failed", error=str(e))
            if self._conn:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
            return alerts

        if not baseline or not current:
            return alerts

        baseline_wr = float(baseline["win_rate"] or 0.0)
        baseline_std = float(baseline["stddev"] or 0.05)
        current_wr = float(current["win_rate"] or 0.0)
        current_trades = current["total_trades"]

        # Check if we have enough data
        if current_trades < self.config.min_trades_for_analysis:
            return alerts

        # Calculate z-score
        z_score = (current_wr - baseline_wr) / (baseline_std + 1e-6)

        # Create metric
        metric = HealthMetric(
            name="win_rate",
            value=current_wr,
            baseline=baseline_wr,
            threshold=baseline_wr - (self.config.win_rate_stddev_threshold * baseline_std),
            is_healthy=z_score > -self.config.win_rate_stddev_threshold,
            timestamp=datetime.now(tz=timezone.utc),
            context={
                "z_score": z_score,
                "baseline_trades": baseline["total_trades"],
                "current_trades": current_trades,
            },
        )

        # Generate alert if unhealthy
        if not metric.is_healthy:
            severity = AlertSeverity.CRITICAL if z_score < -3.0 else AlertSeverity.WARNING

            pct_change = ((current_wr - baseline_wr) / baseline_wr * 100) if baseline_wr > 0 else 0

            alert = HealthAlert(
                severity=severity,
                title="Win Rate Anomaly Detected",
                message=(
                    f"Win rate dropped to {current_wr:.1%} (baseline: {baseline_wr:.1%}, {pct_change:+.1f}%)\n"
                    f"Z-score: {z_score:.2f} ({abs(z_score):.1f} std deviations below normal)\n"
                    f"Recent trades: {current_trades} (last {self.config.comparison_window_hours}h)"
                ),
                metric=metric,
                suggested_actions=[
                    "Review recent losing trades for common patterns",
                    "Check if market regime changed (volatility spike, spread widening)",
                    "Verify data quality and execution issues",
                    "Consider pausing trading until issue identified",
                ] if severity == AlertSeverity.CRITICAL else [
                    "Monitor closely for continued degradation",
                    "Review pattern performance for failing strategies",
                ],
                context=metric.context,
                timestamp=metric.timestamp,
                alert_id=f"win_rate_anomaly_{metric.timestamp.timestamp()}",
            )

            alerts.append(alert)
            logger.warning("win_rate_anomaly", **metric.context)

        return alerts

    def check_profit_anomaly(self) -> List[HealthAlert]:
        """Check for unusual profit/loss patterns."""
        alerts = []
        
        try:
            self.connect()
            
            with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Baseline P&L statistics
                cur.execute(
                    """
                    SELECT
                        AVG(net_profit_gbp) as avg_profit,
                        STDDEV(net_profit_gbp) as stddev_profit,
                        SUM(net_profit_gbp) as total_profit,
                        COUNT(*) as total_trades
                    FROM trade_memory
                    WHERE entry_timestamp >= NOW() - INTERVAL '%s days'
                      AND entry_timestamp < NOW() - INTERVAL '%s hours'
                      AND is_winner IS NOT NULL
                    """,
                    (self.config.baseline_window_days, self.config.comparison_window_hours),
                )
                baseline = cur.fetchone()

                # Current P&L
                cur.execute(
                    """
                    SELECT
                        AVG(net_profit_gbp) as avg_profit,
                        SUM(net_profit_gbp) as total_profit,
                        COUNT(*) as total_trades
                    FROM trade_memory
                    WHERE entry_timestamp >= NOW() - INTERVAL '%s hours'
                      AND is_winner IS NOT NULL
                    """,
                    (self.config.comparison_window_hours,),
                )
                current = cur.fetchone()
        except psycopg2.errors.UndefinedTable:
            logger.debug("table_trade_memory_not_exists", message="Table doesn't exist yet - skipping profit check")
            self._conn.rollback()
            return alerts
        except Exception as e:
            logger.warning("profit_check_failed", error=str(e))
            if self._conn:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
            return alerts

        if not baseline or not current or current["total_trades"] < 20:
            return alerts

        baseline_avg = float(baseline["avg_profit"] or 0.0)
        baseline_std = float(baseline["stddev_profit"] or 0.5)
        current_avg = float(current["avg_profit"] or 0.0)
        current_total = float(current["total_profit"] or 0.0)

        # Z-score for average profit
        z_score = (current_avg - baseline_avg) / (baseline_std + 1e-6)

        # Alert if profit significantly negative
        if z_score < -self.config.profit_stddev_threshold and current_total < -10.0:
            metric = HealthMetric(
                name="profit_per_trade",
                value=current_avg,
                baseline=baseline_avg,
                threshold=baseline_avg - (self.config.profit_stddev_threshold * baseline_std),
                is_healthy=False,
                timestamp=datetime.now(tz=timezone.utc),
                context={
                    "z_score": z_score,
                    "current_total_profit": current_total,
                    "current_trades": current["total_trades"],
                },
            )

            alert = HealthAlert(
                severity=AlertSeverity.CRITICAL if current_total < -50.0 else AlertSeverity.WARNING,
                title="Profit Anomaly - Unusual Losses",
                message=(
                    f"Average profit per trade: £{current_avg:.2f} (baseline: £{baseline_avg:.2f})\n"
                    f"Total P&L (24h): £{current_total:.2f}\n"
                    f"Z-score: {z_score:.2f} ({abs(z_score):.1f}σ below normal)"
                ),
                metric=metric,
                suggested_actions=[
                    "Review largest losing trades",
                    "Check for execution issues (slippage, fees)",
                    "Verify cost model accuracy",
                    "Consider reducing position sizes",
                ],
                context=metric.context,
                timestamp=metric.timestamp,
                alert_id=f"profit_anomaly_{metric.timestamp.timestamp()}",
            )

            alerts.append(alert)
            logger.warning("profit_anomaly", total_loss=current_total, z_score=z_score)

        return alerts

    def check_trade_volume(self) -> List[HealthAlert]:
        """Check for unusual trade volume (too few trades)."""
        alerts = []
        
        try:
            self.connect()
            
            with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Baseline trades per hour
                cur.execute(
                    """
                    SELECT COUNT(*) / NULLIF(%s, 0) as trades_per_hour
                    FROM trade_memory
                    WHERE entry_timestamp >= NOW() - INTERVAL '%s days'
                      AND entry_timestamp < NOW() - INTERVAL '%s hours'
                    """,
                    (
                        self.config.baseline_window_days * 24,
                        self.config.baseline_window_days,
                        self.config.comparison_window_hours,
                    ),
                )
                baseline = cur.fetchone()

                # Current trades per hour
                cur.execute(
                    """
                    SELECT COUNT(*) / NULLIF(%s, 0) as trades_per_hour,
                           COUNT(*) as total_trades
                    FROM trade_memory
                    WHERE entry_timestamp >= NOW() - INTERVAL '%s hours'
                    """,
                    (self.config.comparison_window_hours, self.config.comparison_window_hours),
                )
                current = cur.fetchone()
        except psycopg2.errors.UndefinedTable:
            logger.debug("table_trade_memory_not_exists", message="Table doesn't exist yet - skipping volume check")
            self._conn.rollback()
            return alerts
        except Exception as e:
            logger.warning("volume_check_failed", error=str(e))
            if self._conn:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
            return alerts

        if not baseline or not current:
            return alerts

        baseline_tph = float(baseline["trades_per_hour"] or 0.0)
        current_tph = float(current["trades_per_hour"] or 0.0)
        current_total = current["total_trades"]

        # Check if volume dropped significantly
        if baseline_tph > 0:
            drop_pct = (baseline_tph - current_tph) / baseline_tph

            if drop_pct > self.config.trade_volume_drop_pct:
                metric = HealthMetric(
                    name="trade_volume",
                    value=current_tph,
                    baseline=baseline_tph,
                    threshold=baseline_tph * (1 - self.config.trade_volume_drop_pct),
                    is_healthy=False,
                    timestamp=datetime.now(tz=timezone.utc),
                    context={
                        "drop_percentage": drop_pct * 100,
                        "current_total_trades": current_total,
                        "expected_trades": baseline_tph * self.config.comparison_window_hours,
                    },
                )

                alert = HealthAlert(
                    severity=AlertSeverity.WARNING if current_total > 5 else AlertSeverity.CRITICAL,
                    title="Trade Volume Drop Detected",
                    message=(
                        f"Trade volume: {current_tph:.1f}/hour (baseline: {baseline_tph:.1f}/hour)\n"
                        f"Drop: {drop_pct*100:.1f}% below normal\n"
                        f"Total trades (24h): {current_total} (expected: ~{baseline_tph * self.config.comparison_window_hours:.0f})"
                    ),
                    metric=metric,
                    suggested_actions=[
                        "Check if trading is paused or patterns blacklisted",
                        "Verify exchange connectivity and data feeds",
                        "Review pattern confidence thresholds (may be too strict)",
                        "Check for market liquidity issues",
                    ],
                    context=metric.context,
                    timestamp=metric.timestamp,
                    alert_id=f"volume_drop_{metric.timestamp.timestamp()}",
                )

                alerts.append(alert)
                logger.warning("trade_volume_drop", drop_pct=drop_pct*100, current=current_total)

        return alerts

    def check_all(self) -> List[HealthAlert]:
        """Run all anomaly checks and return combined alerts."""
        all_alerts = []

        try:
            all_alerts.extend(self.check_win_rate())
        except Exception as exc:
            logger.exception("win_rate_check_failed", error=str(exc))

        try:
            all_alerts.extend(self.check_profit_anomaly())
        except Exception as exc:
            logger.exception("profit_check_failed", error=str(exc))

        try:
            all_alerts.extend(self.check_trade_volume())
        except Exception as exc:
            logger.exception("volume_check_failed", error=str(exc))

        logger.info("anomaly_detection_complete", alerts_generated=len(all_alerts))
        return all_alerts
