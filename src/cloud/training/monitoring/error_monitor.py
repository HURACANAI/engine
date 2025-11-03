"""Error monitoring and log analysis."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import structlog

from .types import AlertSeverity, ErrorSummary, HealthAlert

logger = structlog.get_logger(__name__)


@dataclass
class ErrorMonitorConfig:
    """Configuration for error monitoring."""
    error_spike_multiplier: float = 3.0  # 3x baseline = spike
    baseline_window_hours: int = 24
    comparison_window_minutes: int = 60
    critical_error_types: List[str] = None

    def __post_init__(self):
        if self.critical_error_types is None:
            self.critical_error_types = [
                "DATABASE_CONNECTION_ERROR",
                "EXCHANGE_API_ERROR",
                "TRADING_HALTED",
                "MEMORY_ERROR",
            ]


class ErrorMonitor:
    """
    Monitors application logs for errors and anomalies.

    Detects error spikes, recurring errors, and correlates
    errors with trading performance issues.
    """

    def __init__(self, config: Optional[ErrorMonitorConfig] = None):
        self.config = config or ErrorMonitorConfig()
        self._error_counts: Dict[str, List[datetime]] = defaultdict(list)

    def parse_logs(self, log_entries: List[Dict]) -> List[ErrorSummary]:
        """
        Parse log entries and categorize errors.

        Args:
            log_entries: List of log dictionaries from structlog

        Returns:
            List of error summaries
        """
        errors_by_type: Dict[str, List[Dict]] = defaultdict(list)

        for entry in log_entries:
            level = entry.get("log_level", "").upper()
            if level in ("ERROR", "CRITICAL", "EXCEPTION"):
                error_type = self._categorize_error(entry)
                errors_by_type[error_type].append(entry)

        # Create summaries
        summaries = []
        for error_type, entries in errors_by_type.items():
            if not entries:
                continue

            timestamps = [e.get("timestamp", datetime.now(tz=timezone.utc)) for e in entries]
            first_seen = min(timestamps) if timestamps else datetime.now(tz=timezone.utc)
            last_seen = max(timestamps) if timestamps else datetime.now(tz=timezone.utc)

            # Determine impact
            impact = self._assess_error_impact(error_type, len(entries))

            summary = ErrorSummary(
                error_type=error_type,
                count=len(entries),
                first_seen=first_seen,
                last_seen=last_seen,
                impact=impact,
                affected_trades=[],  # Would be populated from correlation analysis
                sample_messages=[
                    entry.get("event", "")[:200]  # First 200 chars
                    for entry in entries[:3]  # Up to 3 samples
                ],
            )
            summaries.append(summary)

        logger.info("log_parsing_complete", error_types=len(summaries), total_errors=len(log_entries))
        return summaries

    def _categorize_error(self, log_entry: Dict) -> str:
        """Categorize error by type based on log content."""
        event = log_entry.get("event", "").lower()
        error = str(log_entry.get("error", "")).lower()

        # Database errors
        if "psycopg2" in error or "database" in event or "sql" in event:
            return "DATABASE_ERROR"

        # Exchange/API errors
        if "exchange" in event or "api" in event or "connection" in error:
            if "timeout" in error:
                return "EXCHANGE_TIMEOUT"
            return "EXCHANGE_API_ERROR"

        # Data quality errors
        if "data_quality" in event or "validation" in event:
            return "DATA_QUALITY_ERROR"

        # Memory/Resource errors
        if "memory" in error or "out of memory" in error:
            return "MEMORY_ERROR"

        # Trading errors
        if "trade" in event and "fail" in event:
            return "TRADE_EXECUTION_ERROR"

        # Model errors
        if "model" in event or "training" in event:
            return "MODEL_ERROR"

        return "UNKNOWN_ERROR"

    def _assess_error_impact(self, error_type: str, count: int) -> str:
        """Assess the impact level of an error."""
        if error_type in self.config.critical_error_types:
            return "HIGH"

        if count > 50:
            return "MEDIUM"
        elif count > 10:
            return "LOW"
        else:
            return "NONE"

    def detect_error_spikes(self, summaries: List[ErrorSummary]) -> List[HealthAlert]:
        """Detect unusual spikes in error rates."""
        alerts = []

        for summary in summaries:
            error_type = summary.error_type
            current_count = summary.count

            # Track errors over time
            self._error_counts[error_type].append(datetime.now(tz=timezone.utc))

            # Clean old entries (beyond baseline window)
            cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=self.config.baseline_window_hours)
            self._error_counts[error_type] = [
                ts for ts in self._error_counts[error_type] if ts > cutoff
            ]

            # Calculate baseline rate
            baseline_count = len([
                ts for ts in self._error_counts[error_type]
                if ts < datetime.now(tz=timezone.utc) - timedelta(minutes=self.config.comparison_window_minutes)
            ])

            # Normalize to same time window
            baseline_rate = baseline_count / max(1, self.config.baseline_window_hours - 1)
            current_rate = current_count

            # Detect spike
            if baseline_rate > 0 and current_rate > baseline_rate * self.config.error_spike_multiplier:
                severity = (
                    AlertSeverity.CRITICAL
                    if summary.impact == "HIGH" or current_count > 100
                    else AlertSeverity.WARNING
                )

                alert = HealthAlert(
                    severity=severity,
                    title=f"Error Spike Detected: {error_type}",
                    message=(
                        f"Error rate spiked {current_rate / baseline_rate:.1f}x above baseline\n"
                        f"Current: {current_count} errors (last hour)\n"
                        f"Baseline: ~{baseline_rate:.1f} errors/hour\n"
                        f"Impact: {summary.impact}\n"
                        f"Sample: {summary.sample_messages[0] if summary.sample_messages else 'N/A'}"
                    ),
                    metric=None,
                    suggested_actions=self._get_error_suggestions(error_type),
                    context={
                        "error_type": error_type,
                        "current_count": current_count,
                        "baseline_rate": baseline_rate,
                        "spike_multiplier": current_rate / baseline_rate,
                        "first_seen": summary.first_seen.isoformat(),
                        "last_seen": summary.last_seen.isoformat(),
                    },
                    timestamp=datetime.now(tz=timezone.utc),
                    alert_id=f"error_spike_{error_type}_{datetime.now(tz=timezone.utc).timestamp()}",
                )

                alerts.append(alert)
                logger.warning(
                    "error_spike_detected",
                    error_type=error_type,
                    count=current_count,
                    baseline=baseline_rate,
                )

        return alerts

    def _get_error_suggestions(self, error_type: str) -> List[str]:
        """Get suggested actions for specific error types."""
        suggestions = {
            "DATABASE_ERROR": [
                "Check database connection and credentials",
                "Verify database is not at capacity",
                "Review slow queries and add indexes if needed",
                "Check if migrations pending",
            ],
            "EXCHANGE_TIMEOUT": [
                "Check exchange API status page",
                "Verify network connectivity",
                "Consider increasing timeout thresholds",
                "Switch to backup exchange if persistent",
            ],
            "EXCHANGE_API_ERROR": [
                "Check exchange API status",
                "Verify API credentials are valid",
                "Check rate limits not exceeded",
                "Review API usage patterns",
            ],
            "DATA_QUALITY_ERROR": [
                "Review data validation rules",
                "Check data source reliability",
                "Verify candle data completeness",
                "Investigate specific failing symbols",
            ],
            "MEMORY_ERROR": [
                "Check system memory usage",
                "Review batch sizes and reduce if needed",
                "Investigate memory leaks",
                "Consider scaling up resources",
            ],
            "TRADE_EXECUTION_ERROR": [
                "Check exchange connectivity",
                "Verify sufficient balance",
                "Review order parameters",
                "Check for symbol/market issues",
            ],
        }

        return suggestions.get(error_type, [
            "Review error logs for root cause",
            "Check recent system changes",
            "Monitor for continued issues",
        ])

    def check_recurring_errors(self, summaries: List[ErrorSummary]) -> List[HealthAlert]:
        """Detect errors that happen repeatedly (e.g., daily at same time)."""
        alerts = []

        # Group errors by type and check if they recur at similar times
        for summary in summaries:
            if summary.count < 5:  # Need multiple occurrences
                continue

            # Simple check: if errors happen consistently, flag as recurring
            time_span = (summary.last_seen - summary.first_seen).total_seconds()

            # If errors spread over >1 hour but happening consistently
            if time_span > 3600 and summary.count > 10:
                alert = HealthAlert(
                    severity=AlertSeverity.INFO,
                    title=f"Recurring Error Pattern: {summary.error_type}",
                    message=(
                        f"Error occurring repeatedly over {time_span/3600:.1f} hours\n"
                        f"Total occurrences: {summary.count}\n"
                        f"May indicate systematic issue rather than transient problem"
                    ),
                    metric=None,
                    suggested_actions=[
                        "Investigate root cause - may be configuration issue",
                        "Check if error correlates with specific time/event",
                        "Consider permanent fix rather than handling exceptions",
                    ],
                    context={
                        "error_type": summary.error_type,
                        "count": summary.count,
                        "time_span_hours": time_span / 3600,
                    },
                    timestamp=datetime.now(tz=timezone.utc),
                    alert_id=f"recurring_{summary.error_type}_{datetime.now(tz=timezone.utc).timestamp()}",
                )

                alerts.append(alert)

        return alerts
