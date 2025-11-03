"""Enhanced alert management with intelligent routing and deduplication."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set

import requests
import structlog

from ..config.settings import NotificationSettings
from .types import AlertSeverity, HealthAlert

logger = structlog.get_logger(__name__)


class AlertManager:
    """
    Manages alert lifecycle: deduplication, prioritization, routing, and delivery.

    Prevents alert spam while ensuring critical issues are immediately visible.
    """

    def __init__(self, notification_settings: NotificationSettings):
        self.settings = notification_settings
        self._sent_alerts: Dict[str, datetime] = {}  # alert_id -> last sent time
        self._pending_alerts: Dict[AlertSeverity, List[HealthAlert]] = defaultdict(list)
        self._muted_alert_types: Set[str] = set()

    def add_alert(self, alert: HealthAlert) -> None:
        """Add an alert to the queue."""
        # Check if recently sent (deduplication)
        if self._is_duplicate(alert):
            logger.debug("alert_deduplicated", alert_id=alert.alert_id)
            return

        # Check if muted
        if self._is_muted(alert):
            logger.debug("alert_muted", alert_id=alert.alert_id)
            return

        # Add to appropriate queue
        self._pending_alerts[alert.severity].append(alert)
        logger.info("alert_queued", severity=alert.severity.value, title=alert.title)

    def _is_duplicate(self, alert: HealthAlert, window_minutes: int = 60) -> bool:
        """Check if this alert was recently sent."""
        if alert.alert_id not in self._sent_alerts:
            return False

        last_sent = self._sent_alerts[alert.alert_id]
        elapsed = (datetime.now(tz=timezone.utc) - last_sent).total_seconds() / 60

        return elapsed < window_minutes

    def _is_muted(self, alert: HealthAlert) -> bool:
        """Check if this alert type is muted."""
        # Extract alert type from ID
        alert_type = alert.alert_id.split("_")[0]
        return alert_type in self._muted_alert_types

    def send_immediate(self) -> None:
        """Send all CRITICAL alerts immediately."""
        critical_alerts = self._pending_alerts[AlertSeverity.CRITICAL]

        if not critical_alerts:
            return

        for alert in critical_alerts:
            self._send_to_telegram(alert)
            self._sent_alerts[alert.alert_id] = datetime.now(tz=timezone.utc)

        logger.info("critical_alerts_sent", count=len(critical_alerts))
        self._pending_alerts[AlertSeverity.CRITICAL].clear()

    def send_digest(self, severity: AlertSeverity = AlertSeverity.WARNING) -> None:
        """Send a digest of WARNING alerts."""
        alerts = self._pending_alerts[severity]

        if not alerts:
            return

        # Group by category
        grouped: Dict[str, List[HealthAlert]] = defaultdict(list)
        for alert in alerts:
            category = alert.title.split(":")[0]  # Extract category from title
            grouped[category].append(alert)

        # Build digest message
        message_parts = [
            f"⚠️ {severity.value} Alert Digest ({len(alerts)} alerts)\n",
            "=" * 50,
        ]

        for category, category_alerts in grouped.items():
            message_parts.append(f"\n📌 {category} ({len(category_alerts)})")
            for alert in category_alerts[:3]:  # Max 3 per category in digest
                message_parts.append(f"  • {alert.message.split(chr(10))[0]}")  # First line only

            if len(category_alerts) > 3:
                message_parts.append(f"  ... and {len(category_alerts) - 3} more")

        full_message = "\n".join(message_parts)
        self._send_telegram_message(full_message)

        # Mark as sent
        for alert in alerts:
            self._sent_alerts[alert.alert_id] = datetime.now(tz=timezone.utc)

        logger.info("digest_sent", severity=severity.value, count=len(alerts))
        self._pending_alerts[severity].clear()

    def send_daily_report(self) -> None:
        """Send comprehensive daily health report."""
        all_alerts = []
        for severity_alerts in self._pending_alerts.values():
            all_alerts.extend(severity_alerts)

        if not all_alerts:
            # Send "all clear" message
            self._send_telegram_message(
                "✅ Daily Health Report\n"
                "No issues detected in the past 24 hours.\n"
                "System operating normally."
            )
            return

        # Build comprehensive report
        message_parts = [
            "📊 Daily Health Report",
            "=" * 50,
            f"Date: {datetime.now(tz=timezone.utc).date().isoformat()}",
            f"Total alerts: {len(all_alerts)}\n",
        ]

        # Group by severity
        by_severity = defaultdict(list)
        for alert in all_alerts:
            by_severity[alert.severity].append(alert)

        for severity in [AlertSeverity.CRITICAL, AlertSeverity.WARNING, AlertSeverity.INFO]:
            alerts = by_severity[severity]
            if not alerts:
                continue

            emoji = "🚨" if severity == AlertSeverity.CRITICAL else "⚠️" if severity == AlertSeverity.WARNING else "ℹ️"
            message_parts.append(f"\n{emoji} {severity.value} ({len(alerts)})")

            for alert in alerts[:5]:  # Max 5 per severity
                message_parts.append(f"  • {alert.title}")

            if len(alerts) > 5:
                message_parts.append(f"  ... and {len(alerts) - 5} more")

        full_message = "\n".join(message_parts)
        self._send_telegram_message(full_message)

        # Clear all
        for alert in all_alerts:
            self._sent_alerts[alert.alert_id] = datetime.now(tz=timezone.utc)

        self._pending_alerts.clear()
        logger.info("daily_report_sent", total_alerts=len(all_alerts))

    def _send_to_telegram(self, alert: HealthAlert) -> None:
        """Send individual alert to Telegram with rich formatting."""
        # Format alert
        emoji = self._get_severity_emoji(alert.severity)

        message_parts = [
            f"{emoji} {alert.severity.value}: {alert.title}",
            "=" * 40,
            alert.message,
        ]

        if alert.suggested_actions:
            message_parts.append("\n🔧 Suggested Actions:")
            for i, action in enumerate(alert.suggested_actions, 1):
                message_parts.append(f"{i}. {action}")

        message_parts.append(f"\nTime: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")

        full_message = "\n".join(message_parts)
        self._send_telegram_message(full_message)

    def _send_telegram_message(self, message: str) -> bool:
        """Send message to Telegram."""
        if not self.settings.telegram_enabled:
            logger.info("telegram_disabled", message=message[:100])
            return False

        if not self.settings.telegram_webhook_url or not self.settings.telegram_chat_id:
            logger.warning("telegram_not_configured")
            return False

        try:
            payload = {
                "chat_id": self.settings.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML",  # Support rich formatting
            }

            response = requests.post(
                self.settings.telegram_webhook_url,
                json=payload,
                timeout=10,
            )

            if response.ok:
                logger.info("telegram_sent", chars=len(message))
                return True
            else:
                logger.error(
                    "telegram_failed",
                    status=response.status_code,
                    text=response.text[:200],
                )
                return False

        except Exception as exc:
            logger.exception("telegram_exception", error=str(exc))
            return False

    def _get_severity_emoji(self, severity: AlertSeverity) -> str:
        """Get emoji for severity level."""
        return {
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.INFO: "ℹ️",
        }[severity]

    def mute_alert_type(self, alert_type: str, duration_hours: int = 24) -> None:
        """Temporarily mute a specific alert type."""
        self._muted_alert_types.add(alert_type)
        logger.info("alert_type_muted", type=alert_type, duration_hours=duration_hours)

    def unmute_alert_type(self, alert_type: str) -> None:
        """Unmute an alert type."""
        self._muted_alert_types.discard(alert_type)
        logger.info("alert_type_unmuted", type=alert_type)

    def get_pending_count(self) -> Dict[str, int]:
        """Get count of pending alerts by severity."""
        return {
            severity.value: len(alerts)
            for severity, alerts in self._pending_alerts.items()
        }

    def clear_old_sent_records(self, hours: int = 168) -> None:
        """Clear old sent alert records (default: 1 week)."""
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=hours)
        old_ids = [
            alert_id
            for alert_id, sent_time in self._sent_alerts.items()
            if sent_time < cutoff
        ]

        for alert_id in old_ids:
            del self._sent_alerts[alert_id]

        if old_ids:
            logger.info("old_alerts_cleared", count=len(old_ids))
