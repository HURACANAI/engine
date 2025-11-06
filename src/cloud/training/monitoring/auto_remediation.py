"""Safe auto-remediation actions with comprehensive logging."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import psycopg2
import structlog

from .types import HealthAlert, AlertSeverity

logger = structlog.get_logger(__name__)


@dataclass
class RemediationAction:
    """Record of a remediation action taken."""
    action_id: str
    action_type: str
    description: str
    triggered_by_alert: str
    parameters: Dict[str, Any]
    success: bool
    error_message: Optional[str]
    timestamp: datetime
    reversible: bool
    reversal_command: Optional[str]


class AutoRemediator:
    """
    Takes safe, reversible corrective actions automatically.

    CRITICAL SAFETY RULES:
    1. NEVER modifies code or config files
    2. All actions are reversible
    3. All actions are logged comprehensively
    4. Only runtime state changes (pause patterns, log context)
    5. User can always override/reverse
    """

    def __init__(self, dsn: str, dry_run: bool = False):
        self.dsn = dsn
        self.dry_run = dry_run  # If True, log but don't execute
        self._conn: Optional[psycopg2.extensions.connection] = None
        self._action_history: List[RemediationAction] = []

        logger.info(
            "auto_remediator_initialized",
            dry_run=dry_run,
            safety_mode="ENABLED",
            rules="NO_CODE_CHANGES|REVERSIBLE_ONLY|FULL_LOGGING",
        )

    def connect(self) -> None:
        """Establish database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.dsn)
            logger.info("auto_remediation_db_connected")

    def close(self) -> None:
        """Close database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            logger.info("auto_remediation_db_closed")

    def handle_alert(self, alert: HealthAlert) -> Optional[RemediationAction]:
        """
        Determine and execute appropriate remediation for an alert.

        Returns RemediationAction if action taken, None otherwise.
        """
        logger.info(
            "evaluating_remediation",
            alert_id=alert.alert_id,
            severity=alert.severity.value,
            title=alert.title,
        )

        # Only auto-remediate CRITICAL alerts (user reviews warnings)
        if alert.severity != AlertSeverity.CRITICAL:
            logger.info("remediation_skipped", reason="not_critical", alert_id=alert.alert_id)
            return None

        # Determine action based on alert type
        action = None

        if "pattern" in alert.alert_id.lower() and "failing" in alert.message.lower():
            action = self._pause_failing_pattern(alert)
        elif "win_rate" in alert.alert_id and "anomaly" in alert.alert_id:
            action = self._log_win_rate_context(alert)
        elif "error_spike" in alert.alert_id:
            action = self._log_error_context(alert)

        if action:
            self._action_history.append(action)
            logger.info(
                "remediation_action_completed",
                action_id=action.action_id,
                action_type=action.action_type,
                success=action.success,
                reversible=action.reversible,
            )

        return action

    def _pause_failing_pattern(self, alert: HealthAlert) -> RemediationAction:
        """
        Pause a pattern that's critically failing.

        SAFE: Sets reliability_score to 0 (soft disable), easily reversible.
        DOES NOT: Delete pattern or modify code.
        """
        pattern_id = alert.context.get("pattern_id")

        if not pattern_id:
            logger.warning("cannot_pause_pattern", reason="no_pattern_id", alert_id=alert.alert_id)
            return self._create_failed_action(
                "pause_pattern",
                "No pattern ID in alert context",
                alert.alert_id,
            )

        logger.info(
            "attempting_pattern_pause",
            pattern_id=pattern_id,
            dry_run=self.dry_run,
            reason="critical_failure",
        )

        try:
            if not self.dry_run:
                self.connect()
                with self._conn.cursor() as cur:
                    # Soft disable by setting reliability to 0
                    cur.execute(
                        """
                        UPDATE pattern_library
                        SET
                            reliability_score = 0.0,
                            last_updated = NOW()
                        WHERE pattern_id = %s
                        RETURNING pattern_name, win_rate
                        """,
                        (pattern_id,),
                    )
                    result = cur.fetchone()
                    self._conn.commit()

                    pattern_name = result[0] if result else f"Pattern{pattern_id}"
                    win_rate = result[1] if result else 0.0

                    logger.info(
                        "pattern_paused",
                        pattern_id=pattern_id,
                        pattern_name=pattern_name,
                        previous_win_rate=win_rate,
                        method="reliability_score_0",
                    )
            else:
                pattern_name = f"Pattern{pattern_id}"
                logger.info("pattern_pause_simulated", pattern_id=pattern_id, dry_run=True)

            return RemediationAction(
                action_id=f"pause_pattern_{pattern_id}_{datetime.now(tz=timezone.utc).timestamp()}",
                action_type="PAUSE_PATTERN",
                description=f"Paused failing pattern '{pattern_name}' (ID: {pattern_id})",
                triggered_by_alert=alert.alert_id,
                parameters={"pattern_id": pattern_id},
                success=True,
                error_message=None,
                timestamp=datetime.now(tz=timezone.utc),
                reversible=True,
                reversal_command=f"UPDATE pattern_library SET reliability_score = 0.5 WHERE pattern_id = {pattern_id}",
            )

        except Exception as exc:
            logger.exception("pattern_pause_failed", pattern_id=pattern_id, error=str(exc))
            return self._create_failed_action(
                "pause_pattern",
                str(exc),
                alert.alert_id,
                {"pattern_id": pattern_id},
            )

    def _log_win_rate_context(self, alert: HealthAlert) -> RemediationAction:
        """
        Log comprehensive context for win rate anomaly.

        SAFE: Only logs data, no state changes.
        """
        logger.info(
            "logging_win_rate_context",
            alert_id=alert.alert_id,
            current_value=alert.metric.value if alert.metric else None,
            baseline=alert.metric.baseline if alert.metric else None,
        )

        try:
            self.connect()
            with self._conn.cursor() as cur:
                # Get recent losing trades for analysis
                cur.execute(
                    """
                    SELECT
                        symbol,
                        exit_reason,
                        market_regime,
                        gross_profit_bps,
                        entry_timestamp
                    FROM trade_memory
                    WHERE is_winner = FALSE
                      AND entry_timestamp >= NOW() - INTERVAL '24 hours'
                    ORDER BY entry_timestamp DESC
                    LIMIT 20
                    """
                )
                recent_losses = cur.fetchall()

                logger.info(
                    "win_rate_context_captured",
                    recent_losses_count=len(recent_losses),
                    loss_reasons=[row[1] for row in recent_losses],
                    affected_symbols=list(set(row[0] for row in recent_losses)),
                    regimes=list(set(row[2] for row in recent_losses)),
                )

            return RemediationAction(
                action_id=f"log_context_{datetime.now(tz=timezone.utc).timestamp()}",
                action_type="LOG_CONTEXT",
                description="Logged detailed context for win rate investigation",
                triggered_by_alert=alert.alert_id,
                parameters={"losses_analyzed": len(recent_losses)},
                success=True,
                error_message=None,
                timestamp=datetime.now(tz=timezone.utc),
                reversible=True,  # Logging is always reversible (just info)
                reversal_command=None,
            )

        except Exception as exc:
            logger.exception("context_logging_failed", error=str(exc))
            return self._create_failed_action("log_context", str(exc), alert.alert_id)

    def _log_error_context(self, alert: HealthAlert) -> RemediationAction:
        """
        Log comprehensive context for error spike.

        SAFE: Only logs data, no state changes.
        """
        error_type = alert.context.get("error_type", "UNKNOWN")

        logger.info(
            "logging_error_context",
            error_type=error_type,
            count=alert.context.get("current_count"),
            spike_multiplier=alert.context.get("spike_multiplier"),
        )

        # Log all relevant context for investigation
        logger.info(
            "error_spike_full_context",
            **alert.context,
            alert_message=alert.message,
            suggested_actions=alert.suggested_actions,
        )

        return RemediationAction(
            action_id=f"log_errors_{datetime.now(tz=timezone.utc).timestamp()}",
            action_type="LOG_ERROR_CONTEXT",
            description=f"Logged context for {error_type} error spike",
            triggered_by_alert=alert.alert_id,
            parameters=alert.context,
            success=True,
            error_message=None,
            timestamp=datetime.now(tz=timezone.utc),
            reversible=True,
            reversal_command=None,
        )

    def _create_failed_action(
        self,
        action_type: str,
        error: str,
        alert_id: str,
        parameters: Optional[Dict] = None,
    ) -> RemediationAction:
        """Create a failed remediation action record."""
        logger.error(
            "remediation_action_failed",
            action_type=action_type,
            error=error,
            alert_id=alert_id,
        )

        return RemediationAction(
            action_id=f"{action_type}_failed_{datetime.now(tz=timezone.utc).timestamp()}",
            action_type=action_type,
            description=f"Failed to execute {action_type}",
            triggered_by_alert=alert_id,
            parameters=parameters or {},
            success=False,
            error_message=error,
            timestamp=datetime.now(tz=timezone.utc),
            reversible=False,
            reversal_command=None,
        )

    def reverse_action(self, action: RemediationAction) -> bool:
        """
        Reverse a previous remediation action.

        Returns True if successful, False otherwise.
        """
        if not action.reversible:
            logger.warning(
                "action_not_reversible",
                action_id=action.action_id,
                action_type=action.action_type,
            )
            return False

        if not action.reversal_command:
            logger.warning(
                "no_reversal_command",
                action_id=action.action_id,
                message="Action reversible but no command provided",
            )
            return False

        logger.info(
            "reversing_action",
            action_id=action.action_id,
            action_type=action.action_type,
            command=action.reversal_command,
        )

        try:
            if not self.dry_run:
                self.connect()
                with self._conn.cursor() as cur:
                    cur.execute(action.reversal_command)
                    self._conn.commit()

            logger.info(
                "action_reversed",
                action_id=action.action_id,
                action_type=action.action_type,
                success=True,
            )
            return True

        except Exception as exc:
            logger.exception(
                "action_reversal_failed",
                action_id=action.action_id,
                error=str(exc),
            )
            return False

    def get_action_history(self, hours: int = 24) -> List[RemediationAction]:
        """Get recent remediation actions."""
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=hours)
        recent = [a for a in self._action_history if a.timestamp >= cutoff]

        logger.info(
            "action_history_retrieved",
            total_actions=len(self._action_history),
            recent_actions=len(recent),
            hours=hours,
        )

        return recent

    def get_statistics(self) -> Dict[str, Any]:
        """Get remediation statistics."""
        total = len(self._action_history)
        successful = sum(1 for a in self._action_history if a.success)
        by_type = {}

        for action in self._action_history:
            by_type[action.action_type] = by_type.get(action.action_type, 0) + 1

        stats = {
            "total_actions": total,
            "successful_actions": successful,
            "failed_actions": total - successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "actions_by_type": by_type,
        }

        logger.info("remediation_statistics", **stats)
        return stats
