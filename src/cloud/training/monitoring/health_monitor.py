"""
Main health monitoring orchestrator.

Coordinates all monitoring components with comprehensive logging
at every step so you know exactly what's happening.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, List, Optional

import structlog

from ..config.settings import EngineSettings
from .alert_manager import AlertManager
from .anomaly_detector import AnomalyConfig, StatisticalAnomalyDetector
from .auto_remediation import AutoRemediator
from .error_monitor import ErrorMonitor, ErrorMonitorConfig
from .pattern_health import PatternHealthConfig, PatternHealthMonitor
from .system_status import SystemStatusReporter
from .types import HealthAlert
from .enhanced_health_check import EnhancedHealthChecker, ComprehensiveHealthReport

logger = structlog.get_logger(__name__)


class HealthMonitorOrchestrator:
    """
    Main health monitoring system orchestrator.

    Runs all health checks on schedule, logs everything extensively,
    and coordinates alerts and remediation.
    """

    def __init__(self, settings: EngineSettings, dsn: str, dropbox_sync: Optional[Any] = None):
        self.settings = settings
        self.dsn = dsn
        self.dropbox_sync = dropbox_sync

        logger.info(
            "===== INITIALIZING HEALTH MONITOR =====",
            operation="HEALTH_MONITOR_INIT",
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            monitoring_enabled=True,  # Always enabled for visibility
        )

        # Initialize all monitoring components
        self.anomaly_detector = StatisticalAnomalyDetector(
            dsn=dsn,
            config=AnomalyConfig(),
        )
        logger.info("component_initialized", component="AnomalyDetector", status="OK")

        self.pattern_monitor = PatternHealthMonitor(
            dsn=dsn,
            config=PatternHealthConfig(),
        )
        logger.info("component_initialized", component="PatternHealthMonitor", status="OK")

        self.error_monitor = ErrorMonitor(
            config=ErrorMonitorConfig(),
        )
        logger.info("component_initialized", component="ErrorMonitor", status="OK")

        self.alert_manager = AlertManager(
            notification_settings=settings.notifications,
        )
        logger.info(
            "component_initialized",
            component="AlertManager",
            telegram_enabled=settings.notifications.telegram_enabled,
            status="OK",
        )

        self.auto_remediator = AutoRemediator(
            dsn=dsn,
            dry_run=False,  # Set to True for testing
        )
        logger.info("component_initialized", component="AutoRemediator", status="OK")

        self.system_status = SystemStatusReporter(dsn=dsn)
        logger.info("component_initialized", component="SystemStatusReporter", status="OK")

        # Initialize enhanced health checker
        try:
            self.enhanced_health_checker = EnhancedHealthChecker(
                dsn=dsn,
                settings=settings,
                dropbox_sync=dropbox_sync,
            )
            logger.info("component_initialized", component="EnhancedHealthChecker", status="OK")
        except Exception as e:
            logger.warning("enhanced_health_checker_init_failed", error=str(e))
            self.enhanced_health_checker = None

        self._running = False
        self._check_count = 0
        self._last_check_time: Optional[datetime] = None
        self._last_enhanced_report: Optional[ComprehensiveHealthReport] = None

        logger.info(
            "===== HEALTH MONITOR INITIALIZED =====",
            all_components="READY",
            check_interval_seconds=300,  # 5 minutes
        )

        # Log startup status
        self.system_status.log_startup_status()

    def run_health_check(self) -> List[HealthAlert]:
        """
        Run complete health check cycle.

        Logs every step for complete visibility.
        """
        check_start = datetime.now(tz=timezone.utc)
        self._check_count += 1

        logger.info(
            "===== STARTING HEALTH CHECK =====",
            check_number=self._check_count,
            timestamp=check_start.isoformat(),
            operation="HEALTH_CHECK_CYCLE",
        )

        all_alerts: List[HealthAlert] = []

        # Step 0: Enhanced comprehensive health check (NEW - runs FIRST for complete visibility)
        enhanced_report = None
        if self.enhanced_health_checker:
            logger.info("health_check_step", step=0, operation="ENHANCED_COMPREHENSIVE_HEALTH_CHECK")
            try:
                enhanced_report = self.enhanced_health_checker.run_comprehensive_check()
                self._last_enhanced_report = enhanced_report
                logger.info(
                    "enhanced_health_check_complete",
                    overall_status=enhanced_report.overall_status,
                    total_checks=len(enhanced_report.checks),
                    healthy=enhanced_report.summary.get("healthy", 0),
                    warnings=enhanced_report.summary.get("warnings", 0),
                    critical=enhanced_report.summary.get("critical", 0),
                    disabled=enhanced_report.summary.get("disabled", 0),
                )
                # Log each check result
                for check in enhanced_report.checks:
                    logger.info(
                        "health_check_result",
                        name=check.name,
                        status=check.status,
                        message=check.message,
                        issues_count=len(check.issues),
                        recommendations_count=len(check.recommendations),
                    )
            except Exception as exc:
                logger.exception("enhanced_health_check_failed", error=str(exc))

        # Step 1: System status check (legacy - still runs for compatibility)
        logger.info("health_check_step", step=1, operation="SYSTEM_STATUS_CHECK")
        try:
            system_report = self.system_status.generate_full_report()
            logger.info(
                "system_status_checked",
                overall_status=system_report.overall_status,
                services_healthy=sum(1 for s in system_report.services if s.healthy),
                services_total=len(system_report.services),
            )
        except Exception as exc:
            logger.exception("system_status_check_failed", error=str(exc))

        # Step 2: Anomaly detection
        logger.info("health_check_step", step=2, operation="ANOMALY_DETECTION")
        try:
            anomaly_alerts = self.anomaly_detector.check_all()
            all_alerts.extend(anomaly_alerts)
            logger.info(
                "anomaly_detection_completed",
                alerts_generated=len(anomaly_alerts),
                critical=sum(1 for a in anomaly_alerts if a.severity.value == "CRITICAL"),
                warning=sum(1 for a in anomaly_alerts if a.severity.value == "WARNING"),
            )
        except Exception as exc:
            logger.exception("anomaly_detection_failed", error=str(exc))

        # Step 3: Pattern health check
        logger.info("health_check_step", step=3, operation="PATTERN_HEALTH_CHECK")
        try:
            pattern_reports = self.pattern_monitor.check_all_patterns()
            pattern_alerts = self.pattern_monitor.generate_pattern_alerts(pattern_reports)
            all_alerts.extend(pattern_alerts)

            logger.info(
                "pattern_health_checked",
                patterns_checked=len(pattern_reports),
                alerts_generated=len(pattern_alerts),
                failing_patterns=sum(1 for r in pattern_reports if r.status.value == "CRITICAL"),
            )

            # Log each pattern status
            for report in pattern_reports:
                logger.info(
                    "pattern_status_detail",
                    pattern_id=report.pattern_id,
                    pattern_name=report.pattern_name,
                    win_rate=report.current_win_rate,
                    baseline=report.baseline_win_rate,
                    status=report.status.value,
                    degradation_pct=report.degradation_pct,
                )

        except Exception as exc:
            logger.exception("pattern_health_check_failed", error=str(exc))

        # Step 4: Overfitting detection
        logger.info("health_check_step", step=4, operation="OVERFITTING_DETECTION")
        try:
            overfit_alerts = self.pattern_monitor.check_for_overfitting()
            all_alerts.extend(overfit_alerts)
            logger.info(
                "overfitting_check_completed",
                overfitted_patterns=len(overfit_alerts),
            )
        except Exception as exc:
            logger.exception("overfitting_detection_failed", error=str(exc))

        # Step 5: Error monitoring (would parse recent logs)
        logger.info("health_check_step", step=5, operation="ERROR_MONITORING")
        # Note: In production, would fetch recent logs from logging system
        # For now, log that this step is ready
        logger.info("error_monitoring_ready", status="AWAITING_LOG_INTEGRATION")

        # Step 6: Process alerts
        logger.info(
            "health_check_step",
            step=6,
            operation="ALERT_PROCESSING",
            total_alerts=len(all_alerts),
        )

        for alert in all_alerts:
            logger.info(
                "alert_generated",
                alert_id=alert.alert_id,
                severity=alert.severity.value,
                title=alert.title,
                timestamp=alert.timestamp.isoformat(),
            )

            # Add to alert manager
            self.alert_manager.add_alert(alert)

            # Attempt auto-remediation for critical alerts
            if alert.severity.value == "CRITICAL":
                logger.info("attempting_auto_remediation", alert_id=alert.alert_id)
                remediation = self.auto_remediator.handle_alert(alert)

                if remediation:
                    logger.info(
                        "remediation_executed",
                        alert_id=alert.alert_id,
                        action_type=remediation.action_type,
                        success=remediation.success,
                        reversible=remediation.reversible,
                    )
                else:
                    logger.info(
                        "no_remediation_taken",
                        alert_id=alert.alert_id,
                        reason="REQUIRES_MANUAL_INTERVENTION",
                    )

        # Step 7: Send alerts
        logger.info("health_check_step", step=7, operation="ALERT_DELIVERY")

        # Send critical alerts immediately
        critical_count = self.alert_manager.get_pending_count().get("CRITICAL", 0)
        if critical_count > 0:
            logger.info("sending_critical_alerts", count=critical_count)
            self.alert_manager.send_immediate()

        # Completion
        check_duration = (datetime.now(tz=timezone.utc) - check_start).total_seconds()
        self._last_check_time = check_start

        logger.info(
            "===== HEALTH CHECK COMPLETE =====",
            check_number=self._check_count,
            duration_seconds=check_duration,
            total_alerts=len(all_alerts),
            critical_alerts=sum(1 for a in all_alerts if a.severity.value == "CRITICAL"),
            warning_alerts=sum(1 for a in all_alerts if a.severity.value == "WARNING"),
            remediation_actions=len(self.auto_remediator.get_action_history(hours=1)),
            enhanced_report_available=enhanced_report is not None,
        )

        return all_alerts
    
    def get_enhanced_health_report(self) -> Optional[ComprehensiveHealthReport]:
        """Get the last enhanced health check report."""
        return self._last_enhanced_report

    def run_continuous(self, interval_seconds: int = 300) -> None:
        """
        Run health checks continuously.

        Args:
            interval_seconds: Time between checks (default: 300 = 5 minutes)
        """
        self._running = True

        logger.info(
            "===== STARTING CONTINUOUS MONITORING =====",
            interval_seconds=interval_seconds,
            check_frequency=f"Every {interval_seconds/60:.1f} minutes",
        )

        try:
            while self._running:
                logger.info(
                    "continuous_monitor_cycle_start",
                    check_count=self._check_count + 1,
                    uptime_checks=self._check_count,
                )

                # Run health check
                try:
                    alerts = self.run_health_check()
                except Exception as exc:
                    logger.exception("health_check_failed", error=str(exc))

                # Wait for next cycle
                logger.info(
                    "continuous_monitor_waiting",
                    next_check_in_seconds=interval_seconds,
                    next_check_at=(datetime.now(tz=timezone.utc) + timedelta(seconds=interval_seconds)).isoformat(),
                )

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("continuous_monitoring_interrupted", reason="KEYBOARD_INTERRUPT")
            self.stop()

        except Exception as exc:
            logger.exception("continuous_monitoring_failed", error=str(exc))
            self.stop()

    def stop(self) -> None:
        """Stop continuous monitoring."""
        self._running = False

        logger.info(
            "===== STOPPING HEALTH MONITOR =====",
            total_checks_performed=self._check_count,
            uptime_hours=(datetime.now(tz=timezone.utc) - self._last_check_time).total_seconds() / 3600 if self._last_check_time else 0,
        )

        # Send final daily report
        logger.info("sending_final_daily_report")
        self.alert_manager.send_daily_report()

        # Log final statistics
        remediation_stats = self.auto_remediator.get_statistics()
        logger.info("final_remediation_stats", **remediation_stats)

        logger.info("health_monitor_stopped", status="SHUTDOWN_COMPLETE")

    def send_hourly_digest(self) -> None:
        """
        Send hourly digest of warnings.
        
        NOTE: This is used by Engine for monitoring during daily training.
        May also be used by Mechanic for hourly updates (future).
        """
        logger.info("sending_hourly_digest", operation="HOURLY_ALERT_DIGEST")
        self.alert_manager.send_digest()

    def send_daily_report(self) -> None:
        """Send comprehensive daily report."""
        logger.info("sending_daily_report", operation="DAILY_HEALTH_REPORT")

        # Generate full system report
        system_report = self.system_status.generate_full_report()

        logger.info(
            "daily_report_system_summary",
            overall_status=system_report.overall_status,
            active_features=len(system_report.active_features),
            trades_24h=system_report.recent_activity.get("trades_24h", 0),
        )

        # Send via alert manager
        self.alert_manager.send_daily_report()

    def get_current_status(self) -> dict:
        """Get current monitoring status."""
        status = {
            "running": self._running,
            "total_checks": self._check_count,
            "last_check": self._last_check_time.isoformat() if self._last_check_time else None,
            "pending_alerts": self.alert_manager.get_pending_count(),
            "remediation_actions_24h": len(self.auto_remediator.get_action_history(hours=24)),
        }

        logger.info("current_monitoring_status", **status)
        return status
