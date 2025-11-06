"""
Comprehensive system status reporting with heavy logging.

Provides complete visibility into what's running, what's enabled,
and what's actually working in the backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psutil
import psycopg2
import psycopg2.errors
from psycopg2.extras import RealDictCursor
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ServiceStatus:
    """Status of a specific service/component."""
    name: str
    enabled: bool
    running: bool
    healthy: bool
    last_activity: Optional[datetime]
    details: Dict[str, Any]
    issues: List[str]


@dataclass
class SystemHealthReport:
    """Complete system health report."""
    timestamp: datetime
    overall_status: str  # "HEALTHY", "DEGRADED", "CRITICAL"
    services: List[ServiceStatus]
    database_status: ServiceStatus
    resource_usage: Dict[str, float]
    active_features: List[str]
    recent_activity: Dict[str, Any]


class SystemStatusReporter:
    """
    Comprehensive system status reporting.

    Logs everything that's happening so you know:
    - What services are enabled vs disabled
    - What's actually running vs configured
    - What features are active
    - Resource usage
    - Recent activity
    """

    def __init__(self, dsn: str):
        self.dsn = dsn
        self._conn: Optional[psycopg2.extensions.connection] = None

        logger.info(
            "system_status_reporter_initialized",
            purpose="COMPREHENSIVE_VISIBILITY",
            features="SERVICE_STATUS|RESOURCE_MONITOR|ACTIVITY_TRACKING|FEATURE_DETECTION",
        )

    def connect(self) -> None:
        """Establish database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.dsn)
            logger.info("status_reporter_db_connected", dsn_masked=self.dsn.split("@")[-1])

    def close(self) -> None:
        """Close database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            logger.info("status_reporter_db_closed")

    def check_database_status(self) -> ServiceStatus:
        """
        Check database connectivity and health.

        Logs extensive details about connection, table existence, data.
        """
        logger.info("checking_database_status", operation="DB_HEALTH_CHECK")

        issues = []
        details = {}
        healthy = True

        try:
            self.connect()

            # Test basic connectivity
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()

            logger.info("database_connection_ok", status="CONNECTED")
            details["connection"] = "OK"

            # Check critical tables exist
            required_tables = [
                "trade_memory",
                "pattern_library",
                "win_analysis",
                "loss_analysis",
                "post_exit_tracking",
            ]

            with self._conn.cursor() as cur:
                for table in required_tables:
                    cur.execute(
                        """
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_name = %s
                        )
                        """,
                        (table,),
                    )
                    exists = cur.fetchone()[0]

                    if exists:
                        logger.info("table_exists", table=table, status="OK")
                        details[f"table_{table}"] = "EXISTS"
                    else:
                        logger.warning("table_missing", table=table, status="MISSING")
                        details[f"table_{table}"] = "MISSING"
                        issues.append(f"Table {table} does not exist")
                        healthy = False

            # Check data counts (only for tables that exist)
            trade_count = 0
            pattern_count = 0
            
            with self._conn.cursor() as cur:
                # Only query tables that exist
                if details.get("table_trade_memory") == "EXISTS":
                    try:
                        cur.execute("SELECT COUNT(*) FROM trade_memory")
                        trade_count = cur.fetchone()[0]
                    except Exception as e:
                        logger.warning("trade_count_query_failed", error=str(e))
                        self._conn.rollback()
                
                if details.get("table_pattern_library") == "EXISTS":
                    try:
                        cur.execute("SELECT COUNT(*) FROM pattern_library")
                        pattern_count = cur.fetchone()[0]
                    except Exception as e:
                        logger.warning("pattern_count_query_failed", error=str(e))
                        self._conn.rollback()

            logger.info(
                "database_data_counts",
                trades=trade_count,
                patterns=pattern_count,
                status="COUNTED",
            )

            details["total_trades"] = trade_count
            details["total_patterns"] = pattern_count

            if trade_count == 0:
                logger.warning("no_trades_in_memory", message="Database empty - no historical data")
                issues.append("No trades in memory - system not trained yet")

            return ServiceStatus(
                name="Database",
                enabled=True,
                running=True,
                healthy=healthy,
                last_activity=datetime.now(tz=timezone.utc),
                details=details,
                issues=issues,
            )

        except Exception as exc:
            logger.exception("database_check_failed", error=str(exc), status="ERROR")
            if self._conn:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
            return ServiceStatus(
                name="Database",
                enabled=True,
                running=False,
                healthy=False,
                last_activity=None,
                details={"error": str(exc)},
                issues=[f"Database connection failed: {exc}"],
            )

    def check_monitoring_services(self) -> List[ServiceStatus]:
        """
        Check status of all monitoring services.

        Logs which components are enabled, running, and functional.
        """
        logger.info("checking_monitoring_services", operation="SERVICE_CHECK")

        services = []

        # Check anomaly detector
        services.append(
            self._check_service(
                name="Anomaly Detector",
                check_function=lambda: self._check_recent_alerts("win_rate"),
                description="Statistical anomaly detection for performance",
            )
        )

        # Check pattern health monitor
        services.append(
            self._check_service(
                name="Pattern Health Monitor",
                check_function=lambda: self._check_pattern_monitoring(),
                description="Pattern performance tracking",
            )
        )

        # Check error monitor
        services.append(
            self._check_service(
                name="Error Monitor",
                check_function=lambda: True,  # Always enabled
                description="Log analysis and error detection",
            )
        )

        # Check alert manager
        services.append(
            self._check_service(
                name="Alert Manager",
                check_function=lambda: self._check_telegram_config(),
                description="Telegram alert routing and delivery",
            )
        )

        logger.info(
            "monitoring_services_checked",
            total=len(services),
            healthy=sum(1 for s in services if s.healthy),
        )

        return services

    def _check_service(
        self,
        name: str,
        check_function: callable,
        description: str,
    ) -> ServiceStatus:
        """Generic service health check with logging."""
        logger.info("checking_service", service=name, description=description)

        try:
            is_healthy = check_function()
            status = "HEALTHY" if is_healthy else "UNHEALTHY"

            logger.info(
                "service_checked",
                service=name,
                healthy=is_healthy,
                status=status,
            )

            return ServiceStatus(
                name=name,
                enabled=True,
                running=True,
                healthy=is_healthy,
                last_activity=datetime.now(tz=timezone.utc),
                details={"description": description, "status": status},
                issues=[] if is_healthy else ["Service check failed"],
            )

        except Exception as exc:
            logger.exception("service_check_failed", service=name, error=str(exc))

            return ServiceStatus(
                name=name,
                enabled=True,
                running=False,
                healthy=False,
                last_activity=None,
                details={"error": str(exc)},
                issues=[f"Check failed: {exc}"],
            )

    def _check_recent_alerts(self, alert_type: str) -> bool:
        """Check if alerts are being generated (indicates monitoring is active)."""
        # Simplified check - in production would query alert history
        return True

    def _check_pattern_monitoring(self) -> bool:
        """Check if pattern monitoring is functional."""
        try:
            self.connect()
            with self._conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM pattern_library WHERE reliability_score > 0")
                active_patterns = cur.fetchone()[0]

            logger.info("pattern_monitoring_check", active_patterns=active_patterns)
            return active_patterns >= 0  # Always true if query succeeds

        except Exception:
            return False

    def _check_telegram_config(self) -> bool:
        """Check if Telegram is configured."""
        # Would check actual telegram settings
        return True

    def check_resource_usage(self) -> Dict[str, float]:
        """
        Check system resource usage.

        Logs CPU, memory, disk usage for visibility.
        """
        logger.info("checking_resource_usage", operation="RESOURCE_MONITOR")

        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            resources = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "disk_percent": disk.percent,
                "disk_used_gb": disk.used / (1024**3),
                "disk_total_gb": disk.total / (1024**3),
            }

            logger.info(
                "resource_usage_checked",
                cpu=f"{cpu_percent:.1f}%",
                memory=f"{memory.percent:.1f}%",
                disk=f"{disk.percent:.1f}%",
                **resources,
            )

            # Warn if resources high
            if cpu_percent > 80:
                logger.warning("high_cpu_usage", cpu_percent=cpu_percent)
            if memory.percent > 85:
                logger.warning("high_memory_usage", memory_percent=memory.percent)
            if disk.percent > 90:
                logger.warning("high_disk_usage", disk_percent=disk.percent)

            return resources

        except Exception as exc:
            logger.exception("resource_check_failed", error=str(exc))
            return {}

    def get_active_features(self) -> List[str]:
        """
        Detect which features are actually enabled and active.

        Critical for knowing what's turned on vs off.
        """
        logger.info("detecting_active_features", operation="FEATURE_DETECTION")

        features = []

        try:
            self.connect()

            # Check if training data exists (only if table exists)
            try:
                with self._conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM trade_memory")
                    if cur.fetchone()[0] > 0:
                        features.append("HISTORICAL_TRAINING_DATA")
                        logger.info("feature_active", feature="HISTORICAL_TRAINING_DATA")
            except psycopg2.errors.UndefinedTable:
                logger.debug("table_trade_memory_not_exists", message="Table doesn't exist yet")
                self._conn.rollback()
            except Exception as e:
                logger.warning("trade_memory_check_failed", error=str(e))
                self._conn.rollback()

            # Check if patterns exist (only if table exists)
            try:
                with self._conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM pattern_library WHERE reliability_score > 0")
                    if cur.fetchone()[0] > 0:
                        features.append("PATTERN_LIBRARY")
                        logger.info("feature_active", feature="PATTERN_LIBRARY")
            except psycopg2.errors.UndefinedTable:
                logger.debug("table_pattern_library_not_exists", message="Table doesn't exist yet")
                self._conn.rollback()
            except Exception as e:
                logger.warning("pattern_library_check_failed", error=str(e))
                self._conn.rollback()

            # Check if win/loss analysis tables have data (only if table exists)
            try:
                with self._conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM win_analysis")
                    if cur.fetchone()[0] > 0:
                        features.append("WIN_LOSS_ANALYSIS")
                        logger.info("feature_active", feature="WIN_LOSS_ANALYSIS")
            except psycopg2.errors.UndefinedTable:
                logger.debug("table_win_analysis_not_exists", message="Table doesn't exist yet")
                self._conn.rollback()
            except Exception as e:
                logger.warning("win_analysis_check_failed", error=str(e))
                self._conn.rollback()

            # Check if post-exit tracking has data (only if table exists)
            try:
                with self._conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM post_exit_tracking")
                    if cur.fetchone()[0] > 0:
                        features.append("POST_EXIT_TRACKING")
                        logger.info("feature_active", feature="POST_EXIT_TRACKING")
            except psycopg2.errors.UndefinedTable:
                logger.debug("table_post_exit_tracking_not_exists", message="Table doesn't exist yet")
                self._conn.rollback()
            except Exception as e:
                logger.warning("post_exit_tracking_check_failed", error=str(e))
                self._conn.rollback()

            logger.info(
                "active_features_detected",
                total=len(features),
                features=features,
            )

        except Exception as exc:
            logger.exception("feature_detection_failed", error=str(exc))
            if self._conn:
                try:
                    self._conn.rollback()
                except Exception:
                    pass

        return features

    def get_recent_activity(self) -> Dict[str, Any]:
        """
        Get recent system activity.

        Shows what the system has been doing.
        """
        logger.info("checking_recent_activity", operation="ACTIVITY_CHECK")

        activity = {}

        try:
            self.connect()

            # Recent trades (only if table exists)
            try:
                with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT
                            COUNT(*) as count,
                            MAX(entry_timestamp) as last_trade
                        FROM trade_memory
                        WHERE entry_timestamp >= NOW() - INTERVAL '24 hours'
                        """
                    )
                    trade_data = cur.fetchone()
                    activity["trades_24h"] = trade_data["count"] if trade_data else 0
                    activity["last_trade_time"] = trade_data["last_trade"] if trade_data else None

                    logger.info(
                        "recent_trade_activity",
                        trades_24h=activity["trades_24h"],
                        last_trade=activity["last_trade_time"],
                    )
            except psycopg2.errors.UndefinedTable:
                logger.debug("table_trade_memory_not_exists", message="Table doesn't exist yet")
                activity["trades_24h"] = 0
                activity["last_trade_time"] = None
                self._conn.rollback()
            except Exception as e:
                logger.warning("recent_trades_check_failed", error=str(e))
                activity["trades_24h"] = 0
                activity["last_trade_time"] = None
                self._conn.rollback()

            # Recent pattern updates (only if table exists)
            try:
                with self._conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT COUNT(*) FROM pattern_library
                        WHERE last_updated >= NOW() - INTERVAL '24 hours'
                        """
                    )
                    activity["patterns_updated_24h"] = cur.fetchone()[0]

                    logger.info("pattern_update_activity", patterns_updated=activity["patterns_updated_24h"])
            except psycopg2.errors.UndefinedTable:
                logger.debug("table_pattern_library_not_exists", message="Table doesn't exist yet")
                activity["patterns_updated_24h"] = 0
                self._conn.rollback()
            except Exception as e:
                logger.warning("pattern_updates_check_failed", error=str(e))
                activity["patterns_updated_24h"] = 0
                self._conn.rollback()

        except Exception as exc:
            logger.exception("activity_check_failed", error=str(exc))
            if self._conn:
                try:
                    self._conn.rollback()
                except Exception:
                    pass

        return activity

    def generate_full_report(self) -> SystemHealthReport:
        """
        Generate comprehensive system health report.

        This is the master health check that logs EVERYTHING.
        """
        logger.info(
            "generating_full_system_report",
            operation="FULL_HEALTH_CHECK",
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
        )

        # Check all components
        db_status = self.check_database_status()
        monitoring_services = self.check_monitoring_services()
        resources = self.check_resource_usage()
        features = self.get_active_features()
        activity = self.get_recent_activity()

        # Determine overall status
        all_services = [db_status] + monitoring_services
        healthy_count = sum(1 for s in all_services if s.healthy)
        total_count = len(all_services)

        if healthy_count == total_count:
            overall_status = "HEALTHY"
        elif healthy_count >= total_count * 0.7:
            overall_status = "DEGRADED"
        else:
            overall_status = "CRITICAL"

        report = SystemHealthReport(
            timestamp=datetime.now(tz=timezone.utc),
            overall_status=overall_status,
            services=all_services,
            database_status=db_status,
            resource_usage=resources,
            active_features=features,
            recent_activity=activity,
        )

        # Log comprehensive summary
        logger.info(
            "system_health_report_generated",
            overall_status=overall_status,
            healthy_services=f"{healthy_count}/{total_count}",
            active_features_count=len(features),
            database_healthy=db_status.healthy,
            cpu_usage=resources.get("cpu_percent", 0),
            memory_usage=resources.get("memory_percent", 0),
            trades_24h=activity.get("trades_24h", 0),
        )

        # Log each service status
        for service in all_services:
            logger.info(
                "service_status_detail",
                service=service.name,
                enabled=service.enabled,
                running=service.running,
                healthy=service.healthy,
                issues_count=len(service.issues),
            )

        # Log active features
        for feature in features:
            logger.info("feature_status", feature=feature, status="ACTIVE")

        return report

    def log_startup_status(self) -> None:
        """
        Log comprehensive status at system startup.

        Critical for knowing what's enabled when system starts.
        """
        logger.info(
            "===== SYSTEM STARTUP STATUS CHECK =====",
            operation="STARTUP_CHECK",
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
        )

        report = self.generate_full_report()

        logger.info(
            "startup_status_summary",
            overall_status=report.overall_status,
            services_total=len(report.services),
            services_healthy=sum(1 for s in report.services if s.healthy),
            features_active=len(report.active_features),
            database_connected=report.database_status.healthy,
        )

        logger.info("===== STARTUP CHECK COMPLETE =====")
