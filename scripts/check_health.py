#!/usr/bin/env python3
"""
Quick Health Check Script

Run this to manually check system health with enhanced comprehensive checks.
Shows all 18 health checks with detailed status.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import structlog
from src.cloud.training.config.settings import EngineSettings
from src.cloud.training.monitoring.health_monitor import HealthMonitorOrchestrator
from src.cloud.training.integrations.dropbox_sync import DropboxSync

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(10),  # INFO level
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)

logger = structlog.get_logger(__name__)


def main():
    """Run comprehensive health check."""
    print("=" * 60)
    print("🏥 HURACAN ENGINE - COMPREHENSIVE HEALTH CHECK")
    print("=" * 60)
    print()

    # Load settings
    try:
        settings = EngineSettings.load()
        logger.info("settings_loaded")
    except Exception as e:
        print(f"❌ Failed to load settings: {e}")
        sys.exit(1)

    # Initialize Dropbox sync (if enabled)
    dropbox_sync = None
    if settings.dropbox and settings.dropbox.enabled:
        try:
            dropbox_sync = DropboxSync(
                access_token=settings.dropbox.access_token,
                app_folder=settings.dropbox.app_folder,
            )
            logger.info("dropbox_initialized")
        except Exception as e:
            logger.warning("dropbox_init_failed", error=str(e))

    # Initialize health monitor
    dsn = settings.postgres.dsn if settings.postgres else None
    if not dsn:
        print("⚠️  Warning: No database DSN configured")
        print("   Health check will run but database checks will fail")
        dsn = ""

    try:
        health_monitor = HealthMonitorOrchestrator(
            settings=settings,
            dsn=dsn,
            dropbox_sync=dropbox_sync,
        )
        logger.info("health_monitor_initialized")
    except Exception as e:
        print(f"❌ Failed to initialize health monitor: {e}")
        sys.exit(1)

    # Run health check
    print("🔍 Running comprehensive health check...")
    print()

    try:
        alerts = health_monitor.run_health_check()
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        logger.exception("health_check_failed", error=str(e))
        sys.exit(1)

    # Get enhanced health report
    enhanced_report = health_monitor.get_enhanced_health_report()

    # Print results
    print("=" * 60)
    print("📊 HEALTH CHECK RESULTS")
    print("=" * 60)
    print()

    if enhanced_report:
        print(f"Overall Status: {enhanced_report.overall_status}")
        print(f"Total Checks: {len(enhanced_report.checks)}")
        print(f"Timestamp: {enhanced_report.timestamp}")
        print()

        # Summary
        summary = enhanced_report.summary
        print("📈 SUMMARY:")
        print(f"  ✅ Healthy: {summary.get('healthy', 0)}")
        print(f"  ⚠️  Warnings: {summary.get('warnings', 0)}")
        print(f"  🚨 Critical: {summary.get('critical', 0)}")
        print(f"  ⏸️  Disabled: {summary.get('disabled', 0)}")
        print()

        # Resource usage
        if enhanced_report.resource_usage:
            resources = enhanced_report.resource_usage
            print("💻 RESOURCE USAGE:")
            print(f"  CPU: {resources.get('cpu_percent', 0):.1f}%")
            print(f"  Memory: {resources.get('memory_percent', 0):.1f}%")
            print(f"  Disk: {resources.get('disk_percent', 0):.1f}%")
            print()

        # Detailed checks
        print("🔍 DETAILED CHECKS:")
        print()

        for check in enhanced_report.checks:
            status_icon = {
                "HEALTHY": "✅",
                "WARNING": "⚠️",
                "CRITICAL": "🚨",
                "DISABLED": "⏸️",
            }.get(check.status, "❓")

            print(f"{status_icon} {check.name}: {check.status}")
            print(f"   {check.message}")

            if check.issues:
                print("   Issues:")
                for issue in check.issues:
                    print(f"     • {issue}")

            if check.recommendations:
                print("   Recommendations:")
                for rec in check.recommendations:
                    print(f"     • {rec}")

            print()

        # Critical issues
        if enhanced_report.critical_issues:
            print("🚨 CRITICAL ISSUES:")
            for issue in enhanced_report.critical_issues:
                print(f"  • {issue}")
            print()

        # Warnings
        if enhanced_report.warnings:
            print("⚠️  WARNINGS:")
            for warning in enhanced_report.warnings:
                print(f"  • {warning}")
            print()

        # Recommendations
        if enhanced_report.recommendations:
            print("💡 RECOMMENDATIONS:")
            for rec in enhanced_report.recommendations:
                print(f"  • {rec}")
            print()

    else:
        print("⚠️  Enhanced health report not available")
        print("   Falling back to basic health check results")
        print()

    # Alerts summary
    print("📢 ALERTS:")
    print(f"  Total: {len(alerts)}")
    print(f"  Critical: {sum(1 for a in alerts if a.severity.value == 'CRITICAL')}")
    print(f"  Warning: {sum(1 for a in alerts if a.severity.value == 'WARNING')}")
    print()

    if alerts:
        print("Alert Details:")
        for alert in alerts:
            severity_icon = {
                "CRITICAL": "🚨",
                "WARNING": "⚠️",
            }.get(alert.severity.value, "ℹ️")
            print(f"  {severity_icon} {alert.severity.value}: {alert.title}")
            print(f"     {alert.message}")
            print()

    print("=" * 60)
    print("✅ Health check complete!")
    print("=" * 60)

    # Exit with appropriate code
    if enhanced_report and enhanced_report.overall_status == "CRITICAL":
        sys.exit(2)
    elif enhanced_report and enhanced_report.overall_status == "DEGRADED":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

