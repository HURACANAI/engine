#!/usr/bin/env python3
"""
Standalone health monitoring script.

Runs continuous health monitoring with comprehensive logging.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cloud.training.config.settings import EngineSettings
from cloud.training.monitoring.health_monitor import HealthMonitorOrchestrator
import structlog


def configure_logging():
    """Configure structured logging."""
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )


def main():
    """Run health monitoring."""
    # Configure logging
    configure_logging()
    logger = structlog.get_logger(__name__)

    logger.info(
        "===== STARTING HEALTH MONITORING SYSTEM =====",
        script="run_health_monitor.py",
    )

    # Load settings
    settings = EngineSettings.load()

    # Get database DSN
    if not settings.postgres or not settings.postgres.dsn:
        logger.error("postgres_dsn_not_configured")
        print("ERROR: PostgreSQL DSN not configured in settings")
        print("Set POSTGRES_DSN environment variable or add to config")
        sys.exit(1)

    dsn = settings.postgres.dsn
    logger.info("settings_loaded", postgres_configured=True)

    # Initialize monitor
    monitor = HealthMonitorOrchestrator(settings=settings, dsn=dsn)

    # Run single check first
    logger.info("running_initial_health_check")
    print("\n🔍 Running initial health check...\n")

    alerts = monitor.run_health_check()

    print(f"✅ Health check complete!")
    print(f"   Alerts generated: {len(alerts)}")
    print(f"   Critical: {sum(1 for a in alerts if a.severity.value == 'CRITICAL')}")
    print(f"   Warning: {sum(1 for a in alerts if a.severity.value == 'WARNING')}")

    # Ask to continue
    print("\n🔄 Start continuous monitoring? (Ctrl+C to stop)")
    print("   Checks will run every 5 minutes")

    try:
        input("Press Enter to start, or Ctrl+C to exit...")
    except KeyboardInterrupt:
        print("\n👋 Monitoring cancelled")
        sys.exit(0)

    # Run continuous monitoring
    print("\n🚀 Starting continuous monitoring...\n")
    print("   Logs: Check structlog output")
    print("   Alerts: Check Telegram (if configured)")
    print("   Stop: Press Ctrl+C\n")

    try:
        monitor.run_continuous(interval_seconds=300)
    except KeyboardInterrupt:
        print("\n\n👋 Stopping health monitor...")
        monitor.stop()
        print("✅ Health monitor stopped gracefully")


if __name__ == "__main__":
    main()
