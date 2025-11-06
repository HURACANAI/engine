"""
ENGINE MAIN ENTRY POINT - Daily Baseline Training

This is the main entry point for the Engine (Cloud Training Box).
Runs daily at 02:00 UTC to build a new baseline model.

What the Engine Does:
- Trains on 3-6 months of historical market data
- Builds a clean daily baseline model
- Validates with walk-forward testing
- Saves model to S3 and registers in Postgres
- Generates performance reports

What the Engine Does NOT Do:
- Hourly incremental updates (that's Mechanic)
- Live trading execution (that's Pilot)
- Real-time order management (that's Pilot)

Entry Point:
    python -m cloud.training.pipelines.daily_retrain
    OR
    cloud-training-daily-retrain (poetry script)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import ray
import structlog

from ..config.settings import EngineSettings
from ..services.exchange import ExchangeClient
from ..services.model_registry import ModelRegistry, RegistryConfig
from ..services.notifications import NotificationClient
from ..services.orchestration import TrainingOrchestrator
from ..services.universe import UniverseSelector
from ..services.artifacts import ArtifactPublisher
from ..datasets.data_loader import MarketMetadataLoader
from ..monitoring.health_monitor import HealthMonitorOrchestrator
from ..monitoring.system_status import SystemStatusReporter
from ..monitoring.comprehensive_telegram_monitor import ComprehensiveTelegramMonitor, NotificationLevel
from pathlib import Path


def configure_logging() -> None:
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


def initialize_ray(address: Optional[str], namespace: str, runtime_env: Optional[dict]) -> None:
    if ray.is_initialized():
        return
    init_kwargs = {
        "namespace": namespace,
        "runtime_env": runtime_env or {},
        "ignore_reinit_error": True,
        "log_to_driver": False,
    }
    if address:
        init_kwargs["address"] = address
    ray.init(**init_kwargs)


def run_daily_retrain() -> None:
    """
    Execute the full Engine retraining workflow.
    
    This is the main function for Engine daily baseline training.
    It:
    1. Loads configuration and initializes services
    2. Selects trading universe
    3. Trains models on historical data
    4. Validates with walk-forward testing
    5. Saves models and metrics to S3/Postgres
    6. Generates reports
    
    Runs daily at 02:00 UTC (configured via APScheduler).
    """

    configure_logging()
    logger = structlog.get_logger("daily_retrain")
    start_ts = datetime.now(tz=timezone.utc)
    logger.info("run_start", timestamp=start_ts.isoformat())

    settings = EngineSettings.load()
    
    # Initialize comprehensive Telegram monitoring
    telegram_monitor = None
    log_file = Path("logs") / f"engine_monitoring_{start_ts.strftime('%Y%m%d_%H%M%S')}.log"
    
    if settings.notifications.telegram_enabled and settings.notifications.telegram_chat_id:
        # Get bot token from environment or use default
        bot_token = settings.notifications.telegram_webhook_url or "8229109041:AAFIcLRx3V50khoaEIG7WXeI1ITzy4s6hf0"
        # Extract token from webhook URL if needed
        if "bot" in bot_token and "/" in bot_token:
            bot_token = bot_token.split("/bot")[-1].split("/")[0]
        
        telegram_monitor = ComprehensiveTelegramMonitor(
            bot_token=bot_token,
            chat_id=settings.notifications.telegram_chat_id,
            log_file=log_file,
            enable_trade_notifications=True,
            enable_learning_notifications=True,
            enable_error_notifications=True,
            enable_performance_summaries=True,
            enable_model_updates=True,
            enable_gate_decisions=True,
            enable_health_alerts=True,
            enable_validation_alerts=True,
            min_notification_level=NotificationLevel.LOW,
        )
        logger.info("telegram_monitoring_initialized", log_file=str(log_file))
    else:
        logger.warning("telegram_monitoring_disabled", reason="not configured")

    initialize_ray(settings.ray.address, settings.ray.namespace, settings.ray.runtime_env)

    exchange_settings = settings.exchange
    credentials = {}
    if exchange_settings.credentials.get(exchange_settings.primary):
        cred = exchange_settings.credentials[exchange_settings.primary]
        credentials = {
            "api_key": cred.api_key,
            "api_secret": cred.api_secret,
            "api_passphrase": cred.api_passphrase,
        }

    exchange_client = ExchangeClient(
        exchange_settings.primary,
        credentials=credentials,
        sandbox=exchange_settings.sandbox,
    )
    metadata_loader = MarketMetadataLoader(exchange_client)
    universe_selector = UniverseSelector(exchange_client, metadata_loader, settings.universe)
    
    # Get selected symbols for Telegram notification
    selected_symbols_df = universe_selector.select()
    selected_symbols = selected_symbols_df.to_dicts() if hasattr(selected_symbols_df, 'to_dicts') else selected_symbols_df
    if telegram_monitor:
        telegram_monitor.notify_system_startup(
            symbols=[s['symbol'] for s in selected_symbols],
            total_coins=len(selected_symbols),
        )
    
    if not settings.postgres:
        error_msg = "Postgres DSN must be configured before running the engine"
        if telegram_monitor:
            telegram_monitor.notify_error("Configuration Error", error_msg)
        raise RuntimeError(error_msg)
    model_registry = ModelRegistry(RegistryConfig(dsn=settings.postgres.dsn))

    notifier = NotificationClient(settings.notifications)
    artifact_publisher = ArtifactPublisher(settings.artifacts)

    # Initialize health monitoring if enabled
    health_monitor = None
    if settings.training.monitoring.enabled:
        logger.info(
            "===== INITIALIZING HEALTH MONITORING =====",
            check_interval=settings.training.monitoring.check_interval_seconds,
        )
        health_monitor = HealthMonitorOrchestrator(settings=settings, dsn=settings.postgres.dsn)

        # Run initial system status check
        logger.info("===== STARTUP STATUS CHECK =====", operation="STARTUP_HEALTH_CHECK")
        status_reporter = SystemStatusReporter(dsn=settings.postgres.dsn)
        startup_status = status_reporter.generate_full_report()
        healthy_count = sum(1 for s in startup_status.services if s.healthy)
        total_count = len(startup_status.services)
        logger.info(
            "startup_status_summary",
            overall_status=startup_status.overall_status,
            services_healthy=healthy_count,
            services_total=total_count,
        )
    else:
        logger.info("health_monitoring_disabled", reason="monitoring.enabled=false")

    orchestrator = TrainingOrchestrator(
        settings=settings,
        exchange_client=exchange_client,
        universe_selector=universe_selector,
        model_registry=model_registry,
        notifier=notifier,
        artifact_publisher=artifact_publisher,
        telegram_monitor=telegram_monitor,  # Pass Telegram monitor for validation notifications
    )

    try:
        # Run pre-training health check
        if health_monitor:
            logger.info("===== PRE-TRAINING HEALTH CHECK =====")
            pre_alerts = health_monitor.run_health_check()
            critical_alerts = [a for a in pre_alerts if a.severity.value == "CRITICAL"]
            warnings = [a for a in pre_alerts if a.severity.value == "WARNING"]
            
            logger.info(
                "pre_training_health_check_complete",
                alerts=len(pre_alerts),
                critical=len(critical_alerts),
                warning=len(warnings),
            )
            
            # Notify Telegram about health check
            if telegram_monitor:
                status_reporter = SystemStatusReporter(dsn=settings.postgres.dsn)
                status_report = status_reporter.generate_full_report()
                telegram_monitor.notify_health_check(
                    status=status_report.get("overall_status", "UNKNOWN"),
                    alerts=[{"message": a.message, "severity": a.severity.value} for a in critical_alerts],
                    warnings=[{"message": a.message, "severity": a.severity.value} for a in warnings],
                    healthy_services=status_report.get("services_healthy", 0),
                    total_services=status_report.get("services_total", 0),
                )

        # Run main training
        results = orchestrator.run()
        logger.info("run_complete", total_results=len(results))
        
        # Calculate summary stats for Telegram
        if telegram_monitor and results:
            total_trades = sum(r.metrics.get("trades_oos", 0) for r in results)
            total_profit = sum(r.metrics.get("pnl_bps", 0) * 0.01 for r in results)  # Approximate
            published = sum(1 for r in results if r.published)
            rejected = len(results) - published
            
            # Notify about completion
            end_ts = datetime.now(tz=timezone.utc)
            duration_minutes = int((end_ts - start_ts).total_seconds() / 60)
            
            telegram_monitor.notify_system_shutdown(
                total_trades=total_trades,
                total_profit_gbp=total_profit,
                duration_minutes=duration_minutes,
            )

        # Run post-training health check
        if health_monitor:
            logger.info("===== POST-TRAINING HEALTH CHECK =====")
            post_alerts = health_monitor.run_health_check()
            critical_alerts = [a for a in post_alerts if a.severity.value == "CRITICAL"]
            warnings = [a for a in post_alerts if a.severity.value == "WARNING"]
            
            logger.info(
                "post_training_health_check_complete",
                alerts=len(post_alerts),
                critical=len(critical_alerts),
                warning=len(warnings),
            )
            
            # Notify Telegram about health check
            if telegram_monitor:
                status_reporter = SystemStatusReporter(dsn=settings.postgres.dsn)
                status_report = status_reporter.generate_full_report()
                telegram_monitor.notify_health_check(
                    status=status_report.get("overall_status", "UNKNOWN"),
                    alerts=[{"message": a.message, "severity": a.severity.value} for a in critical_alerts],
                    warnings=[{"message": a.message, "severity": a.severity.value} for a in warnings],
                    healthy_services=status_report.get("services_healthy", 0),
                    total_services=status_report.get("services_total", 0),
                )

    except Exception as exc:  # noqa: BLE001
        logger.exception("run_failed", error=str(exc))
        
        # Notify Telegram about error
        if telegram_monitor:
            telegram_monitor.notify_error(
                error_type=type(exc).__name__,
                error_message=str(exc),
                context={"timestamp": datetime.now(tz=timezone.utc).isoformat()},
            )

        # Run emergency health check on failure
        if health_monitor:
            logger.info("===== EMERGENCY HEALTH CHECK (FAILURE) =====")
            try:
                emergency_alerts = health_monitor.run_health_check()
                critical_alerts = [a for a in emergency_alerts if a.severity.value == "CRITICAL"]
                warnings = [a for a in emergency_alerts if a.severity.value == "WARNING"]
                
                logger.info(
                    "emergency_health_check_complete",
                    alerts=len(emergency_alerts),
                )
                
                # Notify Telegram about emergency health check
                if telegram_monitor:
                    status_reporter = SystemStatusReporter(dsn=settings.postgres.dsn)
                    status_report = status_reporter.generate_full_report()
                    telegram_monitor.notify_health_check(
                        status="CRITICAL",
                        alerts=[{"message": a.message, "severity": a.severity.value} for a in critical_alerts],
                        warnings=[{"message": a.message, "severity": a.severity.value} for a in warnings],
                        healthy_services=status_report.get("services_healthy", 0),
                        total_services=status_report.get("services_total", 0),
                    )
            except Exception as health_exc:  # noqa: BLE001
                logger.exception("emergency_health_check_failed", error=str(health_exc))

        raise
    finally:
        # Export log file path
        if telegram_monitor and log_file.exists():
            logger.info("monitoring_log_available", log_file=str(log_file))
            print(f"\n📝 Monitoring log saved to: {log_file}")
            print(f"   You can copy/paste this file to share with support.\n")
        
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    run_daily_retrain()

