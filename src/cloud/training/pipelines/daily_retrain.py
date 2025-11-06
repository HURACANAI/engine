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


def configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(structlog.INFO),
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
    if not settings.postgres:
        raise RuntimeError("Postgres DSN must be configured before running the engine")
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
        logger.info(
            "startup_status_summary",
            overall_status=startup_status.get("overall_status", "UNKNOWN"),
            services_healthy=startup_status.get("services_healthy", 0),
            services_total=startup_status.get("services_total", 0),
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
    )

    try:
        # Run pre-training health check
        if health_monitor:
            logger.info("===== PRE-TRAINING HEALTH CHECK =====")
            pre_alerts = health_monitor.run_health_check()
            logger.info(
                "pre_training_health_check_complete",
                alerts=len(pre_alerts),
                critical=sum(1 for a in pre_alerts if a.severity.value == "CRITICAL"),
                warning=sum(1 for a in pre_alerts if a.severity.value == "WARNING"),
            )

        # Run main training
        orchestrator.run()
        logger.info("run_complete")

        # Run post-training health check
        if health_monitor:
            logger.info("===== POST-TRAINING HEALTH CHECK =====")
            post_alerts = health_monitor.run_health_check()
            logger.info(
                "post_training_health_check_complete",
                alerts=len(post_alerts),
                critical=sum(1 for a in post_alerts if a.severity.value == "CRITICAL"),
                warning=sum(1 for a in post_alerts if a.severity.value == "WARNING"),
            )

    except Exception as exc:  # noqa: BLE001
        logger.exception("run_failed", error=str(exc))

        # Run emergency health check on failure
        if health_monitor:
            logger.info("===== EMERGENCY HEALTH CHECK (FAILURE) =====")
            try:
                emergency_alerts = health_monitor.run_health_check()
                logger.info(
                    "emergency_health_check_complete",
                    alerts=len(emergency_alerts),
                )
            except Exception as health_exc:  # noqa: BLE001
                logger.exception("emergency_health_check_failed", error=str(health_exc))

        raise
    finally:
        if ray.is_initialized():
            ray.shutdown()

