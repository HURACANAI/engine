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
import os
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
from ..monitoring.learning_tracker import LearningTracker, LearningCategory
from ..integrations.dropbox_sync import DropboxSync
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
        "log_to_driver": True,  # Enable logging from Ray tasks to see what's happening
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
    
    # ===== CREATE DROPBOX DATED FOLDER IMMEDIATELY (FIRST ACTION) =====
    # This happens before anything else so user knows it works
    dropbox_sync = None
    # Get Dropbox token - clean it (remove whitespace, newlines, etc.)
    # Token can come from environment variable DROPBOX_ACCESS_TOKEN or settings
    env_token = os.getenv("DROPBOX_ACCESS_TOKEN")
    settings_token = settings.dropbox.access_token
    hardcoded_fallback = "sl.u.AGGHS11iAzSe4qKe6wzea4pxU6ilDfckS3z-huUVsW9kgytgHB0pEVSub8_6H8ypIsukAGm0ai7RBl5PWTEvd45uI7UybBsG_E_KwboZGwYErU0gzXFComp-uWt24OkyJsf1D5sYja7gtEo_FS1AFlLOsqiUjuZFQNEANjO8f-ShUaUj_by9sN6KMbhpAynzbpztLkYi21Ppr89Xdd27bBRzM7WLZZ7sqBy9mBCep0jav21WqGfJgu9qZpw01nQPWc23Q_c96lgDiIWcu7z5VEhDstNtP0jMRKhzj9vzC7Yx2-VLxye_SkxDEvS4h--20cgosUe3znyRy-c2BC_kVj7gnL8xPcfJnuJl528aYOjEsHrD662PnD7tQzT8sMef90RGWVXbp842BZ_2WcMmbjdCz7HpDZ-EzCB9_6GWBIpJMxYEOAd6rqAFhP9glSBG-7W2hSw3mpwRUVRxCvhKU5IAnWe3Jsu8OGli7RXQ-yMUxwORu7--Y5PKe2_6bRh1y_hv6mtCawiYq1F-RmTseIbApBppI4H1o04YxFFFcnR7nYilMDV_-rnoktRpAusFCzJe6ol9JyEuFugTCwuJFU48eUQ5-i_7EfNT26IG-4WrY8Bfah11Sll5crlD7iCS96aFDUGZzg11a25oJ6CelMtEguSv6X6lcH901_IXkKMpdE0NpQtiOlJKyOqFIEAUb8MJjWuiRIJaPG_YM9bQYFNKFfC7hMjn0YSZpZ6rm-L49zigtr2KGQEIeN9HwLKHb596NbSwBJ_cRT5N1JhJcwVefCTXfSpUhLGmEngzAw7UXE4ZQoHHn46vzWLDFglfwervcLqAzzyX9pTl0ciO7kvmKrehkdKNDHbX2dBvZ5Asn0HDTOgUDnoXoLXrhNeycOVSMj92MPlK_UG2Mo3W4k4PU5YLMUuSIahedFnWxKJiAdPlnmbTHiSSLn_ToVpZMgMgP81gFytDFXVKQzSIylrUHRsugOZLApVF0TehYm4ED_7IOGNEnjUa3ZkbVvyiFbnozX-wC9sS3OA8b19H2pENy2K-oejmG4VSVBjS9Xk4GVy4FICOpRunZ0mNt2xBAlBS6M3TW-LSSu7pLexp7XB3VYhuoQ2M1lJ8vlLKhp0K3-TVP9neLKrKpDTbj8TJiYTec1PVEH_7hVOCx09VGvF47xus8kEs8ZNZcxBz4-0ra5SueynSVbKN5xmT2TDXOofeafHZix0k57ics1fw_ZYd3Ig7075PeHliZtYzRSUDCtuMAIJhEaYEivi-cIFdvPg"
    
    # Determine token source for debugging
    if env_token:
        token_source = "environment_variable"
        dropbox_token_raw = env_token
        logger.info(
            "dropbox_token_source",
            source=token_source,
            token_prefix=env_token[:30] if len(env_token) > 30 else env_token,
            token_length=len(env_token),
        )
    elif settings_token and settings_token != hardcoded_fallback:
        token_source = "settings_file"
        dropbox_token_raw = settings_token
        logger.info(
            "dropbox_token_source",
            source=token_source,
            token_prefix=settings_token[:30] if len(settings_token) > 30 else settings_token,
            token_length=len(settings_token),
        )
    else:
        token_source = "hardcoded_fallback"
        dropbox_token_raw = hardcoded_fallback
        logger.warning(
            "dropbox_token_source",
            source=token_source,
            token_prefix=hardcoded_fallback[:30],
            token_length=len(hardcoded_fallback),
            message="Using hardcoded fallback token - consider setting DROPBOX_ACCESS_TOKEN environment variable",
        )
    
    # Clean token (remove any whitespace, newlines, quotes, etc.)
    dropbox_token = dropbox_token_raw.strip().strip('"').strip("'").strip()
    dropbox_folder = settings.dropbox.app_folder or "Runpodhuracan"
    
    if settings.dropbox.enabled and dropbox_token:
        try:
            logger.info("dropbox_creating_dated_folder_immediately")
            print("\n📁 Creating Dropbox dated folder...")
            
            # Initialize Dropbox sync - this will create dated folder as FIRST action
            dropbox_sync = DropboxSync(
                access_token=dropbox_token,
                app_folder=dropbox_folder,
                enabled=True,
                create_dated_folder=True,  # Create dated folder immediately
            )
            
            # Log the dated folder that was created
            if hasattr(dropbox_sync, "_dated_folder") and dropbox_sync._dated_folder:
                logger.info(
                    "dropbox_dated_folder_created_successfully",
                    folder=dropbox_sync._dated_folder,
                    message="✅ Dated folder created - ready for data sync",
                )
                print(f"✅ Dropbox folder created: {dropbox_sync._dated_folder}\n")
            else:
                logger.warning("dropbox_dated_folder_creation_failed", message="Folder not created")
                print("⚠️  Dropbox folder creation failed - check logs\n")
        except Exception as sync_error:
            # Dropbox errors are non-fatal - continue without Dropbox sync
            error_msg = str(sync_error)
            
            # Check if it's an expired token error
            if "expired" in error_msg.lower() or "expired_access_token" in error_msg:
                logger.warning(
                    "dropbox_token_expired_non_fatal",
                    error=error_msg,
                    message="Dropbox token expired - continuing without Dropbox sync",
                    help_message=(
                        "To fix: Generate a new token at https://www.dropbox.com/developers/apps "
                        "and update DROPBOX_ACCESS_TOKEN environment variable"
                    ),
                )
                print(f"⚠️  Dropbox token expired (non-fatal): {error_msg}\n")
                print("   💡 To fix: Generate a new token at https://www.dropbox.com/developers/apps\n")
                print("   💡 Then update DROPBOX_ACCESS_TOKEN environment variable\n")
                print("   Engine will continue without Dropbox sync\n")
            else:
                logger.warning(
                    "dropbox_folder_creation_failed_non_fatal",
                    error=error_msg,
                    message="Continuing without Dropbox sync - engine will still run",
                )
                print(f"⚠️  Dropbox folder creation failed (non-fatal): {error_msg}\n")
                print("   Engine will continue without Dropbox sync\n")
            dropbox_sync = None
    
    # Initialize comprehensive Telegram monitoring
    telegram_monitor = None
    log_file = Path("logs") / f"engine_monitoring_{start_ts.strftime('%Y%m%d_%H%M%S')}.log"
    
    # Initialize learning tracker
    learning_tracker = LearningTracker(output_dir=Path("logs/learning"))
    
    # ===== START DROPBOX CONTINUOUS SYNC (AFTER FOLDER CREATION) =====
    # Now that folder is created, start syncing data as it's generated
    if dropbox_sync:
        try:
            logger.info("dropbox_starting_continuous_sync")
            print("🔄 Starting Dropbox continuous sync with optimized intervals...\n")
            print(f"   📚 Learning data: every {settings.dropbox.sync_interval_learning_seconds // 60} min")
            print(f"   📝 Logs & monitoring: every {settings.dropbox.sync_interval_logs_seconds // 60} min")
            print(f"   🤖 Models: every {settings.dropbox.sync_interval_models_seconds // 60} min")
            print(f"   📊 Historical data: every {settings.dropbox.sync_interval_data_cache_seconds // 60} min\n")
            
            # Start continuous sync with configurable intervals for different data types
            sync_intervals = {
                "learning": settings.dropbox.sync_interval_learning_seconds,
                "logs": settings.dropbox.sync_interval_logs_seconds,
                "models": settings.dropbox.sync_interval_models_seconds,
                "data_cache": settings.dropbox.sync_interval_data_cache_seconds,
            }
            
            sync_threads = dropbox_sync.start_continuous_sync(
                logs_dir="logs",
                models_dir="models",
                learning_dir="logs/learning",
                monitoring_dir="logs",
                data_cache_dir="data/candles",  # Historical coin data cache
                sync_intervals=sync_intervals,
            )
            
            logger.info(
                "dropbox_continuous_sync_started",
                learning_interval=settings.dropbox.sync_interval_learning_seconds,
                logs_interval=settings.dropbox.sync_interval_logs_seconds,
                models_interval=settings.dropbox.sync_interval_models_seconds,
                data_cache_interval=settings.dropbox.sync_interval_data_cache_seconds,
                total_threads=len(sync_threads),
            )
            
            # Restore historical data cache from Dropbox (if enabled)
            # Try to restore from today's dated folder first, then fallback to previous days
            # NOTE: If this is first startup (no data in Dropbox), training will download data normally
            if settings.dropbox.restore_data_cache_on_startup:
                logger.info("attempting_data_cache_restore", message="Checking Dropbox for existing historical data...")
                # Try to restore from today's dated folder (where new data will be stored)
                restored_count = dropbox_sync.restore_data_cache(
                    data_cache_dir="data/candles",
                    remote_dir=None,  # Will use today's dated folder
                    use_latest_dated_folder=True,
                )
                if restored_count > 0:
                    logger.info(
                        "data_cache_restored",
                        files_restored=restored_count,
                        message="Restored historical data from Dropbox - will skip re-downloading",
                    )
                    print(f"📥 Restored {restored_count} historical data files from Dropbox\n")
                else:
                    logger.info(
                        "data_cache_restore_empty",
                        message="No historical data in Dropbox yet (first startup) - data will be downloaded during training"
                    )
                    print("📊 No historical data in Dropbox yet (first startup)\n")
                    print("   Data will be downloaded from exchange during training...\n")
            else:
                logger.info("data_cache_restore_disabled", reason="restore_data_cache_on_startup=false")
            
            # Initial sync of existing data (if any)
            # Everything goes into dated folder: /Runpodhuracan/YYYY-MM-DD/
            sync_results = {}
            
            # Sync logs
            if settings.dropbox.sync_logs and Path("logs").exists():
                sync_results["logs"] = dropbox_sync.upload_logs("logs")
            
            # Sync models (for Hamilton to use)
            if settings.dropbox.sync_models and Path("models").exists():
                sync_results["models"] = dropbox_sync.upload_models("models")
            
            # Sync monitoring data
            if settings.dropbox.sync_monitoring and Path("logs").exists():
                sync_results["monitoring"] = dropbox_sync.upload_monitoring_data("logs")
            
            # Sync learning data (everything the engine learned)
            if settings.dropbox.sync_learning and Path("logs/learning").exists():
                sync_results["learning"] = dropbox_sync.sync_directory(
                    local_dir="logs/learning",
                    remote_dir="/learning",
                    pattern="*.json",
                    recursive=True,
                )
            
            # Sync historical coin data (in dated folder)
            if settings.dropbox.sync_data_cache and Path("data/candles").exists():
                sync_results["data_cache"] = dropbox_sync.upload_data_cache(
                    data_cache_dir="data/candles",
                    use_dated_folder=True,  # Store in dated folder
                )
            
            # Sync reports/analytics (if they exist)
            if Path("reports").exists():
                sync_results["reports"] = dropbox_sync.upload_reports("reports")
            
            # Sync config files (if they exist)
            if Path("config").exists():
                sync_results["config"] = dropbox_sync.upload_configs("config")
            
            # ===== COMPREHENSIVE DATA EXPORT & SYNC =====
            # Export ALL data to files (A-Z comprehensive export)
            try:
                from ..integrations.data_exporter import ComprehensiveDataExporter
                from ..config.settings import settings
                
                logger.info("starting_comprehensive_data_export", message="Exporting ALL engine data A-Z...")
                print("📊 Exporting ALL engine data (A-Z comprehensive export)...\n")
                
                # Initialize exporter with database connection
                # Get database DSN from settings (PostgreSQL connection string)
                db_dsn = None
                if hasattr(settings, 'postgres') and settings.postgres and hasattr(settings.postgres, 'dsn'):
                    db_dsn = settings.postgres.dsn
                elif hasattr(settings, 'database_dsn'):
                    db_dsn = settings.database_dsn
                else:
                    db_dsn = os.getenv("DATABASE_DSN") or os.getenv("POSTGRES_DSN")
                
                exporter = ComprehensiveDataExporter(
                    dsn=db_dsn,
                    output_dir=Path("exports"),
                )
                
                # Export everything
                export_results = exporter.export_all(run_date=start_ts.date())
                
                total_exported = sum(export_results.values())
                logger.info(
                    "comprehensive_data_export_complete",
                    **export_results,
                    total_exports=total_exported,
                )
                print(f"✅ Comprehensive export complete: {total_exported} items exported\n")
                
                # Sync exported data to Dropbox
                if Path("exports").exists():
                    sync_results["exports"] = dropbox_sync.upload_exports("exports")
                    logger.info("exports_synced_to_dropbox", files_synced=sync_results["exports"])
                    print(f"📤 Exported data synced to Dropbox: {sync_results['exports']} files\n")
                
            except Exception as export_error:
                # Export errors are non-fatal - continue without exports
                logger.warning(
                    "comprehensive_data_export_failed",
                    error=str(export_error),
                    message="Continuing without comprehensive export - engine will still run",
                )
                print(f"⚠️  Comprehensive export failed (non-fatal): {str(export_error)}\n")
            
            total_files = sum(sync_results.values())
            if total_files > 0:
                logger.info(
                    "dropbox_initial_sync_complete",
                    **sync_results,
                    total_files=total_files,
                )
                print(f"📤 Initial sync complete: {total_files} files synced\n")
        except Exception as sync_error:
            logger.error("dropbox_continuous_sync_failed", error=str(sync_error))
            print(f"⚠️  Dropbox continuous sync failed: {str(sync_error)}\n")
    
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
    
    # Log DSN (masked for security) to verify config is loaded correctly
    dsn_masked = settings.postgres.dsn.split("@")[-1] if "@" in settings.postgres.dsn else "***"
    logger.info("postgres_dsn_loaded", dsn_masked=dsn_masked, has_password=":" in settings.postgres.dsn.split("@")[0] if "@" in settings.postgres.dsn else False)
    
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
        learning_tracker=learning_tracker,  # Pass learning tracker
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
                healthy_count = sum(1 for s in status_report.services if s.healthy)
                total_count = len(status_report.services)
                telegram_monitor.notify_health_check(
                    status=status_report.overall_status,
                    alerts=[{"message": a.message, "severity": a.severity.value} for a in critical_alerts],
                    warnings=[{"message": a.message, "severity": a.severity.value} for a in warnings],
                    healthy_services=healthy_count,
                    total_services=total_count,
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
                healthy_count = sum(1 for s in status_report.services if s.healthy)
                total_count = len(status_report.services)
                telegram_monitor.notify_health_check(
                    status=status_report.overall_status,
                    alerts=[{"message": a.message, "severity": a.severity.value} for a in critical_alerts],
                    warnings=[{"message": a.message, "severity": a.severity.value} for a in warnings],
                    healthy_services=healthy_count,
                    total_services=total_count,
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
                    healthy_count = sum(1 for s in status_report.services if s.healthy)
                    total_count = len(status_report.services)
                    telegram_monitor.notify_health_check(
                        status="CRITICAL",
                        alerts=[{"message": a.message, "severity": a.severity.value} for a in critical_alerts],
                        warnings=[{"message": a.message, "severity": a.severity.value} for a in warnings],
                        healthy_services=healthy_count,
                        total_services=total_count,
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
        
        # Dropbox sync is already initialized at startup and running in background
        # Continuous sync will keep syncing data every 1 minute
        if dropbox_sync:
            logger.info(
                "dropbox_sync_active",
                folder=dropbox_sync._dated_folder if hasattr(dropbox_sync, "_dated_folder") else "unknown",
                message="Dropbox sync is running in background - data will be synced continuously",
            )
        
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    run_daily_retrain()

