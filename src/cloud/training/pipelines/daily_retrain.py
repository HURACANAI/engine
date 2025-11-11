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
import sys
from datetime import datetime, timezone
from typing import Optional, Dict, List

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
from ..integrations.data_exporter import ManifestBuilder
from ..metrics.enhanced_metrics import EnhancedMetricsCalculator
from pathlib import Path
import uuid
import threading
import time as time_module


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
    2. Runs comprehensive health check
    3. Selects trading universe
    4. Trains models on historical data
    5. Validates with walk-forward testing
    6. Saves models and metrics to S3/Postgres
    7. Generates reports
    
    Runs daily at 02:00 UTC (configured via APScheduler).
    """

    configure_logging()
    logger = structlog.get_logger("daily_retrain")
    start_ts = datetime.now(tz=timezone.utc)
    run_id = f"engine_{start_ts.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    logger.info("run_start", timestamp=start_ts.isoformat(), run_id=run_id)

    settings = EngineSettings.load()
    
    # ===== COMPREHENSIVE HEALTH CHECK =====
    # Run health check before starting - exit if critical failures
    logger.info("running_health_check")
    print("\n" + "=" * 80)
    print("🏥 RUNNING COMPREHENSIVE HEALTH CHECK")
    print("=" * 80)
    print()
    
    from ..services.health_check import validate_health_and_exit
    
    if not validate_health_and_exit(settings=settings, exit_on_failure=True):
        # This should not be reached if exit_on_failure=True, but handle it anyway
        logger.error("health_check_failed", message="Health check failed, shutting down")
        sys.exit(1)
    
    # Initialize feature manager for graceful degradation
    from .feature_manager import get_feature_manager
    feature_manager = get_feature_manager()
    
    # Register all features
    feature_manager.register_feature("dropbox", critical=False, enabled=settings.dropbox.enabled)
    feature_manager.register_feature("telegram", critical=False, enabled=settings.notifications.telegram_enabled)
    feature_manager.register_feature("database", critical=False, enabled=True)  # Database is optional
    feature_manager.register_feature("health_monitor", critical=False, enabled=settings.training.monitoring.enabled)
    feature_manager.register_feature("exchange_client", critical=True, enabled=True)  # CRITICAL - engine needs this
    feature_manager.register_feature("ray", critical=False, enabled=True)
    feature_manager.register_feature("s3", critical=False, enabled=bool(settings.s3 and settings.s3.access_key))
    feature_manager.register_feature("model_registry", critical=False, enabled=True)
    feature_manager.register_feature("telegram_command_handler", critical=False, enabled=settings.notifications.telegram_enabled)
    
    logger.info("features_registered", total_features=len(feature_manager.features))
    
    # ===== CREATE DROPBOX DATED FOLDER IMMEDIATELY (FIRST ACTION) =====
    # This happens before anything else so user knows it works
    dropbox_sync = None
    # Get Dropbox token - clean it (remove whitespace, newlines, etc.)
    # Token can come from environment variable DROPBOX_ACCESS_TOKEN or settings
    env_token = os.getenv("DROPBOX_ACCESS_TOKEN")
    settings_token = settings.dropbox.access_token
    hardcoded_fallback = "sl.u.AGHQ_GZ-nAHejWf16c-t38yKlUaPa7jo9QK7Hrb0RYHPSl5ElgRLq2eoFVt3MTp1pf8vXo7i6--E4c-8Y7_GmnvwvTFE7FdsTzjLdYd9-tIVbtzPtyDr8Y0Eo4qgJ4kmWqMGkowVsRKVJhXA23sMwJrvbNO4vqpGHK5n1wDY9bwyOg1t2uF7CgI0kWW8ftRqkzl8iYCMkPIumkaFwE2kuKn7qbQ0Gqd3LK6s1yFq_DLFoXg8m77Ji_m8m6EpOSdGQuPkfVtOQn2qs9SRtI17KlDHA27dlcpu2-6eiebCafHYZ_l4CvkRCERohITkcAuopSsXhrI9qcEI6NzfwRgrb5NSYHJdTUI-7dzHEMJfLBg4YCBmuetm_lFOS2iMhBZoVQWj2kV9VGvF8FWtjBGCtGnPGb5a69tXcFUPudKDGmWuF4WmpQDLhBzxPCdeo2n-yBV8QNoZ9Qn5pwwfJIPozafSQI6qOr4sSwxGOq_DBQ3QaLTlgy8JGFAxK2f-khqAWfxMdmXvX3FKf-D5cUnjbC5gWGn-IaCAWgQeemncPVLUm5I3fA9_Q12IAJgbhnXZbT-CDj1LdzVo1NyIAIMM7bZFaTkIHquKZPQ9BARkYlznPMJpkzQ4IjnO3M63ne4ZRLFOUwLbVAhVvewPPA6VqotbFG5VnP2eIoJn3W4wcKNRJqPAye7JhUlNUgaJwH8RjZf0YiLUVmiSxhbXGq-7U0jpustKnMhs56rG59gIHaApvWlCWBNGOGqhcvZxi-egZX9m_vQT6jf0tunyb9szmhC7rvaV-oQth1dobmJ10EN-fiKHcmzhlPHzNz1pEhz83gD-WSrfqJr4SgL6nFqmbioNGkQTNqlvxGdMq92qfu9DUL4XN0i-z7SIolqEcBZge5RdPNtBHQCW4osiU0i3FabFWbt6lJIa8-71r93hBgfY4F6uEGVPSfDU0ECEbe0ONAS-hpBXswb33TzeNeKiGGPl2q-eauqbFb1wkLKeghV07Uv3H9BjPHaI2hGW4C9GdKHYiLUphHY0_pYt5xW2l9Ka21N2N8IWkibNT0M4_up4aIRAgTrnYFuSfGreHv8ZzW4PoQ6Q0tx7kjtGbpW7Ej-rzSLptO1-Nu5PNXEBsWbEdQWFrGVNwEfbZ61pdHJ8cV2yCFYlwzW4iGSo4fsE4MSDh6GtUvwlJP9HEPiDmMvQ_xAeMH-Mxa9Oi_pd-G6h6CzuD5jvxh_wNl5eFtGOxJLHWvSTbJ5ls5PjUb1XuTDH6jmHLtSmFfbWEu6Hg0PpKx4"
    
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
    
    # Initialize Dropbox (NON-FATAL - engine continues if it fails)
    if settings.dropbox.enabled:
        if not dropbox_token:
            logger.warning(
                "dropbox_token_missing",
                message="Dropbox is enabled but no token provided. Continuing without Dropbox.",
            )
            print("\n⚠️  WARNING: Dropbox is enabled but no token provided\n")
            print("   💡 To fix: Set DROPBOX_ACCESS_TOKEN environment variable\n")
            print("   💡 Engine will continue without Dropbox sync\n")
            feature_manager.mark_feature_failed("dropbox", "No token provided", stop_engine=False)
            dropbox_sync = None
        else:
            logger.info("dropbox_creating_dated_folder_immediately")
            print("\n📁 Initializing Dropbox...\n")
            
            try:
                # Initialize Dropbox sync - NON-FATAL if it fails
                dropbox_sync = DropboxSync(
                    access_token=dropbox_token,
                    app_folder=dropbox_folder,
                    enabled=True,
                    create_dated_folder=True,
                )
                
                # Verify folder was created
                if hasattr(dropbox_sync, "_dated_folder") and dropbox_sync._dated_folder:
                    logger.info(
                        "dropbox_dated_folder_created_successfully",
                        folder=dropbox_sync._dated_folder,
                        message="✅ Dated folder created - ready for data sync",
                    )
                    print(f"✅ Dropbox initialized: {dropbox_sync._dated_folder}\n")
                    feature_manager.mark_feature_working("dropbox")
                else:
                    raise ValueError("Dropbox folder creation failed")
                    
            except Exception as e:
                logger.warning(
                    "dropbox_init_failed_non_fatal",
                    error=str(e),
                    message="Dropbox initialization failed - engine will continue without Dropbox",
                )
                print(f"⚠️  WARNING: Dropbox initialization failed: {e}\n")
                print("   💡 Engine will continue without Dropbox sync\n")
                feature_manager.mark_feature_failed("dropbox", str(e), stop_engine=False)
                dropbox_sync = None
    else:
        # Dropbox is disabled
        logger.info("dropbox_disabled", message="Dropbox sync is disabled - engine will run without Dropbox")
        print("\n⚠️  Dropbox sync is disabled in settings\n")
        dropbox_sync = None
        feature_manager.mark_feature_failed("dropbox", "Disabled in settings", stop_engine=False)
    
    # Initialize comprehensive Telegram monitoring (NON-FATAL)
    telegram_monitor = None
    telegram_command_handler = None
    log_file = Path("logs") / f"engine_monitoring_{start_ts.strftime('%Y%m%d_%H%M%S')}.log"
    
    if settings.notifications.telegram_enabled and settings.notifications.telegram_chat_id:
        try:
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
            feature_manager.mark_feature_working("telegram")
            
            # Initialize command handler for interactive commands (/health, /status, /help)
            try:
                from ..monitoring.telegram_command_handler import TelegramCommandHandler
                telegram_command_handler = TelegramCommandHandler(
                    bot_token=bot_token,
                    chat_id=settings.notifications.telegram_chat_id,
                    settings=settings,
                    dropbox_sync=dropbox_sync,
                )
                telegram_command_handler.start_polling()
                logger.info("telegram_command_handler_started", commands=["/health", "/status", "/help"])
                print("🤖 Telegram bot commands enabled: /health, /status, /help")
                feature_manager.mark_feature_working("telegram_command_handler")
            except Exception as e:
                logger.warning("telegram_command_handler_init_failed", error=str(e))
                print(f"⚠️  Telegram command handler failed to start: {e}")
                print("   💡 Engine will continue without command handler\n")
                feature_manager.mark_feature_failed("telegram_command_handler", str(e), stop_engine=False)
                telegram_command_handler = None
        except Exception as e:
            logger.warning("telegram_monitoring_init_failed", error=str(e))
            print(f"⚠️  WARNING: Telegram monitoring failed to initialize: {e}\n")
            print("   💡 Engine will continue without Telegram notifications\n")
            feature_manager.mark_feature_failed("telegram", str(e), stop_engine=False)
    else:
        logger.warning("telegram_monitoring_disabled", reason="not configured")
        feature_manager.mark_feature_failed("telegram", "Disabled in settings", stop_engine=False)
    
    # Initialize learning tracker (NON-FATAL)
    learning_tracker = None
    try:
        learning_tracker = LearningTracker(output_dir=Path("logs/learning"))
    except Exception as e:
        logger.warning("learning_tracker_init_failed", error=str(e))
        print(f"⚠️  WARNING: Learning tracker failed to initialize: {e}\n")
        print("   💡 Engine will continue without learning tracker\n")
    
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
            # Historical data is stored in SHARED location (/Runpodhuracan/data/candles/)
            # so it persists across days and can be restored on every startup
            # NOTE: If this is first startup (no data in Dropbox), training will download data normally
            if settings.dropbox.restore_data_cache_on_startup:
                logger.info("attempting_data_cache_restore", message="Checking Dropbox for existing historical data...")
                print("📥 Checking Dropbox for existing historical data...\n")
                
                # Restore from shared location (not dated folder) - this ensures data persists across days
                restored_count = dropbox_sync.restore_data_cache(
                    data_cache_dir="data/candles",
                    remote_dir=None,  # Will use shared location: /Runpodhuracan/data/candles/
                    use_latest_dated_folder=False,  # Use shared location, not dated folder
                )
                
                if restored_count > 0:
                    logger.info(
                        "data_cache_restored",
                        files_restored=restored_count,
                        message="Restored historical data from Dropbox - will skip re-downloading existing data",
                    )
                    print(f"✅ Restored {restored_count} historical data files from Dropbox\n")
                    print("   Only new/missing data will be downloaded from exchange\n")
                else:
                    logger.info(
                        "data_cache_restore_empty",
                        message="No historical data in Dropbox yet (first startup) - data will be downloaded during training"
                    )
                    print("📊 No historical data in Dropbox yet (first startup)\n")
                    print("   Data will be downloaded from exchange during training...\n")
            else:
                logger.info("data_cache_restore_disabled", reason="restore_data_cache_on_startup=false")
                print("⚠️  Data cache restore is disabled in settings\n")
            
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
            
            # Sync historical coin data (in SHARED location - not dated folder)
            # Historical data should persist across days, so store in shared location
            if settings.dropbox.sync_data_cache and Path("data/candles").exists():
                sync_results["data_cache"] = dropbox_sync.upload_data_cache(
                    data_cache_dir="data/candles",
                    use_dated_folder=False,  # Store in shared location (persists across days)
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
                
                logger.info("starting_comprehensive_data_export", message="Exporting ALL engine data A-Z...")
                print("📊 Exporting ALL engine data (A-Z comprehensive export)...\n")
                
                # Initialize exporter with database connection
                # Get database DSN from settings (PostgreSQL connection string)
                # settings is already loaded at the top of the function
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
        
        # Initialize command handler for interactive commands (/health, /status, /help)
        try:
            from ..monitoring.telegram_command_handler import TelegramCommandHandler
            telegram_command_handler = TelegramCommandHandler(
                bot_token=bot_token,
                chat_id=settings.notifications.telegram_chat_id,
                settings=settings,
                dropbox_sync=dropbox_sync,
            )
            telegram_command_handler.start_polling()
            logger.info("telegram_command_handler_started", commands=["/health", "/status", "/help"])
            print("🤖 Telegram bot commands enabled: /health, /status, /help")
        except Exception as e:
            logger.warning("telegram_command_handler_init_failed", error=str(e))
            print(f"⚠️  Telegram command handler failed to start: {e}")
            telegram_command_handler = None
    else:
        logger.warning("telegram_monitoring_disabled", reason="not configured")

    # Initialize Ray (NON-FATAL - engine can work without it)
    try:
        initialize_ray(settings.ray.address, settings.ray.namespace, settings.ray.runtime_env)
        feature_manager.mark_feature_working("ray")
    except Exception as e:
        logger.warning("ray_init_failed_non_fatal", error=str(e))
        print(f"⚠️  WARNING: Ray initialization failed: {e}\n")
        print("   💡 Engine will continue without Ray (may be slower)\n")
        feature_manager.mark_feature_failed("ray", str(e), stop_engine=False)

    # Initialize Exchange Client (CRITICAL - engine needs this to work)
    exchange_client = None
    metadata_loader = None
    universe_selector = None
    try:
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
        feature_manager.mark_feature_working("exchange_client")
        logger.info("exchange_client_initialized", exchange=exchange_settings.primary)
        
        # Get selected symbols for Telegram notification (NON-FATAL)
        try:
            selected_symbols_df = universe_selector.select()
            selected_symbols = selected_symbols_df.to_dicts() if hasattr(selected_symbols_df, 'to_dicts') else selected_symbols_df
            if telegram_monitor and feature_manager.is_feature_working("telegram"):
                telegram_monitor.notify_system_startup(
                    symbols=[s['symbol'] for s in selected_symbols],
                    total_coins=len(selected_symbols),
                )
        except Exception as e:
            logger.warning("universe_selection_failed_non_fatal", error=str(e))
            print(f"⚠️  WARNING: Universe selection failed: {e}\n")
            print("   💡 Engine will continue\n")
            
    except Exception as e:
        error_msg = f"CRITICAL: Exchange client initialization failed: {e}"
        logger.error("exchange_client_init_failed_critical", error=str(e))
        print(f"\n❌ FATAL ERROR: {error_msg}\n")
        print("   💡 Engine CANNOT continue without exchange client\n")
        print("   💡 Please check your exchange credentials and settings\n")
        feature_manager.mark_feature_failed("exchange_client", str(e), stop_engine=True)
        
        # Check if engine should stop
        if feature_manager.should_stop_engine():
            status_report = feature_manager.get_status_report()
            print("\n" + "=" * 60)
            print("🚨 CRITICAL FEATURE FAILURE - ENGINE STOPPING")
            print("=" * 60)
            print(f"\nFailed critical features: {status_report['critical_failed_features']}")
            print(f"Working features: {len(status_report['working_features'])}/{status_report['total_features']}")
            print(f"Failed non-critical features: {status_report['failed_features']}")
            print("\n" + "=" * 60 + "\n")
            sys.exit(1)
    
    # Database is optional (NON-FATAL)
    if not settings.postgres or not settings.postgres.dsn:
        logger.warning("postgres_not_configured", message="Postgres DSN not configured - some features may be disabled")
        print("⚠️  WARNING: Postgres DSN not configured\n")
        print("   💡 Engine will continue but model registry and some features may not work\n")
        feature_manager.mark_feature_failed("database", "Postgres DSN not configured", stop_engine=False)
        if telegram_monitor and feature_manager.is_feature_working("telegram"):
            try:
                telegram_monitor.notify_error("Configuration Warning", "Postgres DSN not configured - some features disabled")
            except Exception:
                pass  # Non-fatal
    
    # Log DSN (masked for security) to verify config is loaded correctly (if available)
    if settings.postgres and settings.postgres.dsn:
        dsn_masked = settings.postgres.dsn.split("@")[-1] if "@" in settings.postgres.dsn else "***"
        logger.info("postgres_dsn_loaded", dsn_masked=dsn_masked, has_password=":" in settings.postgres.dsn.split("@")[0] if "@" in settings.postgres.dsn else False)
    
    # Initialize Model Registry (NON-FATAL)
    model_registry = None
    try:
        if settings.postgres and settings.postgres.dsn:
            model_registry = ModelRegistry(RegistryConfig(dsn=settings.postgres.dsn))
            feature_manager.mark_feature_working("model_registry")
        else:
            logger.warning("model_registry_disabled", reason="No database DSN")
            feature_manager.mark_feature_failed("model_registry", "No database DSN", stop_engine=False)
    except Exception as e:
        logger.warning("model_registry_init_failed_non_fatal", error=str(e))
        print(f"⚠️  WARNING: Model registry initialization failed: {e}\n")
        print("   💡 Engine will continue without model registry\n")
        feature_manager.mark_feature_failed("model_registry", str(e), stop_engine=False)

    # Initialize Notifier (NON-FATAL)
    notifier = None
    try:
        notifier = NotificationClient(settings.notifications)
    except Exception as e:
        logger.warning("notifier_init_failed_non_fatal", error=str(e))
        print(f"⚠️  WARNING: Notifier initialization failed: {e}\n")
        print("   💡 Engine will continue without notifications\n")

    # Initialize Artifact Publisher (NON-FATAL)
    artifact_publisher = None
    try:
        artifact_publisher = ArtifactPublisher(settings.artifacts)
    except Exception as e:
        logger.warning("artifact_publisher_init_failed_non_fatal", error=str(e))
        print(f"⚠️  WARNING: Artifact publisher initialization failed: {e}\n")
        print("   💡 Engine will continue without artifact publishing\n")

    # Initialize health monitoring if enabled
    health_monitor = None
    if settings.training.monitoring.enabled:
        logger.info(
            "===== INITIALIZING HEALTH MONITORING =====",
            check_interval=settings.training.monitoring.check_interval_seconds,
        )
        health_monitor = HealthMonitorOrchestrator(
            settings=settings,
            dsn=settings.postgres.dsn,
            dropbox_sync=dropbox_sync,
        )

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

    # Check if engine should stop due to critical failures
    if feature_manager.should_stop_engine():
        status_report = feature_manager.get_status_report()
        print("\n" + "=" * 60)
        print("🚨 CRITICAL FEATURE FAILURE - ENGINE STOPPING")
        print("=" * 60)
        print(f"\nFailed critical features: {status_report['critical_failed_features']}")
        print(f"Working features: {len(status_report['working_features'])}/{status_report['total_features']}")
        print(f"Failed non-critical features: {status_report['failed_features']}")
        print("\n" + "=" * 60 + "\n")
        sys.exit(1)
    
    # Print feature status summary
    status_report = feature_manager.get_status_report()
    print("\n" + "=" * 60)
    print("📊 FEATURE STATUS SUMMARY")
    print("=" * 60)
    print(f"✅ Working: {status_report['working']}/{status_report['total_features']}")
    if status_report['failed'] > 0:
        print(f"⚠️  Failed (non-critical): {status_report['failed']}")
        for feature_name in status_report['failed_features']:
            feature_info = feature_manager.get_feature_info(feature_name)
            if feature_info:
                print(f"   • {feature_name}: {feature_info.error}")
    print("=" * 60 + "\n")

    # Initialize ManifestBuilder for integration contracts
    manifest_builder = None
    dated_folder_str = ""
    if dropbox_sync and hasattr(dropbox_sync, "_dated_folder") and dropbox_sync._dated_folder:
        # Extract date part from dated folder (e.g., "/Runpodhuracan/2025-11-11" -> "2025-11-11")
        dated_folder_full = dropbox_sync._dated_folder
        if hasattr(dropbox_sync, "_app_folder") and dropbox_sync._app_folder:
            # Remove app folder prefix
            dated_folder_str = dated_folder_full.replace(f"/{dropbox_sync._app_folder}/", "")
        else:
            # Extract date from path (format: /Runpodhuracan/YYYY-MM-DD)
            parts = dated_folder_full.strip("/").split("/")
            if len(parts) >= 2:
                dated_folder_str = parts[-1]  # Last part should be the date
        manifest_builder = ManifestBuilder(
            dropbox_sync=dropbox_sync,
            output_dir=Path("exports"),
        )
        logger.info("manifest_builder_initialized", dated_folder=dated_folder_str, full_path=dated_folder_full)
    
    # Initialize heartbeat tracking
    heartbeat_thread = None
    heartbeat_stop_event = threading.Event()
    current_phase = "loading"
    current_symbol: Optional[str] = None
    training_progress = 0.0
    
    def update_heartbeat():
        """Update heartbeat every 30-60 seconds."""
        while not heartbeat_stop_event.is_set():
            try:
                if manifest_builder:
                    manifest_builder.write_heartbeat(
                        utc_timestamp=datetime.now(tz=timezone.utc),
                        phase=current_phase,
                        current_symbol=current_symbol,
                        progress=training_progress,
                    )
                time_module.sleep(45)  # Update every 45 seconds
            except Exception as e:
                logger.warning("heartbeat_update_failed", error=str(e))
                time_module.sleep(60)  # Retry after 60 seconds on error
    
    # Start heartbeat thread
    if manifest_builder:
        heartbeat_thread = threading.Thread(target=update_heartbeat, daemon=True)
        heartbeat_thread.start()
        logger.info("heartbeat_thread_started")
    
    # Initialize Training Orchestrator
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

    # Track last files written for failure reporting
    last_files_written: List[str] = []
    
    try:
        # Update phase to features
        current_phase = "features"
        training_progress = 0.1
        
        # Run pre-training health check (NON-FATAL)
        if health_monitor and feature_manager.is_feature_working("health_monitor"):
            try:
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
                
                # Notify Telegram about health check (enhanced comprehensive) (NON-FATAL)
                if telegram_monitor and feature_manager.is_feature_working("telegram"):
                    try:
                        # Get enhanced report from health monitor (already ran during health check)
                        enhanced_report = health_monitor.get_enhanced_health_report()
                        
                        if enhanced_report:
                            # Use enhanced report for Telegram notification
                            healthy_count = enhanced_report.summary.get("healthy", 0)
                            total_count = enhanced_report.summary.get("total_checks", 0)
                            
                            logger.info(
                                "sending_enhanced_health_check_to_telegram",
                                overall_status=enhanced_report.overall_status,
                                healthy=healthy_count,
                                total=total_count,
                                critical=enhanced_report.summary.get("critical", 0),
                                warnings=enhanced_report.summary.get("warnings", 0),
                            )
                            
                            telegram_monitor.notify_health_check(
                                status=enhanced_report.overall_status,
                                alerts=[{"message": a.message, "severity": a.severity.value} for a in critical_alerts],
                                warnings=[{"message": a.message, "severity": a.severity.value} for a in warnings],
                                healthy_services=healthy_count,
                                total_services=total_count,
                                health_report=enhanced_report,
                            )
                        else:
                            # Fallback to old system if enhanced report not available
                            logger.warning("enhanced_health_report_not_available_using_fallback")
                            status_reporter = SystemStatusReporter(dsn=settings.postgres.dsn if settings.postgres and settings.postgres.dsn else "")
                            try:
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
                            except Exception as fallback_error:
                                logger.warning("fallback_health_check_failed_non_fatal", error=str(fallback_error))
                    except Exception as telegram_error:
                        logger.warning("telegram_health_check_notification_failed_non_fatal", error=str(telegram_error))
            except Exception as health_check_error:
                logger.warning("pre_training_health_check_failed_non_fatal", error=str(health_check_error))
                print(f"⚠️  WARNING: Pre-training health check failed: {health_check_error}\n")
                print("   💡 Engine will continue\n")

        # Update phase to training
        current_phase = "training"
        training_progress = 0.3
        
        # Run main training
        results = orchestrator.run()
        logger.info("run_complete", total_results=len(results))
        
        # Update phase to validating
        current_phase = "validating"
        training_progress = 0.8
        
        # Save learning snapshot after training
        if learning_tracker:
            try:
                snapshot = learning_tracker.generate_snapshot()
                snapshot_path = learning_tracker.save_snapshot(snapshot)
                if snapshot_path:
                    last_files_written.append(str(snapshot_path))
                    logger.info("learning_snapshot_saved", path=str(snapshot_path))
            except Exception as e:
                logger.warning("learning_snapshot_failed", error=str(e))
        
        # Update phase to publishing
        current_phase = "publishing"
        training_progress = 0.9
        
        # Calculate summary stats for Telegram
        if telegram_monitor and results:
            total_trades = sum(r.metrics.get("trades_oos", 0) for r in results)
            total_profit = sum(r.metrics.get("pnl_bps", 0) * 0.01 for r in results)  # Approximate
            published = sum(1 for r in results if r.published)
            rejected = len(results) - published
            
            # Notify about completion
            end_ts = datetime.now(tz=timezone.utc)
            duration_minutes = int((end_ts - start_ts).total_seconds() / 60)
            
            # Build Dropbox path for notification
            dropbox_path = None
            if dropbox_sync and hasattr(dropbox_sync, "_dated_folder"):
                dropbox_path = f"{dropbox_sync._dated_folder}/manifest.json"
            
            telegram_monitor.notify_system_shutdown(
                total_trades=total_trades,
                total_profit_gbp=total_profit,
                duration_minutes=duration_minutes,
                run_id=run_id,
                dropbox_path=dropbox_path,
            )
        
        # Build manifest and integration contracts
        if manifest_builder and results:
            try:
                # Collect symbols, models, and metrics
                symbols_trained = [r.symbol for r in results]
                model_artifacts_map: Dict[str, str] = {}
                metrics_map: Dict[str, Dict[str, float]] = {}
                published_models: Dict[str, str] = {}
                all_gates_passed = True
                costs_bps_default = 15.0  # Default cost estimate
                
                # Initialize cost-aware metrics calculator
                metrics_calculator = EnhancedMetricsCalculator()
                
                for result in results:
                    symbol = result.symbol
                    
                    # Model artifact path
                    if result.artifacts_path:
                        model_artifacts_map[symbol] = result.artifacts_path
                    elif result.published:
                        model_artifacts_map[symbol] = f"models/{symbol}_model.pkl"
                    
                    # Calculate cost-aware metrics
                    if result.metrics_payload:
                        # Extract returns from metrics if available
                        returns = result.metrics.get("returns", [])
                        if not returns and result.metrics.get("pnl_bps"):
                            # Estimate returns from P&L
                            returns = [result.metrics.get("pnl_bps", 0.0) / 10000.0]
                        
                        # Get cost parameters from CostBreakdown
                        taker_fee = result.costs.fee_bps if hasattr(result.costs, "fee_bps") else 10.0
                        maker_fee = taker_fee * 0.5  # Assume maker is half of taker
                        spread_bps = result.costs.spread_bps if hasattr(result.costs, "spread_bps") else 5.0
                        slippage = result.costs.slippage_bps if hasattr(result.costs, "slippage_bps") else 2.0
                        
                        cost_metrics = metrics_calculator.calculate_cost_aware_metrics(
                            returns=returns if returns else [0.0],
                            trades=None,  # Could pass trade list if available
                            taker_fee_bps=taker_fee,
                            maker_fee_bps=maker_fee,
                            avg_spread_bps={symbol: spread_bps} if spread_bps else None,
                            slippage_bps_per_sigma=slippage,
                        )
                        
                        # Combine with existing metrics
                        metrics_map[symbol] = {
                            **result.metrics,
                            **cost_metrics,
                        }
                    else:
                        metrics_map[symbol] = result.metrics
                    
                    # Check if gates passed
                    if result.gate_results:
                        gates_passed = all(result.gate_results.values())
                        if not gates_passed:
                            all_gates_passed = False
                    
                    # Collect published models for champion pointer
                    if result.published and result.artifacts_path:
                        published_models[symbol] = result.artifacts_path
                
                # Data paths
                data_paths = {
                    "candles_dir": "data/candles",
                    "features_dir": "data/features",
                    "logs_dir": "logs",
                }
                
                # Engine version
                engine_version = getattr(settings, "version", "1.0.0")
                
                # Write manifest
                end_ts = datetime.now(tz=timezone.utc)
                manifest_path = manifest_builder.write_manifest(
                    run_id=run_id,
                    utc_started=start_ts,
                    utc_finished=end_ts,
                    engine_version=engine_version,
                    symbols_trained=symbols_trained,
                    model_artifacts_map=model_artifacts_map,
                    metrics_map=metrics_map,
                    data_paths=data_paths,
                    status="ok",
                    dated_folder=dated_folder_str,
                )
                if manifest_path:
                    last_files_written.append(str(manifest_path))
                    logger.info("manifest_written", run_id=run_id, path=str(manifest_path))
                
                # Update champion pointer if gates passed
                if all_gates_passed and published_models:
                    date_str = start_ts.strftime("%Y-%m-%d")
                    champion_updated = manifest_builder.write_champion_pointer(
                        date=date_str,
                        run_id=run_id,
                        models=published_models,
                        costs_bps_default=costs_bps_default,
                        gate_passed=True,
                    )
                    if champion_updated:
                        logger.info("champion_pointer_updated", run_id=run_id)
                
                # Write mechanic contract for published models
                if published_models:
                    # Collect promote rules from settings or use defaults
                    promote_rules = {
                        "min_sharpe": 1.0,
                        "min_hit_rate": 0.5,
                        "max_drawdown": 0.15,
                    }
                    
                    mechanic_contract_path = manifest_builder.write_mechanic_contract(
                        baseline_run_id=run_id,
                        symbols=list(published_models.keys()),
                        model_paths=published_models,
                        evaluation_window_hours=24,  # Recent 24 hours
                        promote_rules=promote_rules,
                        dated_folder=dated_folder_str,
                    )
                    if mechanic_contract_path:
                        last_files_written.append(str(mechanic_contract_path))
                        logger.info("mechanic_contract_written", path=str(mechanic_contract_path))
                
                # Write feature recipes for each symbol
                for result in results:
                    if result.feature_metadata:
                        try:
                            # Extract feature recipe parameters
                            timeframes = ["1h"]  # Default, could be extracted from settings
                            indicators = result.feature_metadata.get("feature_importances", {})
                            fill_rules = {"strategy": "forward_fill"}
                            normalization = {"type": "standard", "scaler": "StandardScaler"}
                            
                            feature_recipe_path = manifest_builder.write_feature_recipe(
                                symbol=result.symbol,
                                timeframes=timeframes,
                                indicators=indicators,
                                fill_rules=fill_rules,
                                normalization=normalization,
                                dated_folder=dated_folder_str,
                            )
                            if feature_recipe_path:
                                last_files_written.append(str(feature_recipe_path))
                        except Exception as e:
                            logger.warning("feature_recipe_write_failed", symbol=result.symbol, error=str(e))
                
            except Exception as e:
                logger.error("manifest_building_failed", error=str(e), exc_info=True)
        
        # Update progress to complete
        training_progress = 1.0
        current_phase = "complete"

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
            
            # Notify Telegram about health check (enhanced comprehensive)
            if telegram_monitor:
                # Get enhanced report from health monitor (already ran during health check)
                enhanced_report = health_monitor.get_enhanced_health_report()
                
                if enhanced_report:
                    # Use enhanced report for Telegram notification
                    healthy_count = enhanced_report.summary.get("healthy", 0)
                    total_count = enhanced_report.summary.get("total_checks", 0)
                    
                    logger.info(
                        "sending_enhanced_health_check_to_telegram",
                        overall_status=enhanced_report.overall_status,
                        healthy=healthy_count,
                        total=total_count,
                        critical=enhanced_report.summary.get("critical", 0),
                        warnings=enhanced_report.summary.get("warnings", 0),
                    )
                    
                    telegram_monitor.notify_health_check(
                        status=enhanced_report.overall_status,
                        alerts=[{"message": a.message, "severity": a.severity.value} for a in critical_alerts],
                        warnings=[{"message": a.message, "severity": a.severity.value} for a in warnings],
                        healthy_services=healthy_count,
                        total_services=total_count,
                        health_report=enhanced_report,
                    )
                else:
                    # Fallback to old system if enhanced report not available
                    logger.warning("enhanced_health_report_not_available_using_fallback")
                    status_reporter = SystemStatusReporter(dsn=settings.postgres.dsn if settings.postgres else "")
                    try:
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
                    except Exception as fallback_error:
                        logger.exception("fallback_health_check_failed", error=str(fallback_error))

    except Exception as exc:  # noqa: BLE001
        logger.exception("run_failed", error=str(exc))
        
        # Update phase to failed
        current_phase = "failed"
        training_progress = 0.0
        
        # Write failure report
        if manifest_builder:
            try:
                # Generate suggestions based on error type
                suggestions = [
                    "Check logs for detailed error messages",
                    "Verify exchange connection and credentials",
                    "Check database connectivity",
                    "Verify data availability for training",
                ]
                
                if "database" in str(exc).lower() or "postgres" in str(exc).lower():
                    suggestions.append("Verify PostgreSQL connection string")
                if "exchange" in str(exc).lower() or "api" in str(exc).lower():
                    suggestions.append("Verify exchange API credentials")
                if "memory" in str(exc).lower() or "oom" in str(exc).lower():
                    suggestions.append("Reduce batch size or number of symbols")
                
                failure_report_path = manifest_builder.write_failure_report(
                    run_id=run_id,
                    step=current_phase,
                    exception_type=type(exc).__name__,
                    message=str(exc),
                    last_files_written=last_files_written,
                    suggestions=suggestions,
                    dated_folder=dated_folder_str,
                )
                if failure_report_path:
                    logger.error("failure_report_written", path=str(failure_report_path))
            except Exception as failure_error:
                logger.error("failure_report_write_failed", error=str(failure_error))
        
        # Notify Telegram about error
        if telegram_monitor:
            telegram_monitor.notify_error(
                error_type=type(exc).__name__,
                error_message=str(exc),
                context={"timestamp": datetime.now(tz=timezone.utc).isoformat(), "run_id": run_id},
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
        # Stop heartbeat thread
        if heartbeat_thread:
            heartbeat_stop_event.set()
            heartbeat_thread.join(timeout=5.0)
            logger.info("heartbeat_thread_stopped")
        
        # Final heartbeat update
        if manifest_builder:
            try:
                manifest_builder.write_heartbeat(
                    utc_timestamp=datetime.now(tz=timezone.utc),
                    phase=current_phase,
                    current_symbol=None,
                    progress=training_progress,
                )
            except Exception as e:
                logger.warning("final_heartbeat_failed", error=str(e))
        
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
        
        # Stop command handler
        if telegram_command_handler:
            telegram_command_handler.stop_polling()
            logger.info("telegram_command_handler_stopped")
        
        if ray.is_initialized():
            ray.shutdown()
        
        # Final summary
        end_ts = datetime.now(tz=timezone.utc)
        duration_seconds = (end_ts - start_ts).total_seconds()
        logger.info(
            "daily_retrain_complete",
            run_id=run_id,
            duration_seconds=duration_seconds,
            duration_minutes=int(duration_seconds / 60),
            status=current_phase,
        )


if __name__ == "__main__":
    run_daily_retrain()

