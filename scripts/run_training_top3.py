#!/usr/bin/env python3
"""
Run intensive training on top 3 coins with all advanced features enabled.

This script:
1. Sets universe to top 3 coins
2. Runs full training pipeline with all features:
   - Multi-model ensemble (XGBoost, Random Forest, LightGBM)
   - Enhanced RL training (V2 pipeline)
   - Intensive model training (2000 estimators)
   - All Phase 1 features enabled
3. Uploads models to Dropbox
4. Sends Telegram notifications

Usage:
    python scripts/run_training_top3.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import ray
import structlog
from datetime import date
from src.cloud.training.config.settings import EngineSettings
from src.cloud.training.services.exchange import ExchangeClient
from src.cloud.training.services.universe import UniverseSelector
from src.cloud.training.services.model_registry import ModelRegistry, RegistryConfig
from src.cloud.training.services.notifications import NotificationClient
from src.cloud.training.services.artifacts import ArtifactBundle, ArtifactPublisher
from src.cloud.training.services.orchestration import TrainingOrchestrator
from src.cloud.training.datasets.data_loader import MarketMetadataLoader

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger("INFO"),
)

logger = structlog.get_logger(__name__)


class DummyArtifactPublisher:
    """Dummy artifact publisher that does nothing (for when S3 is not configured)."""
    
    def publish(self, run_date: date, symbol: str, bundle: ArtifactBundle) -> str:
        """Dummy publish that returns empty string."""
        logger.info("dummy_artifact_publisher_used", symbol=symbol, message="Artifacts saved locally/Dropbox only")
        return ""


def main():
    """Run training on top 3 coins."""
    print("=" * 80)
    print("HUracan ENGINE - INTENSIVE TRAINING ON TOP 3 COINS")
    print("=" * 80)
    print()
    print("Features enabled:")
    print("  ✅ Multi-model ensemble (XGBoost, Random Forest, LightGBM)")
    print("  ✅ Enhanced RL training (V2 pipeline)")
    print("  ✅ Intensive model training (2000 estimators)")
    print("  ✅ Triple-barrier labeling")
    print("  ✅ Meta-labeling")
    print("  ✅ Advanced rewards")
    print("  ✅ Higher-order features")
    print("  ✅ Granger causality")
    print("  ✅ Regime prediction")
    print()
    
    # Load settings
    logger.info("loading_settings")
    environment = os.getenv("HURACAN_ENV", "runpod")
    settings = EngineSettings.load(environment=environment)
    
    # Override universe size to 3 for this test
    settings.universe.target_size = 3
    logger.info("universe_size_override", target_size=3)
    
    # Verify settings
    if not settings.postgres or not settings.postgres.dsn:
        logger.error("postgres_dsn_required")
        print("ERROR: PostgreSQL DSN is required but not configured.")
        print("Please set POSTGRES_DSN environment variable or configure in settings.")
        sys.exit(1)
    
    # Initialize Ray
    logger.info("initializing_ray")
    try:
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                num_cpus=os.cpu_count() or 4,
                object_store_memory=2_000_000_000,  # 2GB
            )
            logger.info("ray_initialized", num_cpus=os.cpu_count())
    except Exception as e:
        logger.warning("ray_init_failed", error=str(e), message="Continuing without Ray")
    
    # Initialize exchange client
    logger.info("initializing_exchange_client")
    primary_exchange = settings.exchange.primary
    credentials = settings.exchange.credentials.get(primary_exchange, {})
    exchange = ExchangeClient(
        exchange_id=primary_exchange,
        credentials=credentials,
        sandbox=settings.exchange.sandbox,
    )
    
    # Initialize services
    logger.info("initializing_services")
    metadata_loader = MarketMetadataLoader(exchange_client=exchange)
    universe_selector = UniverseSelector(
        exchange_client=exchange,
        metadata_loader=metadata_loader,
        settings=settings.universe,
    )
    model_registry = ModelRegistry(config=RegistryConfig(dsn=settings.postgres.dsn))
    notifier = NotificationClient(settings=settings.notifications)
    
    # Initialize artifact publisher (use S3 settings, or create dummy if not configured)
    try:
        if settings.artifacts and settings.artifacts.bucket:
            artifact_publisher = ArtifactPublisher(settings=settings.artifacts)
            logger.info("artifact_publisher_initialized", bucket=settings.artifacts.bucket)
        else:
            # Use dummy publisher - models will still be saved locally and to Dropbox
            artifact_publisher = DummyArtifactPublisher()
            logger.info("using_dummy_artifact_publisher", message="S3 not configured, artifacts will be saved locally/Dropbox only")
    except Exception as e:
        logger.warning("artifact_publisher_init_failed", error=str(e), message="Using dummy publisher")
        artifact_publisher = DummyArtifactPublisher()
    
    # Initialize Telegram monitor if enabled
    telegram_monitor = None
    if settings.notifications.telegram_enabled:
        try:
            from src.cloud.training.monitoring.comprehensive_telegram_monitor import ComprehensiveTelegramMonitor
            telegram_monitor = ComprehensiveTelegramMonitor(settings=settings)
            logger.info("telegram_monitor_enabled")
        except Exception as e:
            logger.warning("telegram_monitor_init_failed", error=str(e))
    
    # Initialize learning tracker if database available
    learning_tracker = None
    if settings.postgres and settings.postgres.dsn:
        try:
            from src.cloud.training.monitoring.learning_tracker import LearningTracker
            learning_tracker = LearningTracker(dsn=settings.postgres.dsn)
            logger.info("learning_tracker_enabled")
        except Exception as e:
            logger.warning("learning_tracker_init_failed", error=str(e))
    
    # Create orchestrator
    logger.info("creating_training_orchestrator")
    orchestrator = TrainingOrchestrator(
        settings=settings,
        exchange_client=exchange,
        universe_selector=universe_selector,
        model_registry=model_registry,
        notifier=notifier,
        artifact_publisher=artifact_publisher,
        telegram_monitor=telegram_monitor,
        learning_tracker=learning_tracker,
    )
    
    # Run training
    print()
    print("Starting training on top 3 coins...")
    print("This may take 20-90 minutes depending on data size and hardware.")
    print()
    
    try:
        results = orchestrator.run()
        
        # Print results
        print()
        print("=" * 80)
        print("TRAINING RESULTS")
        print("=" * 80)
        print(f"{'Symbol':<15} {'Status':<12} {'Published':<10} {'Reason':<30}")
        print("-" * 80)
        
        for result in results:
            status = "✅ Success" if result.published else "❌ Rejected"
            print(f"{result.symbol:<15} {status:<12} {str(result.published):<10} {result.reason[:30]:<30}")
        
        print("=" * 80)
        print(f"Total coins: {len(results)}")
        print(f"Published: {sum(1 for r in results if r.published)}")
        print(f"Rejected: {sum(1 for r in results if not r.published)}")
        print("=" * 80)
        
        # Print metrics for published models
        published_results = [r for r in results if r.published]
        if published_results:
            print()
            print("PUBLISHED MODELS METRICS:")
            print("-" * 80)
            print(f"{'Symbol':<15} {'Sharpe':<10} {'Hit Rate':<10} {'PnL (bps)':<12} {'Trades':<8}")
            print("-" * 80)
            for result in published_results:
                metrics = result.metrics
                sharpe = metrics.get("sharpe", 0.0)
                hit_rate = metrics.get("hit_rate", 0.0)
                pnl_bps = metrics.get("pnl_bps", 0.0)
                trades = metrics.get("trades_oos", 0)
                print(f"{result.symbol:<15} {sharpe:<10.2f} {hit_rate:<10.2%} {pnl_bps:<12.1f} {trades:<8}")
            print("=" * 80)
        
        logger.info("training_complete", total_coins=len(results), published=sum(1 for r in results if r.published))
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("training_interrupted_by_user")
        print("\nTraining interrupted by user.")
        return 1
    except Exception as e:
        logger.exception("training_failed", error=str(e))
        print(f"\nERROR: Training failed: {e}")
        return 1
    finally:
        # Shutdown Ray
        try:
            if ray.is_initialized():
                ray.shutdown()
                logger.info("ray_shutdown")
        except Exception as e:
            logger.warning("ray_shutdown_failed", error=str(e))


if __name__ == "__main__":
    sys.exit(main())
