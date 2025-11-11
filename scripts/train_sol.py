#!/usr/bin/env python3
"""
Train SOL coin - Full Pipeline Test

This script runs the complete training pipeline for SOL:
1. Download historical data (150 days)
2. Feature engineering
3. Labeling with validation
4. LightGBM training (2000 estimators, intensive)
5. RL training (shadow trading)
6. Model evaluation
7. Save to Dropbox
8. Register in database

Usage:
    python scripts/train_sol.py
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
    """Run training on SOL coin."""
    print("=" * 80)
    print("HUracan ENGINE - SOL TRAINING (FULL PIPELINE TEST)")
    print("=" * 80)
    print()
    print("Pipeline steps:")
    print("  1. Download 150 days of historical data")
    print("  2. Feature engineering (75+ features)")
    print("  3. Labeling with validation")
    print("  4. LightGBM training (2000 estimators, intensive)")
    print("  5. RL training (shadow trading)")
    print("  6. Model evaluation")
    print("  7. Save to Dropbox")
    print("  8. Register in database")
    print()
    
    # Load settings
    logger.info("loading_settings")
    environment = os.getenv("HURACAN_ENV", "runpod")
    settings = EngineSettings.load(environment=environment)
    
    # Override universe to only SOL
    # We'll manually set the symbol to SOL/USDT or SOL/USDC
    settings.universe.target_size = 1
    logger.info("universe_size_override", target_size=1)
    
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
    
    # Create a custom universe selector that only returns SOL
    class SOLUniverseSelector:
        """Universe selector that only returns SOL."""
        
        def __init__(self, exchange_client, metadata_loader, settings):
            self._exchange = exchange_client
            self._metadata_loader = metadata_loader
            self._settings = settings
        
        def select(self):
            import polars as pl
            # Get SOL/USDT or SOL/USDC
            markets = self._exchange.fetch_markets()
            sol_symbols = [s for s in markets.keys() if s.startswith("SOL/") and ("USDT" in s or "USDC" in s)]
            
            if not sol_symbols:
                raise ValueError("SOL trading pair not found on exchange")
            
            # Prefer USDT, then USDC
            sol_symbol = None
            for preferred in ["SOL/USDT", "SOL/USDC"]:
                if preferred in sol_symbols:
                    sol_symbol = preferred
                    break
            
            if not sol_symbol:
                sol_symbol = sol_symbols[0]
            
            logger.info("sol_symbol_selected", symbol=sol_symbol, available=sol_symbols)
            
            # Get liquidity data for SOL
            liquidity = self._metadata_loader.liquidity_snapshot([sol_symbol])
            fees = self._metadata_loader.fee_schedule([sol_symbol])
            
            if liquidity.is_empty():
                # Create dummy data if liquidity not available
                liquidity = pl.DataFrame({
                    "symbol": [sol_symbol],
                    "quote_volume": [100000000.0],  # 100M
                    "spread_bps": [5.0],
                })
                fees = pl.DataFrame({
                    "symbol": [sol_symbol],
                    "taker_fee_bps": [10.0],
                })
            
            merged = liquidity.join(fees, on="symbol", how="outer")
            return merged.with_columns(pl.Series("rank", [1]))
    
    universe_selector = SOLUniverseSelector(
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
            bot_token = os.getenv("TELEGRAM_BOT_TOKEN") or settings.notifications.telegram_bot_token
            chat_id = os.getenv("TELEGRAM_CHAT_ID") or settings.notifications.telegram_chat_id
            if bot_token and chat_id:
                telegram_monitor = ComprehensiveTelegramMonitor(
                    bot_token=bot_token,
                    chat_id=chat_id,
                )
                logger.info("telegram_monitor_enabled")
            else:
                logger.warning("telegram_credentials_missing", message="Telegram enabled but bot_token or chat_id missing")
        except Exception as e:
            logger.warning("telegram_monitor_init_failed", error=str(e))
    
    # Initialize learning tracker if available
    learning_tracker = None
    try:
        from src.cloud.training.monitoring.learning_tracker import LearningTracker
        learning_tracker = LearningTracker(output_dir=Path("logs/learning"))
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
    print("Starting SOL training...")
    print("This may take 10-30 minutes depending on data size and hardware.")
    print("Steps:")
    print("  - Downloading 150 days of 1-minute candles")
    print("  - Feature engineering (75+ features)")
    print("  - Labeling with validation")
    print("  - LightGBM training (2000 estimators)")
    print("  - RL training (shadow trading)")
    print("  - Model evaluation")
    print("  - Saving to Dropbox")
    print()
    
    try:
        results = orchestrator.run()
        
        # Print results
        print()
        print("=" * 80)
        print("TRAINING RESULTS")
        print("=" * 80)
        print(f"{'Symbol':<15} {'Status':<12} {'Published':<10} {'Reason':<40}")
        print("-" * 80)
        
        for result in results:
            status = "✅ Success" if result.published else "❌ Rejected"
            reason = result.reason[:40] if result.reason else "N/A"
            print(f"{result.symbol:<15} {status:<12} {str(result.published):<10} {reason:<40}")
        
        print("=" * 80)
        print(f"Total coins: {len(results)}")
        print(f"Published: {sum(1 for r in results if r.published)}")
        print(f"Rejected: {sum(1 for r in results if not r.published)}")
        print("=" * 80)
        
        # Print detailed metrics for SOL
        sol_results = [r for r in results if "SOL" in r.symbol]
        if sol_results:
            result = sol_results[0]
            print()
            print("SOL TRAINING DETAILS:")
            print("-" * 80)
            if result.published:
                print("✅ Model Published Successfully")
                if result.metrics:
                    metrics = result.metrics
                    print(f"  Sharpe Ratio: {metrics.get('sharpe', 'N/A'):.2f}")
                    print(f"  Profit Factor: {metrics.get('profit_factor', 'N/A'):.2f}")
                    print(f"  Hit Rate: {metrics.get('hit_rate', 'N/A'):.2%}")
                    print(f"  Max Drawdown: {metrics.get('max_dd_bps', 'N/A'):.1f} bps")
                    print(f"  PnL (bps): {metrics.get('pnl_bps', 'N/A'):.1f}")
                    print(f"  Trades (OOS): {metrics.get('trades_oos', 'N/A')}")
                    print(f"  Total Costs: {result.costs.total_costs_bps:.1f} bps")
                    print(f"  Recommended Edge Threshold: {metrics.get('recommended_edge_threshold_bps', 'N/A')} bps")
                
                if result.artifacts_path:
                    print(f"  Artifacts Path: {result.artifacts_path}")
                if result.model_id:
                    print(f"  Model ID: {result.model_id}")
            else:
                print("❌ Model Rejected")
                print(f"  Reason: {result.reason}")
                if result.metrics:
                    metrics = result.metrics
                    print(f"  Metrics: {metrics}")
            
            print("-" * 80)
        
        # Check Dropbox sync status
        if settings.dropbox.enabled:
            print()
            print("DROPBOX SYNC:")
            print("-" * 80)
            print("✅ Dropbox sync enabled")
            print("  Models and data will be synced to Dropbox")
            print("  Check Dropbox folder for uploaded files")
            print("-" * 80)
        
        logger.info("training_complete", total_coins=len(results), published=sum(1 for r in results if r.published))
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("training_interrupted_by_user")
        print("\nTraining interrupted by user.")
        return 1
    except Exception as e:
        logger.exception("training_failed", error=str(e))
        print(f"\nERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
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

