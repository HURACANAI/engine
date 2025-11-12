"""
Full SOL training script with Dropbox export.

This script will:
1. Train the model on SOL/USDT with 90 days of data
2. Save the trained model
3. Export results and model to Dropbox
4. Generate performance reports

Usage:
    python scripts/train_sol_full.py
"""

import sys
from pathlib import Path
from datetime import datetime

# Ensure project root and src are on the path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

import structlog
from cloud.training.config.settings import EngineSettings
from cloud.training.pipelines.enhanced_rl_pipeline import EnhancedRLPipeline
from cloud.training.services.exchange import ExchangeClient

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)

logger = structlog.get_logger(__name__)


def main():
    """Train SOL model and export to Dropbox."""
    print("=" * 80)
    print("FULL SOL TRAINING - WITH DROPBOX EXPORT")
    print("=" * 80)
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # Step 1: Load settings
        print("[1/6] Loading engine settings...")
        settings = EngineSettings.load()
        print("✓ Settings loaded successfully")
        print()

        # Step 2: Validate configuration
        print("[2/6] Validating configuration...")
        if not settings.postgres or not settings.postgres.dsn:
            raise RuntimeError("PostgreSQL DSN is required")
        if not settings.dropbox or not settings.dropbox.access_token:
            print("⚠️  Warning: Dropbox not configured - models will only be saved locally")
        print("✓ Configuration validated")
        print()

        # Step 3: Connect to exchange
        print("[3/6] Connecting to Binance...")
        exchange = ExchangeClient(
            exchange_id=settings.exchange.primary,
            credentials=None,
            sandbox=False,  # Use real data
        )
        print(f"✓ Connected to {exchange.exchange_id}")
        print()

        # Step 4: Initialize pipeline
        print("[4/6] Initializing training pipeline...")
        pipeline = EnhancedRLPipeline(
            settings=settings,
            dsn=settings.postgres.dsn,
        )
        print("✓ Pipeline initialized")
        print()

        # Step 5: Train on SOL
        symbol = "SOL/USDT"
        lookback_days = 90

        print(f"[5/6] Training {symbol}...")
        print(f"  Lookback: {lookback_days} days")
        print(f"  Timestamp: {timestamp}")
        print()

        result = pipeline.train_on_symbol(
            symbol=symbol,
            exchange_client=exchange,
            lookback_days=lookback_days,
        )

        # Step 6: Save and export results
        print()
        print("[6/6] Saving and exporting results...")

        if result.get("success", False):
            print("✓ Training completed successfully!")
            print()

            metrics = result.get("metrics", {})
            print("Performance Metrics:")
            print(f"  Total Trades: {metrics.get('total_trades', 0)}")
            print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
            print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  Total Profit (GBP): {metrics.get('total_profit_gbp', 0):.2f}")
            print(f"  Mean Return (bps): {metrics.get('mean_return_bps', 0):.2f}")
            print()

            # Save model locally
            if result.get("model_path"):
                print(f"Model saved locally: {result['model_path']}")

            # Note: The pipeline should already handle Dropbox export if configured
            if result.get("dropbox_path"):
                print(f"✓ Model also saved to Dropbox: {result['dropbox_path']}")

            print()
            print("=" * 80)
            print("✅ TRAINING COMPLETE - SUCCESS")
            print("=" * 80)

        else:
            print("❌ Training failed!")
            error = result.get("error", "Unknown error")
            print(f"Error: {error}")

            if result.get("details"):
                print()
                print("Details:")
                for key, value in result["details"].items():
                    print(f"  {key}: {value}")

            print()
            print("=" * 80)
            print("❌ TRAINING FAILED")
            print("=" * 80)
            sys.exit(1)

    except Exception as e:
        logger.error("training_failed", error=str(e), exc_info=True)
        print()
        print("=" * 80)
        print("❌ TRAINING FAILED")
        print("=" * 80)
        print()
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
