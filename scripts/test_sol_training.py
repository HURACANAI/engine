"""
Test script to run SOL training from start to finish.

This script will:
1. Load configuration
2. Connect to exchange
3. Fetch SOL data
4. Run training pipeline
5. Report results and any errors

Usage:
    python scripts/test_sol_training.py
"""

import sys
from pathlib import Path

# Ensure project root and src are on the path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

import pandas as pd
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
    """Test SOL training from start to finish."""
    print("=" * 80)
    print("SOL TRAINING TEST - START TO FINISH")
    print("=" * 80)
    print()

    try:
        # Step 1: Load settings
        print("[1/5] Loading engine settings...")
        settings = EngineSettings.load()
        print("✓ Settings loaded successfully")
        print()

        # Step 2: Validate database connection
        print("[2/5] Validating database configuration...")
        if not settings.postgres or not settings.postgres.dsn:
            raise RuntimeError("PostgreSQL DSN is required for training but was not configured.")
        print(f"✓ Database DSN configured: {settings.postgres.dsn[:30]}...")
        print()

        # Step 3: Connect to exchange
        print("[3/5] Connecting to exchange...")
        primary_exchange = settings.exchange.primary

        # Use real exchange data (read-only, no credentials needed for public data)
        exchange = ExchangeClient(
            exchange_id=primary_exchange,
            credentials=None,
            sandbox=False,  # Use real data, not sandbox
        )
        print(f"✓ Connected to exchange: {exchange.exchange_id}")
        print()

        # Step 4: Initialize pipeline
        print("[4/5] Initializing training pipeline...")
        pipeline = EnhancedRLPipeline(
            settings=settings,
            dsn=settings.postgres.dsn,
        )
        print("✓ Pipeline initialized successfully")
        print()

        # Step 5: Run training for SOL
        print("[5/5] Training SOL/USDT...")
        print("-" * 80)

        symbol = "SOL/USDT"
        lookback_days = 90  # Use 90 days for more training data

        print(f"Symbol: {symbol}")
        print(f"Lookback: {lookback_days} days")
        print()

        # Run training (pipeline will fetch data itself)
        print("Running training pipeline (will fetch data automatically)...")
        result = pipeline.train_on_symbol(
            symbol=symbol,
            exchange_client=exchange,
            lookback_days=lookback_days,
        )

        # Print results
        print()
        print("=" * 80)
        print("TRAINING RESULTS")
        print("=" * 80)
        print()

        if result.get("success", False):
            print("✅ Training completed successfully!")
            print()
            print("Metrics:")
            metrics = result.get("metrics", {})
            print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}")
            print(f"  Win Rate: {metrics.get('win_rate', 'N/A')}")
            print(f"  Total Return: {metrics.get('total_return', 'N/A')}")
            print(f"  Max Drawdown: {metrics.get('max_drawdown', 'N/A')}")
            print()

            if result.get("model_path"):
                print(f"Model saved to: {result['model_path']}")

            if result.get("warnings"):
                print()
                print("Warnings:")
                for warning in result["warnings"]:
                    print(f"  ⚠️  {warning}")
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
        print("TEST COMPLETE")
        print("=" * 80)

    except Exception as e:
        logger.error("test_failed", error=str(e), exc_info=True)
        print()
        print("=" * 80)
        print("❌ TEST FAILED")
        print("=" * 80)
        print()
        print(f"Error: {e}")
        import traceback
        print()
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
