"""
Example: Run Progressive Historical Training

Trains on full coin history (inception to present), prioritizing recent data.

Usage:
    python scripts/run_progressive_training.py
"""

import sys
from pathlib import Path

# Ensure project root and src are on the path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from cloud.training.config.settings import EngineSettings
from cloud.training.pipelines.enhanced_rl_pipeline import EnhancedRLPipeline
from cloud.training.pipelines.progressive_training import (
    ProgressiveHistoricalTrainer,
    ProgressiveTrainingConfig,
)
from cloud.training.services.exchange import ExchangeClient


def main():
    """Run progressive training on multiple coins."""
    print("=" * 80)
    print("PROGRESSIVE HISTORICAL TRAINING")
    print("=" * 80)
    print()
    print("Training on FULL coin history:")
    print("  1. Start with most recent 2 years (most relevant)")
    print("  2. Progressively train on older data")
    print("  3. Continue until coin inception")
    print()

    # Symbols to train
    symbols = [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
    ]

    print(f"Symbols: {', '.join(symbols)}")
    print()

    # Load settings
    print("Loading engine settings...")
    settings = EngineSettings.load()

    if not settings.postgres or not settings.postgres.dsn:
        raise RuntimeError("PostgreSQL DSN is required for training but was not configured.")

    dsn = settings.postgres.dsn

    # Create exchange client
    print("Connecting to exchange...")
    primary_exchange = settings.exchange.primary
    creds_model = settings.exchange.credentials.get(primary_exchange)
    # Training workflows never require authenticated trading credentials.
    # Force read-only mode to eliminate any possibility of live order placement.
    credentials = None
    if creds_model:
        credentials = {
            "api_key": creds_model.api_key,
            "api_secret": creds_model.api_secret,
            "api_passphrase": creds_model.api_passphrase,
        }
        if any(credentials.values()):
            print("⚠️  Warning: Credentials detected but will be ignored during training (read-only mode).")
        credentials = None

    sandbox_mode = True
    if not settings.exchange.sandbox:
        print("ℹ️  Forcing sandbox mode for training to avoid live trading.")

    exchange = ExchangeClient(
        exchange_id=primary_exchange,
        credentials=credentials,
        sandbox=sandbox_mode,
    )
    print(f"Connected to: {exchange.exchange_id}")
    print()

    # Configure progressive training
    config = ProgressiveTrainingConfig(
        initial_epoch_days=730,  # Start with 2 years
        subsequent_epoch_days=365,  # Then 1 year at a time
        max_epochs=None,  # Train until inception (no limit)
        train_from_scratch_first_epoch=True,  # First epoch from scratch
        fine_tune_subsequent_epochs=True,  # Older epochs fine-tune
        save_checkpoints_per_epoch=True,  # Save after each epoch
        min_data_points_per_epoch=10000,  # Skip epochs with <10k candles
        early_stop_if_performance_degrades=True,  # Stop if old data hurts
    )

    print("Configuration:")
    print(f"  Initial epoch: {config.initial_epoch_days} days (~2 years)")
    print(f"  Subsequent epochs: {config.subsequent_epoch_days} days (~1 year)")
    print(f"  Max epochs: {config.max_epochs or 'No limit (until inception)'}")
    print(f"  Fine-tune old epochs: {config.fine_tune_subsequent_epochs}")
    print(f"  Early stop if degraded: {config.early_stop_if_performance_degrades}")
    print()

    # Create base pipeline
    print("Initializing base training pipeline...")
    base_pipeline = EnhancedRLPipeline(settings=settings, dsn=dsn)
    print("Base pipeline ready")
    print()

    # Create progressive trainer
    trainer = ProgressiveHistoricalTrainer(
        config=config,
        base_pipeline=base_pipeline,
    )

    # Train all symbols
    print("=" * 80)
    print("STARTING PROGRESSIVE TRAINING")
    print("=" * 80)
    print()

    results = trainer.train_all_symbols(
        symbols=symbols,
        exchange_client=exchange,
        timeframe="1h",  # Use 1h candles for faster testing
    )

    # Print results
    print()
    print("=" * 80)
    print("TRAINING RESULTS")
    print("=" * 80)
    print()

    for symbol, result in results.items():
        print(f"\n{symbol}:")
        print("-" * 40)

        if not result.get("success", False):
            print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
            continue

        print(f"  ✅ Success")
        print(f"  Inception: {result['inception_date']}")
        print(f"  Total epochs: {result['total_epochs']}")
        print(f"  Trained: {result['epochs_trained']}")
        print(f"  Skipped: {result['epochs_skipped']}")

        # Show epoch details
        if result.get("epoch_results"):
            print(f"\n  Epoch Details:")
            for epoch_result in result["epoch_results"]:
                epoch_num = epoch_result["epoch"]
                success = "✅" if epoch_result.get("success", False) else "❌"
                sharpe = epoch_result.get("sharpe_ratio", 0.0)
                dates = f"{epoch_result.get('start_date', 'N/A')[:10]} → {epoch_result.get('end_date', 'N/A')[:10]}"

                if epoch_result.get("success", False):
                    print(f"    {success} Epoch {epoch_num}: {dates} | Sharpe: {sharpe:.2f}")
                else:
                    reason = epoch_result.get("reason", "unknown")
                    print(f"    {success} Epoch {epoch_num}: {dates} | Skipped: {reason}")

    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
