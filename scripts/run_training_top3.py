#!/usr/bin/env python3
"""
Run training pipeline on top 3 coins: BTC, ETH, SOL

This script:
1. Loads configuration
2. Initializes the per-coin training pipeline
3. Trains on BTC/USDT, ETH/USDT, SOL/USDT
4. Reports how many days of data were used
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import structlog
from src.cloud.training.config.settings import EngineSettings
from src.cloud.training.pipelines.per_coin_training_pipeline import PerCoinTrainingPipeline
from src.cloud.training.integrations.dropbox_sync import DropboxSync
from src.shared.config_loader import load_config

# Import logging for structlog
import logging
logging.basicConfig(level=logging.INFO)

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)

logger = structlog.get_logger(__name__)


def main():
    print("=" * 70)
    print("🚀 TRAINING ENGINE ON TOP 3 COINS")
    print("=" * 70)
    print()
    
    # Top 3 coins
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    print(f"📊 Symbols: {', '.join(symbols)}")
    print()
    
    # Load configuration
    print("📋 Loading configuration...")
    config = load_config()
    
    # Get lookback days from config
    lookback_days = config.get("engine", {}).get("lookback_days", 180)
    print(f"   Lookback days: {lookback_days} days")
    print(f"   Training period: ~{lookback_days} days of historical data")
    print()
    
    # Initialize Dropbox sync (if token available)
    dropbox_sync = None
    try:
        settings = EngineSettings.load(environment=os.getenv("HURACAN_ENV", "local"))
        token = settings.dropbox.access_token or os.getenv("DROPBOX_ACCESS_TOKEN")
        if token:
            dropbox_sync = DropboxSync(
                access_token=token,
                app_folder=settings.dropbox.app_folder,
                enabled=True,
            )
            print("✅ Dropbox sync enabled")
        else:
            print("⚠️  No Dropbox token, will skip uploads")
    except Exception as e:
        print(f"⚠️  Dropbox initialization failed: {e}")
        print("   Continuing without Dropbox...")
    print()
    
    # Initialize training pipeline
    print("🔧 Initializing training pipeline...")
    pipeline = PerCoinTrainingPipeline(
        config=config,
        dropbox_sync=dropbox_sync,
    )
    print("✅ Pipeline initialized")
    print()
    
    # Train each symbol
    print("=" * 70)
    print("📈 TRAINING MODELS")
    print("=" * 70)
    print()
    
    results = {}
    
    for idx, symbol in enumerate(symbols, 1):
        print(f"[{idx}/{len(symbols)}] 🎯 Training {symbol}...")
        print("-" * 70)
        
        try:
            # Train symbol
            result = pipeline.train_symbol(symbol)
            results[symbol] = result
            
            # Report results
            if result.get("success"):
                print(f"✅ {symbol} training successful")
                print(f"   Sample size: {result.get('sample_size', 'N/A')}")
                print(f"   Features: {result.get('num_features', 'N/A')}")
                print(f"   Metrics: {result.get('metrics', {})}")
                
                # Check actual days used
                if "data_info" in result:
                    data_info = result["data_info"]
                    if "days" in data_info:
                        actual_days = data_info["days"]
                        print(f"   📅 Actual days used: {actual_days} days")
                    if "start_date" in data_info and "end_date" in data_info:
                        print(f"   📅 Date range: {data_info['start_date']} to {data_info['end_date']}")
            else:
                print(f"❌ {symbol} training failed: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f"❌ {symbol} training failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results[symbol] = {"success": False, "error": str(e)}
        
        print()
    
    # Final summary
    print("=" * 70)
    print("📊 TRAINING SUMMARY")
    print("=" * 70)
    print()
    
    successful = sum(1 for r in results.values() if r.get("success"))
    failed = len(results) - successful
    
    print(f"✅ Successful: {successful}/{len(symbols)}")
    print(f"❌ Failed: {failed}/{len(symbols)}")
    print()
    
    # Report data usage
    print("📅 DATA USAGE SUMMARY")
    print("-" * 70)
    print(f"Configured lookback: {lookback_days} days")
    print()
    
    for symbol, result in results.items():
        if result.get("success") and "data_info" in result:
            data_info = result["data_info"]
            days = data_info.get("days", lookback_days)
            start_date = data_info.get("start_date", "N/A")
            end_date = data_info.get("end_date", "N/A")
            print(f"   {symbol}:")
            print(f"      Days: {days}")
            print(f"      Range: {start_date} to {end_date}")
        else:
            print(f"   {symbol}: Failed to get data info")
    
    print()
    print("=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print()
    
    if successful == len(symbols):
        print("🎉 All coins trained successfully!")
    else:
        print(f"⚠️  {failed} coin(s) failed. Check logs for details.")
    print()


if __name__ == "__main__":
    main()

