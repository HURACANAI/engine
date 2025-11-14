#!/usr/bin/env python3
"""
Comprehensive Single-Coin Training with Live Monitoring

This script:
1. Downloads/loads data from Dropbox (or downloads if not available)
2. Trains the model on the coin
3. Exports model to Dropbox
4. Streams progress to web dashboard at http://localhost:5055/

Usage:
    python scripts/train_single_coin_with_monitoring.py --symbol BTC/USDT
    python scripts/train_single_coin_with_monitoring.py --symbol SOL/USDT --days 180
"""

import argparse
import json
import os
import sys
import time
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import structlog
from src.cloud.training.config.settings import EngineSettings
from src.cloud.training.datasets.data_loader import CandleDataLoader, CandleQuery
from src.cloud.training.datasets.quality_checks import DataQualitySuite
from src.cloud.training.integrations.dropbox_sync import DropboxSync
from src.cloud.training.services.exchange import ExchangeClient
from src.cloud.training.pipelines.enhanced_rl_pipeline import EnhancedRLPipeline

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)

logger = structlog.get_logger(__name__)

# Global progress state for web dashboard
progress_state = {
    "stage": "initializing",
    "progress": 0,
    "message": "Starting...",
    "details": {},
    "timestamp": datetime.now(timezone.utc).isoformat(),
}

progress_lock = threading.Lock()


def update_progress(stage: str, progress: int, message: str, details: Optional[Dict[str, Any]] = None):
    """Update progress state for web dashboard."""
    global progress_state
    with progress_lock:
        progress_state.update({
            "stage": stage,
            "progress": progress,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


def save_progress_to_file():
    """Save progress to JSON file for web dashboard to read."""
    progress_file = project_root / "training_progress.json"
    with progress_lock:
        with open(progress_file, "w") as f:
            json.dump(progress_state, f, indent=2)


def train_single_coin(
    symbol: str,
    days: int = 180,
    timeframe: str = "1h",
    dropbox_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Train a single coin with full pipeline."""
    
    result = {
        "success": False,
        "symbol": symbol,
        "stages": {},
        "error": None,
    }
    
    try:
        # Stage 1: Initialize
        update_progress("initializing", 5, "Loading settings and initializing components...")
        save_progress_to_file()
        
        settings = EngineSettings.load(environment=os.getenv("HURACAN_ENV", "local"))
        result["stages"]["initialization"] = {"status": "success"}
        
        # Stage 2: Setup Dropbox
        update_progress("setup_dropbox", 10, "Connecting to Dropbox...")
        save_progress_to_file()
        
        dropbox_access_token = dropbox_token or settings.dropbox.access_token if settings.dropbox else None
        dropbox_sync = None
        
        if dropbox_access_token:
            try:
                dropbox_sync = DropboxSync(
                    access_token=dropbox_access_token,
                    app_folder="Runpodhuracan",
                    enabled=True,
                )
                print("✅ Dropbox connected")
                result["stages"]["dropbox_setup"] = {"status": "success", "connected": True}
            except Exception as e:
                print(f"⚠️  Dropbox connection failed: {e}")
                result["stages"]["dropbox_setup"] = {"status": "warning", "error": str(e)}
        else:
            print("⚠️  No Dropbox token - will save locally only")
            result["stages"]["dropbox_setup"] = {"status": "skipped"}
        
        # Stage 3: Setup Exchange
        update_progress("setup_exchange", 15, "Connecting to exchange...")
        save_progress_to_file()
        
        exchange = ExchangeClient(
            exchange_id=settings.exchange.primary,
            credentials=None,
            sandbox=False,
        )
        print(f"✅ Connected to {exchange.exchange_id}")
        result["stages"]["exchange_setup"] = {"status": "success", "exchange": exchange.exchange_id}
        
        # Stage 4: Load/Download Data
        update_progress("loading_data", 20, f"Loading data for {symbol}...")
        save_progress_to_file()
        
        quality_suite = DataQualitySuite()
        loader = CandleDataLoader(
            exchange_client=exchange,
            quality_suite=quality_suite,
            cache_dir=Path("data/candles"),
        )
        
        # Calculate date range
        end_at = datetime.now(tz=timezone.utc)
        start_at = end_at - timedelta(days=days)
        
        # Normalize symbol
        normalized_symbol = symbol.split(":")[0] if ":" in symbol else symbol
        
        # Create query
        query = CandleQuery(
            symbol=normalized_symbol,
            timeframe=timeframe,
            start_at=start_at,
            end_at=end_at,
        )
        
        update_progress("loading_data", 30, f"Downloading {days} days of {timeframe} data from exchange...")
        save_progress_to_file()
        
        # Load data (will download if not cached, or load from cache/Dropbox)
        frame = loader.load(query, use_cache=True)
        
        if frame.is_empty():
            raise ValueError(f"No data available for {symbol}")
        
        data_rows = len(frame)
        print(f"✅ Loaded {data_rows:,} candles for {symbol}")
        result["stages"]["data_loading"] = {
            "status": "success",
            "rows": data_rows,
            "start_date": start_at.isoformat(),
            "end_date": end_at.isoformat(),
        }
        
        # Stage 5: Initialize Training Pipeline
        update_progress("initializing_training", 40, "Initializing training pipeline...")
        save_progress_to_file()
        
        pipeline = EnhancedRLPipeline(
            settings=settings,
            dsn=settings.postgres.dsn if settings.postgres else None,
        )
        print("✅ Training pipeline initialized")
        result["stages"]["pipeline_init"] = {"status": "success"}
        
        # Stage 6: Train Model
        update_progress("training", 50, f"Starting training for {symbol}...", {"stage": "initializing"})
        save_progress_to_file()
        
        print(f"\n{'='*60}")
        print(f"Training {symbol} with {days} days of data")
        print(f"{'='*60}\n")
        
        # Update progress during training
        def training_progress_callback(stage: str, progress: int, message: str):
            """Callback to update progress during training"""
            update_progress("training", 50 + int(progress * 0.4), message, {"stage": stage})
            save_progress_to_file()
        
        # Start a thread to monitor training progress
        def monitor_training():
            """Monitor training and update progress"""
            stages = [
                ("loading_data", "Loading historical data..."),
                ("building_features", "Building enhanced features..."),
                ("shadow_trading", "Running shadow trading (this may take several minutes)..."),
                ("updating_agent", "Updating RL agent with experience..."),
                ("calculating_metrics", "Calculating training metrics..."),
                ("saving_model", "Saving model and artifacts..."),
            ]
            
            for i, (stage, msg) in enumerate(stages):
                time.sleep(5)  # Wait a bit between stages
                if progress_state["stage"] == "training":
                    progress_pct = 50 + int((i + 1) * 40 / len(stages))
                    update_progress("training", progress_pct, msg, {"stage": stage})
                    save_progress_to_file()
        
        monitor_thread = threading.Thread(target=monitor_training, daemon=True)
        monitor_thread.start()
        
        training_result = pipeline.train_on_symbol(
            symbol=normalized_symbol,
            exchange_client=exchange,
            lookback_days=days,
        )
        
        if not training_result.get("success", False):
            error = training_result.get("error", "Unknown training error")
            raise RuntimeError(f"Training failed: {error}")
        
        update_progress("training", 90, "Training completed successfully!")
        save_progress_to_file()
        
        print("✅ Training completed successfully!")
        result["stages"]["training"] = {
            "status": "success",
            "metrics": training_result.get("metrics", {}),
        }
        
        # Stage 7: Export to Dropbox
        if dropbox_sync and training_result.get("model_path"):
            update_progress("exporting", 95, "Exporting model to Dropbox...")
            save_progress_to_file()
            
            model_path = Path(training_result["model_path"])
            if model_path.exists():
                try:
                    # Export model to Dropbox as champion
                    symbol_clean = normalized_symbol.replace("/", "")
                    success = dropbox_sync.upload_champion_model(
                        symbol=symbol_clean,
                        model_path=model_path,
                        archive_previous=True,
                        overwrite=True,
                    )
                    
                    if success:
                        dropbox_path = f"/Runpodhuracan/models/champions/latest/{symbol_clean}.bin"
                        print(f"✅ Model exported to Dropbox: {dropbox_path}")
                        
                        # Also upload metrics if available
                        metrics = training_result.get("metrics", {})
                        if metrics:
                            from datetime import date
                            import json
                            import tempfile
                            
                            # Save metrics to temp file and upload
                            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                                json.dump(metrics, f, indent=2)
                                metrics_path = Path(f.name)
                            
                            try:
                                dropbox_sync.upload_training_artifact(
                                    symbol=symbol_clean,
                                    run_date=date.today(),
                                    artifact_path=metrics_path,
                                    artifact_type="metrics",
                                    overwrite=True,
                                )
                                metrics_path.unlink()  # Clean up temp file
                            except Exception as e:
                                print(f"⚠️  Metrics upload failed: {e}")
                        
                        result["stages"]["export"] = {
                            "status": "success",
                            "dropbox_path": dropbox_path,
                            "local_path": str(model_path),
                        }
                    else:
                        raise Exception("Upload returned False")
                except Exception as e:
                    print(f"⚠️  Dropbox export failed: {e}")
                    result["stages"]["export"] = {
                        "status": "warning",
                        "error": str(e),
                        "local_path": str(model_path),
                    }
            else:
                result["stages"]["export"] = {
                    "status": "skipped",
                    "reason": "Model path not found",
                }
        else:
            result["stages"]["export"] = {
                "status": "skipped",
                "reason": "Dropbox not configured or no model path",
            }
        
        # Final success
        update_progress("complete", 100, "Training pipeline completed successfully!")
        save_progress_to_file()
        
        result["success"] = True
        result["model_path"] = training_result.get("model_path")
        result["metrics"] = training_result.get("metrics", {})
        
        return result
        
    except Exception as e:
        logger.error("training_failed", error=str(e), exc_info=True)
        update_progress("error", 0, f"Error: {str(e)}")
        save_progress_to_file()
        
        result["success"] = False
        result["error"] = str(e)
        return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train single coin with monitoring")
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/USDT",
        help="Symbol to train (e.g., BTC/USDT, SOL/USDT)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Number of days of historical data (default: 180)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Timeframe for candles (default: 1h)",
    )
    parser.add_argument(
        "--dropbox-token",
        type=str,
        default=None,
        help="Dropbox access token (optional, uses settings if not provided)",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SINGLE-COIN TRAINING WITH LIVE MONITORING")
    print("=" * 80)
    print(f"Symbol: {args.symbol}")
    print(f"Days: {args.days}")
    print(f"Timeframe: {args.timeframe}")
    print("=" * 80)
    print()
    
    # Start dashboard automatically
    dashboard_script = project_root / "scripts" / "comprehensive_dashboard_server.py"
    dashboard_pid = None
    
    if dashboard_script.exists():
        print("🚀 Starting comprehensive dashboard...")
        try:
            import subprocess
            import sys
            
            # Check if port 5055 is already in use
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', 5055))
            sock.close()
            
            if result != 0:  # Port is free
                # Start dashboard in background
                process = subprocess.Popen(
                    [sys.executable, str(dashboard_script)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(project_root)
                )
                dashboard_pid = process.pid
                print(f"✅ Dashboard started (PID: {dashboard_pid})")
                print(f"📊 Dashboard URL: http://localhost:5055/")
                time.sleep(3)  # Give dashboard time to start
            else:
                print("✅ Dashboard already running on port 5055")
                print(f"📊 Dashboard URL: http://localhost:5055/")
        except Exception as e:
            print(f"⚠️  Could not start dashboard: {e}")
            print("   Continuing without dashboard...")
    else:
        print("⚠️  Dashboard script not found, continuing without dashboard...")
    
    print()
    
    # Start progress monitoring thread
    def progress_monitor():
        """Periodically save progress."""
        while progress_state["stage"] not in ["complete", "error"]:
            save_progress_to_file()
            time.sleep(1)
        save_progress_to_file()  # Final save
    
    monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
    monitor_thread.start()
    
    # Run training
    result = train_single_coin(
        symbol=args.symbol,
        days=args.days,
        timeframe=args.timeframe,
        dropbox_token=args.dropbox_token,
    )
    
    # Cleanup: Stop dashboard if we started it
    if dashboard_pid:
        try:
            print("\n🛑 Stopping dashboard...")
            import signal
            import os
            os.kill(dashboard_pid, signal.SIGTERM)
            time.sleep(1)
            print("✅ Dashboard stopped")
        except Exception as e:
            print(f"⚠️  Could not stop dashboard: {e}")
    
    # Print summary
    print("\n" + "=" * 80)
    if result["success"]:
        print("✅ TRAINING COMPLETE - SUCCESS")
        print("=" * 80)
        print(f"Symbol: {result['symbol']}")
        if result.get("metrics"):
            metrics = result["metrics"]
            print(f"Total Trades: {metrics.get('total_trades', 0)}")
            print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
            print(f"Total Profit: {metrics.get('total_profit_gbp', 0):.2f} GBP")
        if result.get("model_path"):
            print(f"Model: {result['model_path']}")
        if result.get("stages", {}).get("export", {}).get("dropbox_path"):
            print(f"Dropbox: {result['stages']['export']['dropbox_path']}")
    else:
        print("❌ TRAINING FAILED")
        print("=" * 80)
        print(f"Error: {result.get('error', 'Unknown error')}")
        sys.exit(1)
    
    print("=" * 80)


if __name__ == "__main__":
    main()

