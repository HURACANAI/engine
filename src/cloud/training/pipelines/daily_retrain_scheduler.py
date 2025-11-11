"""
Daily Retrain with Hybrid Scheduler

Entry point for daily retraining with hybrid scheduler support.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from .scheduler import HybridTrainingScheduler, SchedulerConfig, TrainingMode
from .work_item import TrainResult
from ..services.storage import StorageClient, create_storage_client
from ..services.telegram import TelegramService
from ..services.symbol_costs import get_symbol_costs
from ..utils.resume_ledger import ResumeLedger
from ..utils.hash_utils import compute_file_hash, write_hash_file
from src.shared.config_loader import load_config

logger = structlog.get_logger(__name__)


def train_symbol(symbol: str, cfg: Dict[str, Any]) -> TrainResult:
    """
    Train a single symbol.
    
    Args:
        symbol: Trading symbol
        cfg: Configuration dictionary with keys:
            - timeout_minutes: Timeout in minutes
            - dry_run: Dry run mode
            - storage_client: Storage client instance
            - config: Configuration dictionary (optional)
        
    Returns:
        TrainResult with training outcome
    """
    started_at = datetime.now(timezone.utc)
    timeout_minutes = cfg.get("timeout_minutes", 45)
    dry_run = cfg.get("dry_run", False)
    storage_client = cfg.get("storage_client")
    config = cfg.get("config")
    
    logger.info("job_started", symbol=symbol, timeout_minutes=timeout_minutes)
    
    try:
        # Create unique work directory
        timestamp = started_at.strftime("%Y%m%d_%H%M%SZ")
        work_dir = Path("models") / symbol / timestamp
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Also create log directory for this symbol
        log_dir = Path("logs") / datetime.now(timezone.utc).strftime("%Y%m%dZ") / symbol
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # TODO: Implement actual training
        # For now, stub training that writes dummy files
        
        if dry_run:
            # Dry run: just create directory structure
            logger.info("dry_run_mode", symbol=symbol)
            result = TrainResult(
                symbol=symbol,
                status="success",
                started_at=started_at,
                ended_at=datetime.now(timezone.utc),
                wall_minutes=0.1,
                output_path=str(work_dir),
                metrics_path=str(work_dir / "metrics.json"),
            )
            return result
        
        # Step 1: Load data and build features (stub)
        features_path = work_dir / "features.parquet"
        # TODO: Implement actual feature building
        # For now, create empty file
        features_path.touch()
        logger.info("coin_partial_saved", symbol=symbol, file="features.parquet")
        
        # Step 2: Save split indices (stub)
        split_indices_path = work_dir / "split_indices.json"
        import json
        with open(split_indices_path, 'w') as f:
            json.dump({"train": [0, 100], "test": [100, 150]}, f)
        logger.info("coin_partial_saved", symbol=symbol, file="split_indices.json")
        
        # Step 3: Training log (stub)
        training_log_path = work_dir / "training_log.json"
        with open(training_log_path, 'w') as f:
            json.dump({
                "symbol": symbol,
                "started_at": started_at.isoformat(),
                "epochs": [],
            }, f)
        logger.info("coin_partial_saved", symbol=symbol, file="training_log.json")
        
        # Step 4: Train model (stub)
        # TODO: Implement actual model training
        model_path = work_dir / "model.bin"
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump({"model": "stub", "symbol": symbol}, f)
        
        # Step 5: Compute and save hash
        model_hash = compute_file_hash(str(model_path))
        if model_hash:
            hash_path = work_dir / "sha256.txt"
            write_hash_file(str(model_path), model_hash, str(hash_path))
        
        # Step 6: Save config
        config_path = work_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "symbol": symbol,
                "model_type": "xgboost",
                "features": ["rsi", "ema", "volatility"],
                "training_date": started_at.isoformat(),
            }, f, indent=2)
        
        # Step 7: Save metrics
        metrics_path = work_dir / "metrics.json"
        metrics = {
            "symbol": symbol,
            "sharpe": 1.5,
            "hit_rate": 0.55,
            "net_pnl_pct": 2.0,
            "max_drawdown_pct": 10.0,
            "sample_size": 1000,
            "status": "ok",
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Step 8: Fetch and save costs (before training for cost-aware evaluation)
        costs_path = work_dir / "costs.json"
        costs = get_symbol_costs(symbol, config)
        costs["symbol"] = symbol
        costs["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
        with open(costs_path, 'w') as f:
            json.dump(costs, f, indent=2)
        
        # Pass costs to training for after-cost evaluation
        # TODO: Pass costs to actual training function
        
        # Step 9: Upload to storage
        if storage_client:
            # Upload model
            remote_model_path = f"models/{symbol}/{timestamp}/model.bin"
            if storage_client.put_file(str(model_path), remote_model_path):
                logger.info("upload_succeeded", symbol=symbol, file="model.bin")
            else:
                logger.error("upload_failed", symbol=symbol, file="model.bin")
            
            # Upload metrics
            remote_metrics_path = f"models/{symbol}/{timestamp}/metrics.json"
            if storage_client.put_json(metrics, remote_metrics_path):
                logger.info("upload_succeeded", symbol=symbol, file="metrics.json")
            else:
                logger.error("upload_failed", symbol=symbol, file="metrics.json")
            
            # Upload costs
            remote_costs_path = f"models/{symbol}/{timestamp}/costs.json"
            if storage_client.put_json(costs, remote_costs_path):
                logger.info("upload_succeeded", symbol=symbol, file="costs.json")
            else:
                logger.error("upload_failed", symbol=symbol, file="costs.json")
        
        ended_at = datetime.now(timezone.utc)
        wall_minutes = (ended_at - started_at).total_seconds() / 60.0
        
        result = TrainResult(
            symbol=symbol,
            status="success",
            started_at=started_at,
            ended_at=ended_at,
            wall_minutes=wall_minutes,
            output_path=str(work_dir),
            metrics_path=str(metrics_path),
        )
        
        logger.info("coin_succeeded", symbol=symbol, wall_minutes=wall_minutes)
        return result
        
    except Exception as e:
        ended_at = datetime.now(timezone.utc)
        wall_minutes = (ended_at - started_at).total_seconds() / 60.0
        
        error_msg = str(e)
        error_type = type(e).__name__
        
        logger.error("coin_failed", symbol=symbol, error=error_msg, error_type=error_type)
        
        result = TrainResult(
            symbol=symbol,
            status="failed",
            started_at=started_at,
            ended_at=ended_at,
            wall_minutes=wall_minutes,
            error=error_msg,
            error_type=error_type,
        )
        
        return result


def load_symbols(symbols_arg: str, config: Dict[str, Any]) -> List[str]:
    """Load symbols from argument.
    
    Args:
        symbols_arg: Symbols argument (e.g., "top20", "BTCUSDT,ETHUSDT", or path to CSV)
        config: Configuration dictionary
        
    Returns:
        List of symbols
    """
    if symbols_arg.startswith("top"):
        # Load top N symbols
        n = int(symbols_arg[3:])
        # TODO: Implement top N symbol loader
        # For now, return default symbols
        default_symbols = config.get("general", {}).get("symbols", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        return default_symbols[:n]
    elif symbols_arg.endswith(".csv"):
        # Load from CSV file
        with open(symbols_arg, 'r') as f:
            reader = csv.reader(f)
            symbols = [row[0] for row in reader if row]
        return symbols
    else:
        # Comma-separated list
        return [s.strip() for s in symbols_arg.split(",") if s.strip()]


def run_scheduler(
    symbols: List[str],
    config: SchedulerConfig,
) -> List[TrainResult]:
    """
    Run scheduler with symbols and config.
    
    Args:
        symbols: List of symbols to train
        config: Scheduler configuration
        
    Returns:
        List of training results
    """
    # Create resume ledger
    resume_ledger = ResumeLedger()
    
    # Create scheduler
    scheduler = HybridTrainingScheduler(
        config=scheduler_config,
        train_func=train_symbol,
        resume_ledger=resume_ledger,
        app_config=config,  # Pass app config
    )
    
    # Schedule symbols
    results = scheduler.schedule_symbols(symbols)
    
    return results


def generate_summary(
    results: List[TrainResult],
    output_dir: Path,
) -> Path:
    """Generate summary JSON.
    
    Args:
        results: List of training results
        output_dir: Output directory
        
    Returns:
        Path to summary file
    """
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    date_str = datetime.now(timezone.utc).strftime("%Y%m%dZ")
    summary_path = summary_dir / f"{date_str}" / "engine_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate statistics
    total_symbols = len(results)
    succeeded = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status == "failed")
    skipped = sum(1 for r in results if r.status == "skipped")
    
    train_times = [r.wall_minutes for r in results if r.wall_minutes > 0]
    avg_train_minutes = sum(train_times) / len(train_times) if train_times else 0.0
    median_train_minutes = sorted(train_times)[len(train_times) // 2] if train_times else 0.0
    
    total_wall_minutes = sum(r.wall_minutes for r in results)
    
    # Build by_symbol metrics
    by_symbol = []
    for result in results:
        if result.status == "success" and result.metrics_path:
            # Load metrics
            import json
            try:
                with open(result.metrics_path, 'r') as f:
                    metrics = json.load(f)
                by_symbol.append({
                    "symbol": result.symbol,
                    "metrics_path": result.metrics_path,
                    "net_bps_after_costs": metrics.get("net_pnl_pct", 0.0) * 100,  # Convert to bps
                })
            except Exception:
                pass
    
    summary = {
        "total_symbols": total_symbols,
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "avg_train_minutes": round(avg_train_minutes, 2),
        "median_train_minutes": round(median_train_minutes, 2),
        "total_wall_minutes": round(total_wall_minutes, 2),
        "by_symbol": by_symbol,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    
    with open(summary_path, 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    
    logger.info("summary_generated", path=str(summary_path))
    return summary_path


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Huracan Engine Daily Retrain")
    parser.add_argument("--mode", type=str, choices=["sequential", "parallel", "hybrid"], default="hybrid",
                       help="Training mode (sequential, parallel, hybrid)")
    parser.add_argument("--max_concurrent", type=int, default=12,
                       help="Maximum concurrent workers (default: 12 on GPU, 2 on CPU)")
    parser.add_argument("--symbols", type=str, default="top20",
                       help="Symbols to train (topN, CSV file, or comma-separated list)")
    parser.add_argument("--timeout_minutes", type=int, default=45,
                       help="Timeout per symbol in minutes (default: 45)")
    parser.add_argument("--force", action="store_true",
                       help="Force retrain even if already completed")
    parser.add_argument("--driver", type=str, choices=["dropbox", "s3"], default="dropbox",
                       help="Storage driver (default: dropbox)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Dry run mode (no actual training)")
    
    args = parser.parse_args()
    
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
    
    logger.info("engine_starting", mode=args.mode, max_concurrent=args.max_concurrent, symbols=args.symbols)
    
    # Load configuration
    config = load_config()
    
    # Load symbols
    symbols = load_symbols(args.symbols, config)
    logger.info("symbols_loaded", count=len(symbols), symbols=symbols[:10])  # Log first 10
    
    # Create storage client
    storage_client = None
    if not args.dry_run:
        try:
            if args.driver == "dropbox":
                access_token = os.getenv("DROPBOX_ACCESS_TOKEN") or config.get("dropbox", {}).get("access_token")
                if access_token:
                    storage_client = create_storage_client(
                        driver="dropbox",
                        access_token=access_token,
                        base_path=config.get("general", {}).get("dropbox_root", "/Huracan/").strip("/"),
                    )
        except Exception as e:
            logger.warning("storage_client_init_failed", error=str(e))
    
    # Create Telegram service
    telegram_service = None
    if os.getenv("TELEGRAM_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
        telegram_service = TelegramService()
    
    # Create scheduler config
    scheduler_config = SchedulerConfig(
        mode=TrainingMode(args.mode),
        max_concurrent=args.max_concurrent,
        timeout_minutes=args.timeout_minutes,
        force=args.force,
        driver=args.driver,
        dry_run=args.dry_run,
        storage_client=storage_client,
        telegram_service=telegram_service,
    )
    
    # Run scheduler
    start_time = datetime.now(timezone.utc)
    results = run_scheduler(symbols, scheduler_config)
    end_time = datetime.now(timezone.utc)
    
    # Generate summary
    summary_path = generate_summary(results, Path("."))
    
    # Print results table
    print("\n" + "=" * 80)
    print("TRAINING RESULTS")
    print("=" * 80)
    print(f"{'Symbol':<12} {'Status':<10} {'Wall Time (min)':<15} {'Net BPS':<10}")
    print("-" * 80)
    
    for result in results:
        net_bps = "N/A"
        if result.status == "success" and result.metrics_path:
            try:
                import json
                with open(result.metrics_path, 'r') as f:
                    metrics = json.load(f)
                net_bps = f"{metrics.get('net_pnl_pct', 0.0) * 100:.1f}"
            except Exception:
                pass
        
        print(f"{result.symbol:<12} {result.status:<10} {result.wall_minutes:<15.2f} {net_bps:<10}")
    
    print("=" * 80)
    print(f"Total: {len(results)} | Success: {sum(1 for r in results if r.status == 'success')} | "
          f"Failed: {sum(1 for r in results if r.status == 'failed')} | "
          f"Skipped: {sum(1 for r in results if r.status == 'skipped')}")
    print(f"Total wall time: {(end_time - start_time).total_seconds() / 60:.2f} minutes")
    print(f"Summary: {summary_path}")
    print("=" * 80 + "\n")
    
    logger.info("job_completed", total_symbols=len(results), 
                succeeded=sum(1 for r in results if r.status == "success"),
                failed=sum(1 for r in results if r.status == "failed"))
    
    # Exit with error code if any failures
    if any(r.status == "failed" for r in results):
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

