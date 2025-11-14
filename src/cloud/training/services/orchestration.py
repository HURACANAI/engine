"""Orchestration helpers integrating APScheduler and Ray."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
import time as time_module
from datetime import date, datetime, time, timezone, timedelta
from io import BytesIO
from statistics import median
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..monitoring.comprehensive_telegram_monitor import ComprehensiveTelegramMonitor
    from ..monitoring.learning_tracker import LearningTracker
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl  # type: ignore[reportMissingImports]
import ray  # type: ignore[reportMissingImports]
import structlog  # type: ignore[reportMissingImports]
from lightgbm import LGBMRegressor  # type: ignore[reportMissingImports]

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Create a dummy tqdm if not available
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(kwargs.get('total', 0))

from ..config.settings import EngineSettings
from ..datasets.data_loader import CandleDataLoader, CandleQuery
from ..datasets.quality_checks import DataQualitySuite
from ..services.artifacts import ArtifactBundle, ArtifactPublisher
from ..services.costs import CostBreakdown, CostModel
from ..services.labeling import LabelBuilder, LabelingConfig
from .exchange import ExchangeClient
from .model_registry import ModelRegistry
from .notifications import NotificationClient
from .universe import UniverseSelector
from ..memory.store import MemoryStore
from ..pipelines.rl_training_pipeline import RLTrainingPipeline
from ..pipelines.enhanced_rl_pipeline import EnhancedRLPipeline
from ..pipelines.rl_training_pipeline_v2 import RLTrainingPipelineV2
from ..pipelines.progressive_training import ProgressiveHistoricalTrainer, ProgressiveTrainingConfig
from ..models.multi_model_integration import train_multi_model_ensemble, predict_with_ensemble, replace_single_model_training
from .brain_integrated_training import BrainIntegratedTraining
from .model_selector import ModelSelector
from .data_collector import DataCollector
# FUTURE/MECHANIC - Not used in Engine (will be used when building Mechanic component)
from src.shared.contracts.mechanic import MechanicContract  # NOQA: F401
from src.shared.contracts.metrics import MetricsPayload
# FUTURE/PILOT - Not used in Engine (will be used when building Pilot component)
from src.shared.contracts.pilot import PilotContract  # NOQA: F401
from src.shared.features.recipe import FeatureRecipe


logger = structlog.get_logger(__name__)


@dataclass
class TrainingTaskResult:
    symbol: str
    costs: CostBreakdown
    metrics: Dict[str, Any]
    gate_results: Dict[str, bool]
    published: bool
    reason: str
    artifacts: Optional[ArtifactBundle]
    model_id: str
    run_id: str
    pilot_contract: Optional[PilotContract]
    mechanic_contract: Optional[MechanicContract]
    metrics_payload: Optional[MetricsPayload]
    model_params: Dict[str, Any]
    feature_metadata: Dict[str, Any]
    artifacts_path: str = ""


def _window_bounds(run_date: date, run_time: time, window_days: int) -> tuple[datetime, datetime]:
    run_dt = datetime.combine(run_date, run_time, tzinfo=timezone.utc)
    start = run_dt - timedelta(days=window_days)
    return start, run_dt


def _walk_forward_masks(
    ts: pd.Series, 
    train_days: int, 
    test_days: int,
    shuffle_windows: bool = True,
    shift_range_days: int = 0,
    forward_shift_days: int = 0,
    embargo_days: int = 2,  # Days to skip between train and test (prevents leakage)
    random_seed: Optional[int] = None
) -> List[tuple[np.ndarray, np.ndarray]]:
    """
    Create walk-forward validation masks with optional shuffling and shifting.
    
    Args:
        ts: Timestamp series
        train_days: Training window size in days
        test_days: Test window size in days
        shuffle_windows: If True, shuffle the order of validation windows
        shift_range_days: Random shift range in days (0 = no shift, >0 = random shift up to N days)
        forward_shift_days: Forward shift all windows by N days (for robustness testing)
        random_seed: Random seed for reproducibility
    
    Returns:
        List of (train_mask, test_mask) tuples
    """
    if ts.empty:
        return []
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    splits: List[tuple[np.ndarray, np.ndarray]] = []
    start = ts.min()
    end = ts.max()
    
    # Apply forward shift to start position (shifts entire validation forward)
    current = start + timedelta(days=forward_shift_days)
    
    # Collect all splits first
    while True:
        # Apply random shift if enabled
        shift_days = np.random.randint(-shift_range_days, shift_range_days + 1) if shift_range_days > 0 else 0
        shifted_current = current + timedelta(days=shift_days)
        
        # Ensure shifted start is still within valid range
        if shifted_current < start:
            shifted_current = start
        if shifted_current >= end:
            break
            
        train_start = shifted_current
        train_end = train_start + timedelta(days=train_days)
        # CRITICAL: Add embargo between train and test to prevent leakage
        test_start = train_end + timedelta(days=embargo_days)
        test_end = test_start + timedelta(days=test_days)
        
        if test_end > end:
            break
            
        train_mask = (ts >= train_start) & (ts < train_end)
        test_mask = (ts >= test_start) & (ts < test_end)
        
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            current += timedelta(days=test_days)
            if current >= end:
                break
            continue
            
        splits.append((train_mask.to_numpy(), test_mask.to_numpy()))
        current += timedelta(days=test_days)
        if current + timedelta(days=train_days + test_days) > end + timedelta(days=1):
            break
    
    # Shuffle splits if enabled (maintains temporal order within each split)
    if shuffle_windows and len(splits) > 1:
        indices = np.arange(len(splits))
        np.random.shuffle(indices)
        splits = [splits[i] for i in indices]
        logger.info(
            "walk_forward_windows_shuffled",
            total_splits=len(splits),
            shift_range_days=shift_range_days,
            forward_shift_days=forward_shift_days,
        )
    
    return splits


def _simulate_trades(
    predictions: np.ndarray,
    actual: np.ndarray,
    confidence: np.ndarray,
    timestamps: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    executed = predictions >= threshold
    if executed.sum() == 0:
        return pd.DataFrame(columns=["timestamp", "pnl_bps", "prediction_bps", "confidence", "equity_curve"])  # type: ignore[call-overload]
    trades = pd.DataFrame(  # type: ignore[call-overload]
        {
            "timestamp": timestamps[executed],
            "pnl_bps": actual[executed],
            "prediction_bps": predictions[executed],
            "confidence": confidence[executed],
        }
    )
    trades.sort_values("timestamp", inplace=True)
    trades["equity_curve"] = trades["pnl_bps"].cumsum()
    return trades


def _max_drawdown(equity: np.ndarray) -> float:
    """Calculate maximum drawdown from equity curve (in bps)."""
    if equity.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max
    max_dd = float(drawdown.min())
    # Cap drawdown at -10000 bps (-100%) - mathematically impossible to exceed
    # If we see values worse than this, it's a calculation error
    return max(-10000.0, max_dd)


def _render_equity_curve(trades: pd.DataFrame, symbol: str) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if trades.empty:
        ax.text(0.5, 0.5, "No trades executed", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{symbol} Equity Curve (No Trades)")
    else:
        ax.plot(trades["timestamp"], trades["equity_curve"], label="Equity (bps)")
        ax.set_title(f"{symbol} Equity Curve")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Cumulative PnL (bps)")
        ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def _compute_metrics(trades: pd.DataFrame, total_costs_bps: float, total_samples: int) -> Dict[str, Any]:
    if trades.empty:
        return {
            "sharpe": 0.0,
            "profit_factor": 0.0,
            "hit_rate": 0.0,
            "max_dd_bps": 0.0,
            "pnl_bps": 0.0,
            "trades_oos": 0,
            "turnover": 0.0,
            "confidence_mean": 0.0,
            "total_costs_bps": total_costs_bps,
        }
    pnl = trades["pnl_bps"].to_numpy()
    mean_pnl = float(pnl.mean())
    std_pnl = float(pnl.std(ddof=1)) if pnl.size > 1 else 0.0
    
    # Compute Sharpe ratio with proper scaling (annualized if we have enough data)
    # For daily data: Sharpe = (mean_return / std_return) * sqrt(252)
    # For trade-based: Sharpe = (mean_pnl / std_pnl) * sqrt(num_trades)
    # Use trade-based scaling for consistency
    if std_pnl > 0 and pnl.size > 1:
        sharpe = (mean_pnl / std_pnl) * np.sqrt(pnl.size)
    else:
        sharpe = 0.0
    
    # Cap Sharpe ratio to prevent overflow noise (realistic range: -10 to 10)
    # Real-world Sharpe ratios rarely exceed 3-5, so 10 is a reasonable cap
    # If Sharpe > 10, it's likely overfitting or data leakage
    sharpe = max(-10.0, min(10.0, sharpe))
    
    positives = pnl[pnl > 0].sum()
    negatives = pnl[pnl < 0].sum()
    # Cap profit factor at 20 to avoid extreme values when negatives are very small
    # Real-world PF rarely exceeds 5-10, so 20 is a reasonable cap
    profit_factor = min(float(positives / max(abs(negatives), 1e-9)), 20.0)
    hit_rate = float((pnl > 0).mean()) if pnl.size else 0.0
    
    # Compute max drawdown with proper scaling (already in bps, so no scaling needed)
    # Drawdown is already capped in _max_drawdown at -10000 bps (-100%)
    max_dd = _max_drawdown(trades["equity_curve"].to_numpy())
    # Additional safety check: ensure drawdown is negative or zero
    max_dd = min(0.0, max_dd)
    
    # Validate metrics for unrealistic values (indicates overfitting or data leakage)
    is_unrealistic = False
    validation_warnings = []
    
    if hit_rate >= 0.99:  # 99%+ hit rate is unrealistic
        is_unrealistic = True
        validation_warnings.append(f"Hit rate {hit_rate*100:.1f}% is unrealistic (likely overfitting or data leakage)")
    
    if profit_factor >= 19.0:  # PF near cap indicates no losses
        is_unrealistic = True
        validation_warnings.append(f"Profit factor {profit_factor:.2f} is unrealistic (likely overfitting or data leakage)")
    
    if sharpe >= 9.0:  # Sharpe > 9 is extremely rare
        is_unrealistic = True
        validation_warnings.append(f"Sharpe ratio {sharpe:.2f} is extremely high (likely overfitting or data leakage)")
    
    if max_dd < -10000.0:  # Drawdown worse than -100% is impossible
        is_unrealistic = True
        validation_warnings.append(f"Max drawdown {max_dd:.2f} bps is impossible (calculation error)")
    
    # Store validation warnings in metrics for later reporting
    if is_unrealistic:
        logger.warning(
            "unrealistic_metrics_detected",
            sharpe=sharpe,
            profit_factor=profit_factor,
            hit_rate=hit_rate,
            max_dd=max_dd,
            warnings=validation_warnings,
            message="Model metrics indicate overfitting or data leakage"
        )
    
    turnover = (pnl.size / max(total_samples, 1)) * 100
    return {
        "sharpe": sharpe,
        "profit_factor": profit_factor,
        "hit_rate": hit_rate,
        "max_dd_bps": max_dd,
        "pnl_bps": pnl.sum(),
        "trades_oos": int(pnl.size),
        "turnover": turnover,
        "confidence_mean": float(trades["confidence"].mean()),
        "total_costs_bps": total_costs_bps,
        "is_unrealistic": is_unrealistic,
        "validation_warnings": validation_warnings,
    }


class TrainingOrchestrator:
    """Coordinates per-coin jobs, retries, and teardown flow."""

    def __init__(
        self,
        settings: EngineSettings,
        exchange_client: ExchangeClient,
        universe_selector: UniverseSelector,
        model_registry: ModelRegistry,
        notifier: NotificationClient,
        artifact_publisher: ArtifactPublisher,
        telegram_monitor: Optional["ComprehensiveTelegramMonitor"] = None,
        learning_tracker: Optional["LearningTracker"] = None,
        dropbox_sync: Optional[Any] = None,
    ) -> None:
        self._settings = settings
        self._exchange = exchange_client
        self._universe_selector = universe_selector
        self._registry = model_registry
        self._notifier = notifier
        self._artifact_publisher = artifact_publisher
        self._telegram_monitor = telegram_monitor
        self._learning_tracker = learning_tracker
        self._dropbox_sync = dropbox_sync
        self._run_date = datetime.now(tz=timezone.utc).date()

    def run(self) -> List[TrainingTaskResult]:
        # Initialize download progress tracker
        from ..monitoring.download_progress_tracker import get_progress_tracker
        progress_tracker = get_progress_tracker()
        
        universe = self._universe_selector.select()
        rows = list(universe.iter_rows(named=True))
        logger.info("universe_selected", count=len(rows), symbols=[row["symbol"] for row in rows])
        results: List[TrainingTaskResult] = []
        
        # Process coins in batches to avoid rate limiting
        # Binance limit: 2400 requests/minute = 40 requests/second
        # Each coin needs ~216 requests for 150 days of 1m data (1000 candles per request)
        # To stay under 2400/min, we need to process fewer coins in parallel
        # Process 2 coins at a time with 5 second delay between batches
        # This gives ~432 requests per batch, well under the 2400/min limit
        batch_size = 2  # Reduced from 5 to avoid rate limits
        batch_delay = 5.0  # Increased delay between batches
        
        for batch_start in range(0, len(rows), batch_size):
            batch_end = min(batch_start + batch_size, len(rows))
            batch_rows = rows[batch_start:batch_end]
            
            logger.info(
                "processing_batch",
                batch_num=(batch_start // batch_size) + 1,
                total_batches=(len(rows) + batch_size - 1) // batch_size,
                batch_start=batch_start,
                batch_end=batch_end,
                symbols=[row["symbol"] for row in batch_rows],
            )
            
            # Submit tasks for this batch
            batch_tasks = [
                self._submit_task(
                    row=row,
                    history=self._registry.fetch_recent_metrics(row["symbol"], limit=10),
                )
                for row in batch_rows
            ]
            
            # Handle tasks individually to allow graceful failure handling
            # Add small delay between tasks within batch to avoid simultaneous API calls
            for i, task in enumerate(batch_tasks):
                symbol = batch_rows[i]["symbol"] if i < len(batch_rows) else "unknown"
                
                # Add delay between tasks in same batch to stagger API calls
                if i > 0:
                    delay_seconds = 2.0  # 2 second delay between tasks in same batch
                    logger.info(
                        "delaying_task_start",
                        symbol=symbol,
                        delay_seconds=delay_seconds,
                        reason="Staggering API calls to avoid rate limits",
                    )
                    time_module.sleep(delay_seconds)
                
                logger.info(
                    "starting_training_task",
                    symbol=symbol,
                    batch_num=(batch_start // batch_size) + 1,
                    task_num=i + 1,
                    total_in_batch=len(batch_tasks),
                )
                
                # Notify Telegram about training start
                if self._telegram_monitor:
                    self._telegram_monitor.notify_training_progress(
                        symbol=symbol,
                        batch_num=(batch_start // batch_size) + 1,
                        total_batches=(len(rows) + batch_size - 1) // batch_size,
                        task_num=i + 1,
                        total_tasks=len(batch_tasks),
                        status="started",
                    )
                
                try:
                    # Add timeout to prevent indefinite hanging (60 minutes per coin for 365 days of data)
                    # Note: This will raise ray.exceptions.GetTimeoutError if timeout is exceeded
                    logger.info(
                        "waiting_for_training_task",
                        symbol=symbol,
                        timeout_seconds=36000,  # 10 hours for 365 days of data with larger splits
                        message="Task may take time downloading historical data and training on 365 days",
                    )
                    result = ray.get(task, timeout=36000.0)  # 10 hours timeout for 365 days of data
                    logger.info(
                        "training_task_complete",
                        symbol=symbol,
                        published=result.published,
                        reason=result.reason,
                    )
                    
                    # Track download completion
                    rows_downloaded = result.metrics.get("rows_downloaded", 0) if result.metrics else 0
                    if result.reason == "insufficient_data":
                        progress_tracker.fail_symbol_download(
                            symbol=symbol,
                            error=result.reason,
                        )
                    else:
                        progress_tracker.complete_symbol_download(
                            symbol=symbol,
                            rows_downloaded=rows_downloaded,
                        )
                    
                    # Notify Telegram about training completion
                    if self._telegram_monitor:
                        self._telegram_monitor.notify_training_progress(
                            symbol=symbol,
                            batch_num=(batch_start // batch_size) + 1,
                            total_batches=(len(rows) + batch_size - 1) // batch_size,
                            task_num=i + 1,
                            total_tasks=len(batch_tasks),
                            status="completed",
                            details={
                                "published": result.published,
                                "reason": result.reason,
                            },
                        )
                    
                    results.append(result)
                    self._finalize_result(result)
                    
                    # Notify Telegram about validation failures (from main process)
                    if self._telegram_monitor and not result.published and "validation" in result.reason.lower():
                        self._telegram_monitor.notify_validation_failure(
                            validation_type="Model Validation",
                            reason=result.reason,
                            symbol=result.symbol,
                            details=result.metrics,
                        )
                except ray.exceptions.GetTimeoutError:
                    logger.error(
                        "training_task_timeout",
                        symbol=symbol,
                        timeout_seconds=36000,  # 10 hours for 365 days of data with larger splits
                        message="Task exceeded 10 hour timeout - likely stuck on API call or data download",
                    )
                    
                    # Notify Telegram about timeout
                    if self._telegram_monitor:
                        self._telegram_monitor.notify_training_progress(
                            symbol=symbol,
                            batch_num=(batch_start // batch_size) + 1,
                            total_batches=(len(rows) + batch_size - 1) // batch_size,
                            task_num=i + 1,
                            total_tasks=len(batch_tasks),
                            status="failed",
                            details={
                                "error": "Task exceeded 10 hour timeout - likely stuck on API call or data download",
                            },
                        )
                    
                    # Create a timeout result
                    empty_cost = CostBreakdown(fee_bps=0.0, spread_bps=0.0, slippage_bps=0.0)
                    timeout_result = TrainingTaskResult(
                        symbol=symbol,
                        costs=empty_cost,
                        metrics={},
                        gate_results={},
                        published=False,
                        reason="training_timeout: Task exceeded 10 hour timeout",
                        artifacts=None,
                        model_id=str(uuid4()),
                        run_id=f"{symbol}-{self._run_date:%Y%m%d}",
                        pilot_contract=None,
                        mechanic_contract=None,
                        metrics_payload=None,
                        model_params={},
                        feature_metadata={},
                    )
                    results.append(timeout_result)
                except Exception as e:
                    # Handle individual task failures gracefully
                    symbol = batch_rows[i]["symbol"] if i < len(batch_rows) else "unknown"
                    error_msg = str(e)
                    
                    # Check if it's a rate limit issue
                    if "429" in error_msg or "Too Many Requests" in error_msg or "DDoSProtection" in error_msg:
                        logger.warning(
                            "rate_limit_hit",
                            symbol=symbol,
                            error=error_msg,
                            action="skipping_coin_will_retry_later",
                        )
                        # Wait a bit before continuing
                        time_module.sleep(5)
                    
                    # Check if it's a data quality issue
                    if "Coverage" in error_msg and "below threshold" in error_msg:
                        logger.warning(
                            "data_quality_insufficient",
                            symbol=symbol,
                            error=error_msg,
                            action="skipping_coin",
                        )
                        # Create a result indicating the coin was skipped
                        empty_cost = CostBreakdown(fee_bps=0.0, spread_bps=0.0, slippage_bps=0.0)
                        skipped_result = TrainingTaskResult(
                            symbol=symbol,
                            costs=empty_cost,
                            metrics={},
                            gate_results={},
                            published=False,
                            reason=f"data_quality_insufficient: {error_msg}",
                            artifacts=None,
                            model_id=str(uuid4()),
                            run_id=f"{symbol}-{self._run_date:%Y%m%d}",
                            pilot_contract=None,
                            mechanic_contract=None,
                            metrics_payload=None,
                            model_params={},
                            feature_metadata={},
                        )
                        results.append(skipped_result)
                    else:
                        # For other errors, log and continue
                        logger.error(
                            "training_task_failed",
                            symbol=symbol,
                            error=error_msg,
                            action="skipping_coin",
                        )
                        # Create a result indicating the coin failed
                        empty_cost = CostBreakdown(fee_bps=0.0, spread_bps=0.0, slippage_bps=0.0)
                        failed_result = TrainingTaskResult(
                            symbol=symbol,
                            costs=empty_cost,
                            metrics={},
                            gate_results={},
                            published=False,
                            reason=f"training_failed: {error_msg}",
                            artifacts=None,
                            model_id=str(uuid4()),
                            run_id=f"{symbol}-{self._run_date:%Y%m%d}",
                            pilot_contract=None,
                            mechanic_contract=None,
                            metrics_payload=None,
                            model_params={},
                            feature_metadata={},
                        )
                        results.append(failed_result)
            
            # Delay between batches to avoid rate limiting
            if batch_end < len(rows):
                logger.info("batch_complete_waiting", delay_seconds=batch_delay)
                time_module.sleep(batch_delay)
        
        self._notifier.send_summary(results, run_date=self._run_date)
        return results

    def _submit_task(self, row: Dict[str, Any], history: List[Dict[str, Any]]) -> "ray.ObjectRef[TrainingTaskResult]":
        symbol = row["symbol"]
        dsn = self._settings.postgres.dsn if self._settings.postgres else None
        return _train_symbol.remote(  # type: ignore[attr-defined]
            symbol,
            self._settings.model_dump(mode="python"),
            self._exchange.exchange_id,
            row,
            history,
            self._run_date.isoformat(),
            self._settings.mode,
            dsn,
        )

    def _finalize_result(self, result: TrainingTaskResult) -> None:
        logger.info(
            "training_result",
            symbol=result.symbol,
            published=result.published,
            reason=result.reason,
            metrics=result.metrics,
        )
        s3_uri = ""
        if result.published and result.artifacts:
            print(f"☁️  [{result.symbol}] Publishing artifacts to S3/Dropbox...")
            try:
                s3_uri = self._artifact_publisher.publish(self._run_date, result.symbol, result.artifacts)
                result.artifacts_path = s3_uri
                print(f"✅ [{result.symbol}] Artifacts published: {s3_uri}")
                print(f"☁️  [{result.symbol}] Model uploaded to Dropbox: {len(result.artifacts.files)} files")
                logger.info("artifacts_published", symbol=result.symbol, s3_uri=s3_uri, file_count=len(result.artifacts.files))
            except Exception as e:
                # Handle missing credentials gracefully - log warning, not error
                error_str = str(e)
                if "credentials" in error_str.lower() or "NoCredentialsError" in str(type(e).__name__):
                    print(f"⚠️  [{result.symbol}] S3 credentials not configured - artifacts saved locally only")
                    logger.warning("artifact_publish_skipped_no_credentials", symbol=result.symbol, error=error_str)
                else:
                    print(f"❌ [{result.symbol}] Failed to publish artifacts: {e}")
                    logger.error("artifact_publish_failed", symbol=result.symbol, error=error_str, error_type=type(e).__name__)
        
        # Also export to Dropbox in organized coin structure if available
        # This extracts artifacts from the bundle and exports them properly
        if result.published and result.artifacts and hasattr(self, '_dropbox_sync') and self._dropbox_sync:
            try:
                from pathlib import Path
                import tempfile
                
                # Extract artifacts to temp files and export
                additional_files = {}
                for name, content in result.artifacts.files.items():
                    # Create temp file
                    temp_file = Path(tempfile.gettempdir()) / f"{result.symbol}_{name}"
                    temp_file.write_bytes(content)
                    additional_files[name] = temp_file
                
                # Export using organized structure
                export_results = self._dropbox_sync.export_coin_results(
                    symbol=result.symbol,
                    run_date=self._run_date,
                    additional_files=additional_files,
                )
                
                # Clean up temp files
                for temp_file in additional_files.values():
                    if temp_file.exists():
                        temp_file.unlink()
                
                print(f"📦 [{result.symbol}] Results exported to Dropbox: {sum(1 for v in export_results.values() if v)} files")
                logger.info(
                    "coin_results_exported",
                    symbol=result.symbol,
                    files_uploaded=sum(1 for v in export_results.values() if v),
                )
            except Exception as e:
                logger.warning("dropbox_export_failed", symbol=result.symbol, error=str(e))
        
        # Registry operations - wrap in try/except so they don't fail the whole pipeline
        kind = "baseline" if result.published else "candidate"
        try:
            print(f"💾 [{result.symbol}] Registering model in database...")
            self._registry.upsert_model(
                model_id=result.model_id,
                symbol=result.symbol,
                kind=kind,
                created_at=datetime.now(tz=timezone.utc),
                s3_path=s3_uri,
                params=result.model_params,
                features=result.feature_metadata,
                notes=result.reason,
            )
            print(f"✅ [{result.symbol}] Model registered in database")
            logger.info("model_registered", symbol=result.symbol, model_id=result.model_id, kind=kind)
        except Exception as e:
            print(f"⚠️  [{result.symbol}] Failed to register model in database: {e}")
            logger.error("model_registration_failed", symbol=result.symbol, error=str(e), error_type=type(e).__name__)
            # Continue - don't fail the whole pipeline
        
        if result.metrics_payload:
            try:
                payload = {
                    "sharpe": result.metrics_payload.sharpe,
                    "profit_factor": result.metrics_payload.profit_factor,
                    "hit_rate": result.metrics_payload.hit_rate_pct / 100.0,
                    "max_dd_bps": result.metrics_payload.max_drawdown_bps,
                    "pnl_bps": result.metrics_payload.pnl_bps,
                    "trades_oos": result.metrics_payload.trades,
                    "turnover": result.metrics_payload.turnover_pct,
                    "fee_bps": result.metrics_payload.costs.get("fee_bps", 0.0),
                    "spread_bps": result.metrics_payload.costs.get("spread_bps", 0.0),
                    "slippage_bps": result.metrics_payload.costs.get("slippage_bps", 0.0),
                    "total_costs_bps": result.metrics_payload.costs.get("total_costs_bps", 0.0),
                    "validation_window": result.metrics_payload.validation_window,
                }
                self._registry.upsert_metrics(model_id=result.model_id, metrics=payload)
                print(f"✅ [{result.symbol}] Metrics saved to database")
                logger.info("metrics_saved", symbol=result.symbol, model_id=result.model_id)
            except Exception as e:
                print(f"⚠️  [{result.symbol}] Failed to save metrics to database: {e}")
                logger.error("metrics_save_failed", symbol=result.symbol, error=str(e), error_type=type(e).__name__)
                # Continue - don't fail the whole pipeline
        
        try:
            self._registry.log_publish(
                model_id=result.model_id,
                symbol=result.symbol,
                published=result.published,
                reason=result.reason,
                at=datetime.now(tz=timezone.utc),
            )
            print(f"✅ [{result.symbol}] Publish log saved to database")
            logger.info("publish_log_saved", symbol=result.symbol, model_id=result.model_id, published=result.published)
        except Exception as e:
            print(f"⚠️  [{result.symbol}] Failed to save publish log to database: {e}")
            logger.error("publish_log_save_failed", symbol=result.symbol, error=str(e), error_type=type(e).__name__)
            # Continue - don't fail the whole pipeline
        if result.published:
            self._notifier.send_success(result, run_date=self._run_date)
        else:
            self._notifier.send_reject(result, run_date=self._run_date)


def _create_dropbox_upload_callback(
    access_token: Optional[str],
    app_folder: str = "Runpodhuracan",
) -> Optional[Callable[[Path], None]]:
    """Create a callback function that immediately uploads files to Dropbox.
    
    Args:
        access_token: Dropbox access token (if None, returns None - no upload)
        app_folder: Dropbox app folder name
        
    Returns:
        Callback function that takes a Path and uploads it to Dropbox, or None if no token
    """
    if not access_token:
        return None
    
    try:
        import dropbox  # type: ignore[reportMissingImports]
        
        # Create Dropbox client
        dbx = dropbox.Dropbox(access_token)
        
        def upload_callback(cache_path: Path) -> None:
            """Immediately upload coin data file to Dropbox shared location."""
            try:
                # Get relative path from data/candles/
                cache_dir = Path("data/candles")
                if str(cache_path).startswith(str(cache_dir)):
                    rel_path = cache_path.relative_to(cache_dir)
                else:
                    # Fallback: try to extract symbol from filename
                    rel_path = Path(cache_path.name)
                
                # Upload to shared location: /Runpodhuracan/data/candles/
                remote_path = f"/{app_folder}/data/candles/{rel_path.as_posix()}"
                
                # Read file and upload
                with open(cache_path, "rb") as f:
                    file_data = f.read()
                    dbx.files_upload(
                        file_data,
                        remote_path,
                        mode=dropbox.files.WriteMode.overwrite,
                    )
                
                logger.info(
                    "coin_data_uploaded_immediately",
                    symbol=rel_path.stem if rel_path.stem else "unknown",
                    local_path=str(cache_path),
                    remote_path=remote_path,
                    size_bytes=len(file_data),
                    message="Coin data immediately uploaded to Dropbox",
                )
            except Exception as e:
                # Non-fatal - don't break data loading
                logger.warning(
                    "immediate_dropbox_upload_failed",
                    cache_path=str(cache_path),
                    error=str(e),
                    message="Failed to upload to Dropbox immediately, will be synced by background sync",
                )
        
        return upload_callback
    except ImportError:
        logger.warning("dropbox_not_available_for_immediate_upload")
        return None
    except Exception as e:
        logger.warning("dropbox_callback_creation_failed", error=str(e))
        return None


@ray.remote  # type: ignore[misc]
def _train_symbol(
    symbol: str,
    raw_settings: Dict[str, Any],
    exchange_id: str,
    universe_row: Dict[str, Any],
    history: List[Dict[str, Any]],
    run_date_str: str,
    mode: str,
    dsn: Optional[str] = None,
) -> TrainingTaskResult:
    """Train a symbol with comprehensive logging."""
    
    def _print_status(message: str, sym: str = "", level: str = "INFO") -> None:
        """Print human-readable status message."""
        prefix = {
            "INFO": "✅",
            "WARN": "⚠️ ",
            "ERROR": "❌",
            "PROGRESS": "🔄",
            "SUCCESS": "✨",
        }.get(level, "ℹ️ ")
        
        symbol_str = f"[{sym}] " if sym else ""
        print(f"{prefix} {symbol_str}{message}")
    
    logger.info(
        "training_task_started",
        symbol=symbol,
        exchange_id=exchange_id,
        message="Ray task started - beginning data download",
    )
    print(f"\n{'='*80}")
    print(f"🚀 [{symbol}] STARTING TRAINING PIPELINE")
    print(f"{'='*80}")
    print(f"  Symbol:           {symbol}")
    print(f"  Exchange:         {exchange_id}")
    print(f"  Run Date:         {run_date_str}")
    print(f"  Mode:             {mode}")
    print(f"{'='*80}\n")
    
    _print_status(f"Starting training for {symbol}", sym=symbol, level="INFO")
    _print_status(f"Exchange: {exchange_id}", sym=symbol, level="INFO")
    settings = EngineSettings.model_validate(raw_settings)
    
    # DEBUG: Log actual settings values to verify they're correct
    logger.info(
        "settings_loaded_in_ray_task",
        symbol=symbol,
        window_days=settings.training.window_days,
        train_days=settings.training.walk_forward.train_days,
        test_days=settings.training.walk_forward.test_days,
        ensemble_weights=settings.training.advanced.ensemble_weights if settings.training.advanced else None,
        use_fixed_weights=settings.training.advanced.use_fixed_ensemble_weights if settings.training.advanced else None,
        edge_threshold_override=settings.training.advanced.edge_threshold_override_bps if settings.training.advanced else None,
        message="Settings loaded in Ray remote task - verify these match config/base.yaml",
    )
    print(f"📋 [{symbol}] Settings loaded:")
    print(f"   window_days: {settings.training.window_days}")
    print(f"   train_days: {settings.training.walk_forward.train_days}")
    print(f"   test_days: {settings.training.walk_forward.test_days}")
    if settings.training.advanced:
        print(f"   ensemble_weights: {settings.training.advanced.ensemble_weights}")
        print(f"   use_fixed_weights: {settings.training.advanced.use_fixed_ensemble_weights}")
        print(f"   edge_threshold_override: {settings.training.advanced.edge_threshold_override_bps}")
    
    run_date = date.fromisoformat(run_date_str)
    credentials = settings.exchange.credentials.get(exchange_id, {})
    exchange = ExchangeClient(exchange_id, credentials=credentials, sandbox=settings.exchange.sandbox)
    quality_suite = DataQualitySuite()
    
    # CRITICAL: Define advanced_config early so it's available throughout the function
    # Get advanced config with safe fallback
    try:
        advanced_config = settings.training.advanced
        use_ensemble = advanced_config.use_multi_model_ensemble if advanced_config else False
        ensemble_techniques = advanced_config.ensemble_techniques if advanced_config else ["lightgbm"]
        ensemble_method = advanced_config.ensemble_method if advanced_config else "weighted_voting"
    except (AttributeError, KeyError) as e:
        logger.warning("advanced_config_not_available", symbol=symbol, error=str(e), message="Using default single model training")
        advanced_config = None
        use_ensemble = False
        ensemble_techniques = ["lightgbm"]
        ensemble_method = "single_model"
    
    # Multi-exchange fallback support
    # Get credentials for all exchanges for fallback
    all_exchange_credentials = settings.exchange.credentials or {}
    # Only use exchanges that are actually supported by ccxt
    # Removed "coinbasepro" - use "coinbase" instead if needed
    fallback_exchanges = ["kraken", "okx", "bybit"]  # Major exchanges with good API limits
    
    # Create immediate Dropbox upload callback if Dropbox is enabled
    dropbox_callback = None
    if settings.dropbox.enabled and settings.dropbox.access_token:
        dropbox_callback = _create_dropbox_upload_callback(
            access_token=settings.dropbox.access_token,
            app_folder=settings.dropbox.app_folder,
        )
        if dropbox_callback:
            logger.info("immediate_dropbox_upload_enabled", symbol=symbol)
    
    loader = CandleDataLoader(
        exchange_client=exchange,
        quality_suite=quality_suite,
        fallback_exchanges=fallback_exchanges,
        exchange_credentials=all_exchange_credentials,
        on_data_saved=dropbox_callback,  # Immediate upload callback
    )
    start_at, end_at = _window_bounds(run_date, settings.scheduler.daily_run_time_utc, settings.training.window_days)
    query = CandleQuery(symbol=symbol, start_at=start_at, end_at=end_at)
    
    # Calculate actual days requested
    actual_days = (end_at - start_at).total_seconds() / 86400.0
    
    logger.info(
        "downloading_historical_data",
        symbol=symbol,
        start_at=start_at.isoformat(),
        end_at=end_at.isoformat(),
        window_days=settings.training.window_days,
        actual_days=actual_days,
        message=f"Requesting {actual_days:.1f} days of data (window_days={settings.training.window_days})",
    )
    print(f"📥 [{symbol}] Requesting {actual_days:.1f} days of data (from {start_at.date()} to {end_at.date()})")
    raw_frame = loader.load(query)
    
    # Verify raw data has valid close values
    if "close" in raw_frame.columns:
        raw_close_stats = raw_frame.select([
            pl.col("close").min().alias("close_min"),
            pl.col("close").max().alias("close_max"),
            pl.col("close").null_count().alias("close_nan_count"),
            (pl.col("close") == 0).sum().alias("close_zero_count"),
            pl.col("close").count().alias("close_total_count"),
        ]).to_dicts()[0]
        
        logger.info(
            "raw_data_close_stats",
            symbol=symbol,
            close_min=raw_close_stats.get("close_min"),
            close_max=raw_close_stats.get("close_max"),
            close_nan_count=raw_close_stats.get("close_nan_count"),
            close_zero_count=raw_close_stats.get("close_zero_count"),
            close_total_count=raw_close_stats.get("close_total_count"),
            message="Raw data close column statistics",
        )
        
        if raw_close_stats.get("close_zero_count", 0) == raw_close_stats.get("close_total_count", 0):
            logger.error(
                "raw_data_close_all_zeros",
                symbol=symbol,
                message="Raw data close column has ALL zero values - data download issue",
            )
            empty_cost = CostBreakdown(fee_bps=0.0, spread_bps=0.0, slippage_bps=0.0)
            return TrainingTaskResult(
                symbol=symbol,
                costs=empty_cost,
                metrics={},
                gate_results={},
                published=False,
                reason="raw_data_close_all_zeros",
                artifacts=None,
                model_id=str(uuid4()),
                run_id=f"{symbol}-{run_date:%Y%m%d}",
                pilot_contract=None,
                mechanic_contract=None,
                metrics_payload=None,
                model_params={},
                feature_metadata={},
            )
    else:
        logger.error(
            "raw_data_close_missing",
            symbol=symbol,
            available_columns=list(raw_frame.columns),
            message="Close column missing from raw data",
        )
        empty_cost = CostBreakdown(fee_bps=0.0, spread_bps=0.0, slippage_bps=0.0)
        return TrainingTaskResult(
            symbol=symbol,
            costs=empty_cost,
            metrics={},
            gate_results={},
            published=False,
            reason="raw_data_close_missing",
            artifacts=None,
            model_id=str(uuid4()),
            run_id=f"{symbol}-{run_date:%Y%m%d}",
            pilot_contract=None,
            mechanic_contract=None,
            metrics_payload=None,
            model_params={},
            feature_metadata={},
        )
    
    logger.info(
        "data_download_complete",
        symbol=symbol,
        rows=len(raw_frame),
        message="Historical data downloaded, proceeding with training",
    )
    if raw_frame.is_empty():
        empty_cost = CostBreakdown(fee_bps=0.0, spread_bps=0.0, slippage_bps=0.0)
        return TrainingTaskResult(
            symbol=symbol,
            costs=empty_cost,
            metrics={},
            gate_results={},
            published=False,
            reason="insufficient_data",
            artifacts=None,
            model_id=str(uuid4()),
            run_id=f"{symbol}-{run_date:%Y%m%d}",
            pilot_contract=None,
            mechanic_contract=None,
            metrics_payload=None,
            model_params={},
            feature_metadata={},
        )

    _print_status("Building features (75+ technical indicators)...", sym=symbol, level="PROGRESS")
    recipe = FeatureRecipe()
    feature_frame = recipe.build(raw_frame)
    _print_status(f"Feature engineering complete: {len(feature_frame.columns)} features created", sym=symbol, level="SUCCESS")
    
    # Verify close column is preserved after feature engineering
    if "close" in feature_frame.columns:
        feature_close_stats = feature_frame.select([
            pl.col("close").min().alias("close_min"),
            pl.col("close").max().alias("close_max"),
            pl.col("close").null_count().alias("close_nan_count"),
            (pl.col("close") == 0).sum().alias("close_zero_count"),
            pl.col("close").count().alias("close_total_count"),
        ]).to_dicts()[0]
        
        logger.info(
            "feature_frame_close_stats",
            symbol=symbol,
            close_min=feature_close_stats.get("close_min"),
            close_max=feature_close_stats.get("close_max"),
            close_nan_count=feature_close_stats.get("close_nan_count"),
            close_zero_count=feature_close_stats.get("close_zero_count"),
            close_total_count=feature_close_stats.get("close_total_count"),
            message="Feature frame close column statistics after FeatureRecipe.build()",
        )
        
        # Compare with raw data
        if (feature_close_stats.get("close_min") != raw_close_stats.get("close_min") or
            feature_close_stats.get("close_max") != raw_close_stats.get("close_max")):
            logger.warning(
                "close_column_changed_during_feature_engineering",
                symbol=symbol,
                raw_min=raw_close_stats.get("close_min"),
                raw_max=raw_close_stats.get("close_max"),
                feature_min=feature_close_stats.get("close_min"),
                feature_max=feature_close_stats.get("close_max"),
                message="Close column values changed during feature engineering",
            )
    else:
        logger.error(
            "close_column_lost_in_feature_engineering",
            symbol=symbol,
            available_columns=sorted(feature_frame.columns),
            message="Close column lost during feature engineering",
        )
    
    logger.info(
        "feature_engineering_complete",
        symbol=symbol,
        rows_after_features=len(feature_frame),
        columns=len(feature_frame.columns),
    )
    
    volatility_series = feature_frame.select(pl.col("ret_1").abs().mean()).to_series()
    volatility_bps = float((volatility_series[0] if len(volatility_series) else 0.0) * 10_000)
    cost_model = CostModel(settings.costs)
    taker_fee = float(universe_row.get("taker_fee_bps") or settings.costs.default_fee_bps)
    spread_bps = float(universe_row.get("spread_bps") or settings.costs.default_spread_bps)
    adv_quote = float(universe_row.get("quote_volume") or 0.0)
    costs = cost_model.estimate(
        taker_fee_bps=taker_fee,
        spread_bps=spread_bps,
        volatility_bps=volatility_bps,
        adv_quote=adv_quote,
    )
    
    # Verify costs are valid
    if not isinstance(costs.total_costs_bps, (int, float)) or costs.total_costs_bps < 0:
        logger.error(
            "invalid_costs",
            symbol=symbol,
            total_costs_bps=costs.total_costs_bps,
            costs_type=type(costs.total_costs_bps),
            message="Costs calculation produced invalid value",
        )
        raise ValueError(f"Invalid costs.total_costs_bps: {costs.total_costs_bps} (type: {type(costs.total_costs_bps)})")
    
    logger.debug(
        "costs_calculated",
        symbol=symbol,
        total_costs_bps=costs.total_costs_bps,
        fee_bps=costs.fee_bps,
        spread_bps=costs.spread_bps,
        slippage_bps=costs.slippage_bps,
    )
    
    # Use V2 labeling (triple-barrier + meta-labeling) if enabled, otherwise basic labeling
    # Note: advanced_config is already defined at the beginning of the function
    if advanced_config:
        try:
            use_v2_labeling = advanced_config.use_triple_barrier or advanced_config.use_meta_labeling
        except (AttributeError, TypeError):
            use_v2_labeling = False
    else:
        use_v2_labeling = False
    
    if use_v2_labeling and advanced_config:
        logger.info(
            "using_v2_labeling",
            symbol=symbol,
            use_triple_barrier=advanced_config.use_triple_barrier,
            use_meta_labeling=advanced_config.use_meta_labeling,
        )
        
        try:
            from ...engine.labeling import TripleBarrierLabeler, MetaLabeler, ScalpLabelConfig
            from ...engine.costs import CostEstimator
            
            # Create triple-barrier labeler
            # Use scalp config but let costs remain realistic and slightly conservative.
            label_config = ScalpLabelConfig()

            # Derive per-side fee and spread overrides from aggregated costs,
            # then clamp into a sane range for a liquid symbol like SOL/USDT.
            roundtrip_fee_bps = float(costs.fee_bps)
            per_side_fee_bps = roundtrip_fee_bps / 2.0 if roundtrip_fee_bps > 0 else taker_fee
            per_side_fee_bps = float(min(max(per_side_fee_bps, 2.0), 6.0))  # 4–12 bps round-trip

            spread_override_bps = float(spread_bps)
            spread_override_bps = float(min(max(spread_override_bps, 0.5), 5.0))

            slippage_override_bps = float(costs.slippage_bps if hasattr(costs, "slippage_bps") else 2.0)
            slippage_override_bps = float(min(max(slippage_override_bps, 0.5), 6.0))

            logger.info(
                "v2_labeling_cost_overrides",
                symbol=symbol,
                fee_per_side_bps=per_side_fee_bps,
                spread_bps=spread_override_bps,
                slippage_bps=slippage_override_bps,
                total_roundtrip_bps=roundtrip_fee_bps,
            )

            cost_estimator = CostEstimator(
                taker_fee_bps=per_side_fee_bps,
                spread_bps=spread_override_bps,
                slippage_bps=slippage_override_bps,
            )
            triple_barrier_labeler = TripleBarrierLabeler(
                config=label_config,
                cost_estimator=cost_estimator,
            )
            
            # Label using triple-barrier
            labeled_trades = triple_barrier_labeler.label_dataframe(
                df=feature_frame,
                symbol=symbol,
                max_labels=None,  # Label all possible trades
            )
            
            logger.info(
                "triple_barrier_labeling_complete",
                symbol=symbol,
                total_trades_labeled=len(labeled_trades),
            )
            
            # Apply meta-labeling if enabled
            if advanced_config and advanced_config.use_meta_labeling:
                meta_labeler = MetaLabeler(
                    cost_threshold_bps=advanced_config.meta_label_cost_threshold,
                )
                labeled_trades = meta_labeler.apply(labeled_trades)
                
                logger.info(
                    "meta_labeling_complete",
                    symbol=symbol,
                    profitable_trades=sum(1 for t in labeled_trades if t.meta_label == 1),
                    total_trades=len(labeled_trades),
                )
            
            # Convert labeled trades into a dataset compatible with the training pipeline.
            if not labeled_trades:
                logger.warning(
                    "no_labeled_trades_from_v2_labeling",
                    symbol=symbol,
                    message="V2 labeling produced no trades",
                )
                dataset = pd.DataFrame()
            else:
                # Build a Pandas view of feature_frame so we can attach labels at entry indices.
                base_df = feature_frame.to_pandas()

                entry_indices = [t.entry_idx for t in labeled_trades]
                net_edges = [t.pnl_net_bps for t in labeled_trades]
                confidences = [1.0 if t.meta_label == 1 else 0.0 for t in labeled_trades]
                entry_times = [t.entry_time for t in labeled_trades]
                exit_times = [t.exit_time for t in labeled_trades]

                # Select the feature rows at entry indices.
                dataset = base_df.iloc[entry_indices].reset_index(drop=True)

                # Attach v2 labels.
                dataset["net_edge_bps"] = net_edges
                dataset["edge_confidence"] = confidences
                dataset["trade_entry_time"] = entry_times
                dataset["trade_exit_time"] = exit_times

                # Ensure we have a consistent timestamp column named 'ts' for downstream code.
                if "ts" not in dataset.columns:
                    if "timestamp" in dataset.columns:
                        dataset["ts"] = pd.to_datetime(dataset["timestamp"], utc=True)
                    else:
                        logger.warning(
                            "v2_labeling_ts_column_missing",
                            symbol=symbol,
                            available_columns=list(dataset.columns),
                            message="No 'ts' column found; creating from trade_entry_time",
                        )
                        dataset["ts"] = pd.to_datetime(dataset["trade_entry_time"], utc=True)

                logger.info(
                    "v2_labeling_dataset_built",
                    symbol=symbol,
                    total_rows=len(dataset),
                    winners=sum(1 for t in labeled_trades if t.meta_label == 1),
                    losers=sum(1 for t in labeled_trades if t.meta_label == 0),
                )
                
        except Exception as e:
            logger.warning(
                "v2_labeling_failed_falling_back",
                symbol=symbol,
                error=str(e),
                message="Falling back to basic labeling",
            )
            use_v2_labeling = False
    
    if not use_v2_labeling:
        # Use basic labeling (original method)
        logger.info("using_basic_labeling", symbol=symbol)
        
        # CRITICAL: Verify close column exists in feature_frame
        # FeatureRecipe should preserve all original columns, but let's verify
        if "close" not in feature_frame.columns:
            # Try to find it with different casing
            close_cols = [col for col in feature_frame.columns if col.lower() == "close"]
            if close_cols:
                logger.warning(
                    "close_column_case_mismatch",
                    symbol=symbol,
                    found_column=close_cols[0],
                    message=f"Found close column with different case: {close_cols[0]}, renaming to 'close'",
                )
                feature_frame = feature_frame.rename({close_cols[0]: "close"})
            else:
                logger.error(
                    "close_column_missing_from_feature_frame",
                    symbol=symbol,
                    available_columns=sorted(feature_frame.columns),
                    total_columns=len(feature_frame.columns),
                    message="Close column missing from feature frame - FeatureRecipe may have dropped it",
                )
                # Check if raw_frame had close column
                if "close" in raw_frame.columns:
                    logger.error(
                        "close_column_lost_during_feature_engineering",
                        symbol=symbol,
                        raw_frame_columns=sorted(raw_frame.columns),
                        feature_frame_columns=sorted(feature_frame.columns),
                        message="Close column was in raw_frame but lost during feature engineering",
                    )
                return TrainingTaskResult(
                    symbol=symbol,
                    costs=costs,
                    metrics={},
                    gate_results={},
                    published=False,
                    reason="close_column_missing",
                    artifacts=None,
                    model_id=str(uuid4()),
                    run_id=f"{symbol}-{run_date:%Y%m%d}",
                    pilot_contract=None,
                    mechanic_contract=None,
                    metrics_payload=None,
                    model_params={},
                    feature_metadata={},
                )
        
        # Check close column for NaN/zero values BEFORE labeling
        close_stats = feature_frame.select([
            pl.col("close").min().alias("close_min"),
            pl.col("close").max().alias("close_max"),
            pl.col("close").null_count().alias("close_nan_count"),
            (pl.col("close") == 0).sum().alias("close_zero_count"),
            pl.col("close").count().alias("close_total_count"),
        ])
        close_stats_dict = close_stats.to_dicts()[0]
        
        logger.info(
            "close_column_stats_before_labeling",
            symbol=symbol,
            close_min=close_stats_dict.get("close_min"),
            close_max=close_stats_dict.get("close_max"),
            close_nan_count=close_stats_dict.get("close_nan_count"),
            close_zero_count=close_stats_dict.get("close_zero_count"),
            close_total_count=close_stats_dict.get("close_total_count"),
        )
        
        # If close column has issues, we can't proceed with labeling
        if close_stats_dict.get("close_nan_count", 0) > 0:
            logger.error(
                "close_column_has_nan_values",
                symbol=symbol,
                nan_count=close_stats_dict.get("close_nan_count", 0),
                total_count=close_stats_dict.get("close_total_count", 0),
                message="Close column has NaN values - cannot compute labels",
            )
            return TrainingTaskResult(
                symbol=symbol,
                costs=costs,
                metrics={},
                gate_results={},
                published=False,
                reason="close_column_has_nan",
                artifacts=None,
                model_id=str(uuid4()),
                run_id=f"{symbol}-{run_date:%Y%m%d}",
                pilot_contract=None,
                mechanic_contract=None,
                metrics_payload=None,
                model_params={},
                feature_metadata={},
            )
        
        if close_stats_dict.get("close_zero_count", 0) > 0:
            logger.warning(
                "close_column_has_zero_values",
                symbol=symbol,
                zero_count=close_stats_dict.get("close_zero_count", 0),
                message="Close column has zero values - division by zero risk in labeling",
            )
        
        try:
            # Use embargo to prevent lookahead from rolling operations
            embargo_candles = getattr(settings.training.walk_forward, 'embargo_candles', 2)
            label_builder = LabelBuilder(LabelingConfig(horizon_minutes=4, embargo_candles=embargo_candles))
            labeled = label_builder.build(feature_frame, costs)
            
            # Verify net_edge_bps was created correctly in Polars
            if "net_edge_bps" not in labeled.columns:
                raise ValueError("net_edge_bps column not found after labeling")
            
            # Check net_edge_bps in Polars BEFORE converting to pandas
            net_edge_stats_polars = labeled.select([
                pl.col("net_edge_bps").null_count().alias("nan_count"),
                pl.col("net_edge_bps").count().alias("total_count"),
                pl.col("net_edge_bps").min().alias("min"),
                pl.col("net_edge_bps").max().alias("max"),
                pl.col("net_edge_bps").mean().alias("mean"),
            ])
            net_edge_stats_dict = net_edge_stats_polars.to_dicts()[0]
            
            nan_count = net_edge_stats_dict.get("nan_count", 0)
            valid_count = net_edge_stats_dict.get("total_count", 0) - nan_count
            
            logger.info(
                "net_edge_bps_stats_polars",
                symbol=symbol,
                nan_count=nan_count,
                valid_count=valid_count,
                min_value=net_edge_stats_dict.get("min"),
                max_value=net_edge_stats_dict.get("max"),
                mean_value=net_edge_stats_dict.get("mean"),
            )
            
            # If all values are NaN, this is a critical error
            if valid_count == 0:
                raise ValueError(
                    f"All {net_edge_stats_dict.get('total_count', 0)} net_edge_bps values are NaN in Polars. "
                    f"This indicates a problem with the labeling calculation."
                )
            
            # Sample a few values to verify they're not all the same
            sample_values = labeled.select(["ts", "close", "net_edge_bps"]).head(5)
            logger.debug(
                "net_edge_bps_sample_values",
                symbol=symbol,
                sample=sample_values.to_dicts(),
            )
            
        except Exception as e:
            logger.exception(
                "labeling_failed",
                symbol=symbol,
                error=str(e),
                error_type=type(e).__name__,
                message="LabelBuilder.build() failed - check close column and data quality",
            )
            return TrainingTaskResult(
                symbol=symbol,
                costs=costs,
                metrics={},
                gate_results={},
                published=False,
                reason=f"labeling_failed: {str(e)}",
                artifacts=None,
                model_id=str(uuid4()),
                run_id=f"{symbol}-{run_date:%Y%m%d}",
                pilot_contract=None,
                mechanic_contract=None,
                metrics_payload=None,
                model_params={},
                feature_metadata={},
            )
        
        logger.info(
            "basic_labeling_complete",
            symbol=symbol,
            rows_before_labeling=len(feature_frame),
            rows_after_labeling=len(labeled),
            rows_dropped=len(feature_frame) - len(labeled),
            columns_after_labeling=list(labeled.columns),
        )
        _print_status(f"Labeling complete: {len(labeled):,} labeled samples (dropped {len(feature_frame) - len(labeled)} rows)", sym=symbol, level="SUCCESS")
        
        # Convert to Pandas
        # LabelBuilder already validated that net_edge_bps has valid values
        # CRITICAL: Ensure ts column is properly converted to datetime
        # Polars datetime columns should convert correctly, but verify
        if "ts" in labeled.columns:
            # Check ts column type in Polars before conversion
            ts_dtype = labeled["ts"].dtype
            logger.debug(
                "ts_column_before_pandas_conversion",
                symbol=symbol,
                ts_dtype=str(ts_dtype),
                ts_sample=labeled.select("ts").head(3).to_dicts(),
            )
        
        dataset = labeled.to_pandas()
        
        # Verify conversion preserved the data
        if "net_edge_bps" not in dataset.columns:
            raise ValueError("net_edge_bps column missing after Polars to Pandas conversion")
        
        # CRITICAL: Verify and fix ts column if corrupted
        if "ts" in dataset.columns:
            # Check if ts is datetime type
            if not pd.api.types.is_datetime64_any_dtype(dataset["ts"]):
                logger.warning(
                    "ts_column_not_datetime_after_conversion",
                    symbol=symbol,
                    ts_dtype=str(dataset["ts"].dtype),
                    ts_sample=dataset["ts"].head(3).tolist(),
                    message="ts column is not datetime after Polars to Pandas conversion - attempting fix",
                )
                # Try to convert to datetime
                dataset["ts"] = pd.to_datetime(dataset["ts"], utc=True, errors="coerce")
            
            # Verify the dates are reasonable (not 1970)
            ts_min = dataset["ts"].min()
            ts_max = dataset["ts"].max()
            if ts_min.year == 1970 or ts_max.year == 1970:
                logger.error(
                    "ts_column_corrupted_after_conversion",
                    symbol=symbol,
                    ts_min=ts_min.isoformat(),
                    ts_max=ts_max.isoformat(),
                    message="ts column shows 1970 dates after conversion - fixing from raw_frame",
                )
                # Fix: Get ts from raw_frame and align
                # The labeled frame has fewer rows due to shift, so we need to drop the last rows from raw_ts
                if "ts" in raw_frame.columns:
                    logger.info("fixing_ts_from_raw_frame", symbol=symbol)
                    # Convert raw_frame ts to pandas first to check if it's correct
                    raw_ts_df = raw_frame.select(["ts"]).to_pandas()
                    raw_ts_min = raw_ts_df["ts"].min()
                    raw_ts_max = raw_ts_df["ts"].max()
                    
                    logger.info(
                        "raw_frame_ts_check",
                        symbol=symbol,
                        raw_ts_min=raw_ts_min.isoformat() if hasattr(raw_ts_min, 'isoformat') else str(raw_ts_min),
                        raw_ts_max=raw_ts_max.isoformat() if hasattr(raw_ts_max, 'isoformat') else str(raw_ts_max),
                        raw_ts_rows=len(raw_ts_df),
                        message="Checking raw_frame ts column",
                    )
                    
                    # Check if raw_frame ts is also corrupted
                    if raw_ts_min.year == 1970 or raw_ts_max.year == 1970:
                        logger.error(
                            "raw_frame_ts_also_corrupted",
                            symbol=symbol,
                            message="raw_frame ts column is also corrupted - cannot fix",
                        )
                    else:
                        # LabelBuilder drops the last 'horizon' rows (4 rows for horizon_minutes=4)
                        # So we need to drop the last 4 rows from raw_ts to match labeled frame
                        horizon = 4  # From LabelingConfig(horizon_minutes=4)
                        raw_ts_aligned = raw_ts_df.iloc[:-horizon].reset_index(drop=True)
                        
                        # Verify row counts match
                        if len(raw_ts_aligned) == len(dataset):
                            dataset["ts"] = raw_ts_aligned["ts"]
                            logger.info(
                                "ts_column_fixed_from_raw_frame",
                                symbol=symbol,
                                ts_min=dataset["ts"].min().isoformat(),
                                ts_max=dataset["ts"].max().isoformat(),
                                ts_range_days=(dataset["ts"].max() - dataset["ts"].min()).total_seconds() / 86400.0,
                            )
                        else:
                            logger.error(
                                "ts_fix_failed_row_count_mismatch",
                                symbol=symbol,
                                raw_ts_rows=len(raw_ts_aligned),
                                dataset_rows=len(dataset),
                                message="Cannot fix ts column - row count mismatch",
                            )
                else:
                    logger.error(
                        "raw_frame_ts_missing",
                        symbol=symbol,
                        message="Cannot fix ts column - raw_frame does not have ts column",
                    )
            else:
                logger.debug(
                    "ts_column_converted_correctly",
                    symbol=symbol,
                    ts_min=ts_min.isoformat(),
                    ts_max=ts_max.isoformat(),
                    ts_range_days=(ts_max - ts_min).total_seconds() / 86400.0,
                )
        
        # Quick sanity check - should match Polars stats
        net_edge_pandas_valid = dataset["net_edge_bps"].notna().sum()
        if net_edge_pandas_valid != valid_count:
            logger.warning(
                "pandas_conversion_value_mismatch",
                symbol=symbol,
                polars_valid_count=valid_count,
                pandas_valid_count=net_edge_pandas_valid,
                message="Value count mismatch between Polars and Pandas - investigating",
            )
        
        logger.debug(
            "pandas_conversion_complete",
            symbol=symbol,
            net_edge_valid_count=net_edge_pandas_valid,
            total_rows=len(dataset),
        )
        
        # Check if ts column exists
        if "ts" not in dataset.columns:
            # Try to find timestamp column
            ts_cols = [col for col in dataset.columns if "ts" in col.lower() or "time" in col.lower()]
            if ts_cols:
                logger.warning(
                    "ts_column_not_found_using_alternative",
                    symbol=symbol,
                    found_columns=ts_cols,
                    using_column=ts_cols[0],
                )
                dataset = dataset.rename(columns={ts_cols[0]: "ts"})
            else:
                logger.error(
                    "no_timestamp_column_found",
                    symbol=symbol,
                    available_columns=list(dataset.columns),
                    message="Cannot find timestamp column in labeled dataset",
                )
                dataset = pd.DataFrame()  # Empty dataset
    else:
        # V2 labeling path has already constructed `dataset` above.
        pass
    
    if dataset.empty:
        logger.error(
            "labeling_produced_empty_dataset",
            symbol=symbol,
            original_rows=len(raw_frame),
            feature_rows=len(feature_frame),
            labeled_rows=len(labeled) if not use_v2_labeling and 'labeled' in locals() else 0,
            message="Labeling produced empty dataset - check labeling configuration and data quality",
        )
        return TrainingTaskResult(
            symbol=symbol,
            costs=costs,
            metrics={},
            gate_results={},
            published=False,
            reason="insufficient_labeled_data",
            artifacts=None,
            model_id=str(uuid4()),
            run_id=f"{symbol}-{run_date:%Y%m%d}",
            pilot_contract=None,
            mechanic_contract=None,
            metrics_payload=None,
            model_params={},
            feature_metadata={},
        )
    
    # Ensure ts column exists and is datetime
    if "ts" in dataset.columns:
        dataset["ts"] = pd.to_datetime(dataset["ts"], utc=True)
    else:
        logger.error(
            "ts_column_missing_after_labeling",
            symbol=symbol,
            available_columns=list(dataset.columns),
            message="Timestamp column missing after labeling",
        )
        return TrainingTaskResult(
            symbol=symbol,
            costs=costs,
            metrics={},
            gate_results={},
            published=False,
            reason="insufficient_labeled_data",
            artifacts=None,
            model_id=str(uuid4()),
            run_id=f"{symbol}-{run_date:%Y%m%d}",
            pilot_contract=None,
            mechanic_contract=None,
            metrics_payload=None,
            model_params={},
            feature_metadata={},
        )
    
    # Check net_edge_bps column BEFORE replacing inf
    if "net_edge_bps" in dataset.columns:
        net_edge_before_replace = dataset["net_edge_bps"].copy()
        net_edge_nan_before = net_edge_before_replace.isna().sum()
        net_edge_inf_before = np.isinf(net_edge_before_replace).sum()
        net_edge_valid_before = net_edge_before_replace.notna().sum() - net_edge_inf_before
        
        logger.info(
            "net_edge_bps_before_cleanup",
            symbol=symbol,
            net_edge_nan_count=net_edge_nan_before,
            net_edge_inf_count=net_edge_inf_before,
            net_edge_valid_count=net_edge_valid_before,
            net_edge_min=float(net_edge_before_replace.replace([np.inf, -np.inf], np.nan).min()) if net_edge_valid_before > 0 else None,
            net_edge_max=float(net_edge_before_replace.replace([np.inf, -np.inf], np.nan).max()) if net_edge_valid_before > 0 else None,
        )
    
    # Replace inf values with NaN before dropna
    rows_before_dropna = len(dataset)
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Check net_edge_bps column AFTER replacing inf
    if "net_edge_bps" in dataset.columns:
        net_edge_stats = {
            "net_edge_nan_count": dataset["net_edge_bps"].isna().sum(),
            "net_edge_valid_count": dataset["net_edge_bps"].notna().sum(),
        }
        if net_edge_stats["net_edge_valid_count"] > 0:
            net_edge_stats["net_edge_min"] = float(dataset["net_edge_bps"].min())
            net_edge_stats["net_edge_max"] = float(dataset["net_edge_bps"].max())
        logger.info(
            "net_edge_bps_after_cleanup",
            symbol=symbol,
            **net_edge_stats,
        )
    else:
        logger.error(
            "net_edge_bps_column_missing",
            symbol=symbol,
            available_columns=list(dataset.columns),
        )
        return TrainingTaskResult(
            symbol=symbol,
            costs=costs,
            metrics={},
            gate_results={},
            published=False,
            reason="insufficient_labeled_data",
            artifacts=None,
            model_id=str(uuid4()),
            run_id=f"{symbol}-{run_date:%Y%m%d}",
            pilot_contract=None,
            mechanic_contract=None,
            metrics_payload=None,
            model_params={},
            feature_metadata={},
        )
    
    # Check ts column
    if "ts" in dataset.columns:
        ts_nan_count = dataset["ts"].isna().sum()
        logger.info(
            "ts_column_diagnostic",
            symbol=symbol,
            ts_nan_count=ts_nan_count,
            ts_valid_count=dataset["ts"].notna().sum(),
        )
    else:
        logger.error(
            "ts_column_missing_before_dropna",
            symbol=symbol,
            available_columns=list(dataset.columns),
        )
    
    # Only drop rows where critical columns are NaN
    critical_cols = ["ts", "net_edge_bps"]
    rows_with_ts_nan = dataset["ts"].isna().sum() if "ts" in dataset.columns else 0
    rows_with_net_edge_nan = dataset["net_edge_bps"].isna().sum() if "net_edge_bps" in dataset.columns else len(dataset)
    
    logger.info(
        "before_dropna_diagnostic",
        symbol=symbol,
        rows_before_dropna=rows_before_dropna,
        rows_with_ts_nan=rows_with_ts_nan,
        rows_with_net_edge_nan=rows_with_net_edge_nan,
    )
    
    dataset = dataset.dropna(subset=critical_cols)
    rows_after_dropna = len(dataset)
    
    logger.info(
        "data_cleaning_complete",
        symbol=symbol,
        rows_before_dropna=rows_before_dropna,
        rows_after_dropna=rows_after_dropna,
        rows_removed_by_dropna=rows_before_dropna - rows_after_dropna,
    )
    
    dataset.sort_values("ts", inplace=True)
    if dataset.empty:
        logger.error(
            "dataset_empty_after_processing",
            symbol=symbol,
            original_rows=len(raw_frame),
            feature_rows=len(feature_frame),
            rows_after_labeling=rows_before_dropna,
            rows_after_dropna=rows_after_dropna,
            message="Dataset is empty after all processing steps",
        )
        return TrainingTaskResult(
            symbol=symbol,
            costs=costs,
            metrics={},
            gate_results={},
            published=False,
            reason="insufficient_labeled_data",
            artifacts=None,
            model_id=str(uuid4()),
            run_id=f"{symbol}-{run_date:%Y%m%d}",
            pilot_contract=None,
            mechanic_contract=None,
            metrics_payload=None,
            model_params={},
            feature_metadata={},
        )

    excluded = {
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "net_edge_bps",
        "edge_confidence",
        "timestamp",
        "trade_entry_time",
        "trade_exit_time",
        "trade_pnl_bps",
    }
    feature_cols = [col for col in dataset.columns if col not in excluded and dataset[col].dtype != object]
    if not feature_cols:
        return TrainingTaskResult(
            symbol=symbol,
            costs=costs,
            metrics={},
            gate_results={},
            published=False,
            reason="no_features_available",
            artifacts=None,
            model_id=str(uuid4()),
            run_id=f"{symbol}-{run_date:%Y%m%d}",
            pilot_contract=None,
            mechanic_contract=None,
            metrics_payload=None,
            model_params={},
            feature_metadata={},
        )

    # Use configurable training settings
    model_config = settings.training.model_training
    
    # Get regularization settings (with defaults - increased for better generalization)
    reg_alpha = getattr(model_config, 'reg_alpha', 2.0)  # L1 regularization (increased from 0.1)
    reg_lambda = getattr(model_config, 'reg_lambda', 5.0)  # L2 regularization (increased from 1.0)
    feature_fraction = getattr(model_config, 'feature_fraction', 0.8)  # Feature dropout
    noise_injection_std = getattr(model_config, 'noise_injection_std', 0.025)  # Noise injection (increased from 0.01)
    
    # Get advanced config for ensemble weights and edge threshold
    advanced_config = settings.training.advanced
    fixed_weights = getattr(advanced_config, 'ensemble_weights', None)
    use_fixed = getattr(advanced_config, 'use_fixed_ensemble_weights', False)
    
    # Build hyperparameters dict for ensemble models
    ensemble_hyperparams = {
        'n_estimators': model_config.n_estimators,
        'learning_rate': model_config.learning_rate,
        'max_depth': model_config.max_depth,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'lambda_l1': getattr(model_config, 'lambda_l1', 3.0),
        'lambda_l2': getattr(model_config, 'lambda_l2', 3.0),
        'min_data_in_leaf': getattr(model_config, 'min_data_in_leaf', 50),
        'rf_max_depth': getattr(model_config, 'rf_max_depth', 5),
        'rf_min_samples_leaf': getattr(model_config, 'rf_min_samples_leaf', 10),
        'feature_fraction': feature_fraction,
    }
    
    hyperparams = {
        "objective": "regression",
        "learning_rate": model_config.learning_rate,
        "n_estimators": model_config.n_estimators,
        "max_depth": model_config.max_depth,
        "subsample": model_config.subsample,
        "colsample_bytree": model_config.colsample_bytree,
        "min_child_samples": model_config.min_child_samples,
        "random_state": model_config.random_state,
        "n_jobs": model_config.n_jobs,
        # Regularization to prevent overfitting
        "reg_alpha": reg_alpha,  # L1 regularization
        "reg_lambda": reg_lambda,  # L2 regularization
        "feature_fraction": feature_fraction,  # Feature dropout (similar to dropout in neural networks)
        "verbose": model_config.verbose,
    }

    ts_series = dataset["ts"]
    
    # CRITICAL: Validate timestamps are monotonic and timezone-aware
    if not ts_series.is_monotonic_increasing:
        logger.warning(
            "timestamps_not_monotonic",
            symbol=symbol,
            message="Timestamps are not strictly increasing - sorting to fix"
        )
        dataset = dataset.sort_values("ts").reset_index(drop=True)
        ts_series = dataset["ts"]
    
    # Check timezone awareness
    if hasattr(ts_series.iloc[0], 'tzinfo') and ts_series.iloc[0].tzinfo is None:
        logger.warning(
            "timestamps_not_timezone_aware",
            symbol=symbol,
            message="Timestamps are not timezone-aware - assuming UTC"
        )
    
    # Diagnostic: Check timestamp range before walk-forward splits
    ts_min = ts_series.min()
    ts_max = ts_series.max()
    ts_range_days = (ts_max - ts_min).total_seconds() / 86400.0
    
    logger.info(
        "timestamp_range_for_walk_forward",
        symbol=symbol,
        ts_min=ts_min.isoformat() if hasattr(ts_min, 'isoformat') else str(ts_min),
        ts_max=ts_max.isoformat() if hasattr(ts_max, 'isoformat') else str(ts_max),
        ts_range_days=ts_range_days,
        train_days=settings.training.walk_forward.train_days,
        test_days=settings.training.walk_forward.test_days,
        min_splits_possible=int((ts_range_days - settings.training.walk_forward.train_days) / settings.training.walk_forward.test_days) if ts_range_days > settings.training.walk_forward.train_days else 0,
        message="Timestamp range for walk-forward validation",
    )
    
    # Get validation window configuration from settings
    shuffle_windows = getattr(settings.training.walk_forward, 'shuffle_windows', True)
    shift_range_days = getattr(settings.training.walk_forward, 'shift_range_days', 0)
    forward_shift_days = getattr(settings.training.walk_forward, 'forward_shift_days', 0)
    random_seed = getattr(settings.training.walk_forward, 'random_seed', 42)
    shuffle_candles = getattr(settings.training.walk_forward, 'shuffle_candles', False)  # Test for overfitting
    
    # Get embargo_days from config (default 2 days to prevent leakage)
    embargo_days = getattr(settings.training.walk_forward, 'embargo_days', 2)
    
    splits = _walk_forward_masks(
        ts_series, 
        settings.training.walk_forward.train_days, 
        settings.training.walk_forward.test_days,
        shuffle_windows=shuffle_windows,
        shift_range_days=shift_range_days,
        forward_shift_days=forward_shift_days,
        embargo_days=embargo_days,
        random_seed=random_seed
    )
    
    # Filter splits by min_coverage (reject splits with too many gaps)
    min_coverage = getattr(settings.training.walk_forward, 'min_coverage', 0.98)
    if min_coverage > 0:
        expected_train_candles = settings.training.walk_forward.train_days * 1440  # 1-min candles per day
        expected_test_candles = settings.training.walk_forward.test_days * 1440
        original_splits = len(splits)
        filtered_splits = []
        
        for train_mask, test_mask in splits:
            train_count = train_mask.sum()
            test_count = test_mask.sum()
            train_coverage = train_count / expected_train_candles if expected_train_candles > 0 else 0
            test_coverage = test_count / expected_test_candles if expected_test_candles > 0 else 0
            
            if train_coverage >= min_coverage and test_coverage >= min_coverage:
                filtered_splits.append((train_mask, test_mask))
            else:
                logger.warning(
                    "split_rejected_low_coverage",
                    symbol=symbol,
                    train_coverage=train_coverage,
                    test_coverage=test_coverage,
                    min_coverage=min_coverage,
                    train_count=train_count,
                    test_count=test_count
                )
        
        splits = filtered_splits
        if len(splits) < original_splits:
            logger.warning(
                "splits_filtered_by_coverage",
                symbol=symbol,
                original_splits=original_splits,
                filtered_splits=len(splits),
                min_coverage=min_coverage
            )
    
    # Shuffle candle order if enabled (to test if model is overfitting to temporal patterns)
    if shuffle_candles:
        logger.warning(
            "candle_order_shuffled",
            symbol=symbol,
            message="CANDLE ORDER SHUFFLED - This tests if model is overfitting to temporal patterns. Results should degrade significantly if overfitting."
        )
        # Shuffle the dataset while preserving feature-target relationships
        # This breaks temporal order to test if model relies on time-based patterns
        shuffled_indices = np.random.permutation(len(dataset))
        dataset = dataset.iloc[shuffled_indices].reset_index(drop=True)
        # DO NOT re-sort by timestamp - keep shuffled order to break temporal patterns
        # This will cause walk-forward validation to use non-sequential data
        # If model performance degrades significantly, it's overfitting to temporal patterns
    
    # Get edge threshold (allow override for lower threshold to capture more trades)
    edge_threshold_override = getattr(advanced_config, 'edge_threshold_override_bps', None)
    # Base recommendation from cost model (in bps, round-trip)
    recommended_threshold = float(cost_model.recommended_edge_threshold(costs))

    base_cost_bps = float(getattr(costs, "total_costs_bps", 0.0))
    median_cost_bps = base_cost_bps
    p90_cost_bps = base_cost_bps * 1.5 if base_cost_bps > 0 else base_cost_bps

    # Also look at the realized net_edge_bps distribution from labels.
    if "net_edge_bps" in dataset.columns:
        net_edges_series = dataset["net_edge_bps"].replace([np.inf, -np.inf], np.nan).dropna()
        if not net_edges_series.empty:
            net_edge_median = float(net_edges_series.median())
            net_edge_p90 = float(net_edges_series.quantile(0.9))
            net_edge_max = float(net_edges_series.max())
        else:
            net_edge_median = net_edge_p90 = net_edge_max = 0.0
    else:
        net_edge_median = net_edge_p90 = net_edge_max = 0.0

    if use_v2_labeling:
        # V2 labels (triple-barrier) use realized PnL net of costs.
        # Edge threshold should therefore be based on the distribution of net_edge_bps,
        # not on a large cost-based target that may be unattainable.
        base_threshold = max(0.0, net_edge_p90)

        if edge_threshold_override is not None:
            # Treat override as an upper bound when using v2 labels.
            base_threshold = min(base_threshold, float(edge_threshold_override))

        cost_threshold = base_threshold
        logger.info(
            "edge_threshold_v2_labels_selected",
            symbol=symbol,
            final_edge_threshold=cost_threshold,
            net_edge_median=net_edge_median,
            net_edge_p90=net_edge_p90,
            net_edge_max=net_edge_max,
            edge_threshold_override=edge_threshold_override,
            total_costs_bps=base_cost_bps,
            recommended_threshold=recommended_threshold,
            median_cost_bps=median_cost_bps,
            p90_cost_bps=p90_cost_bps,
        )
    else:
        # For basic labels, keep a conservative, cost-based minimum threshold.
        min_edge_bps = max(recommended_threshold, p90_cost_bps + 2.0)

        if edge_threshold_override is not None:
            requested = float(edge_threshold_override)
            # Never allow threshold below min_edge_bps (we don't want obviously negative after costs).
            cost_threshold = max(requested, min_edge_bps)
            logger.info(
                "edge_threshold_overridden",
                symbol=symbol,
                recommended_threshold=recommended_threshold,
                override_threshold=requested,
                final_edge_threshold=cost_threshold,
                median_cost_bps=median_cost_bps,
                p90_cost_bps=p90_cost_bps,
                message="Using overridden edge threshold, clamped to remain above costs",
            )
        else:
            cost_threshold = max(recommended_threshold, min_edge_bps)
            logger.info(
                "edge_threshold_selected",
                symbol=symbol,
                recommended_threshold=recommended_threshold,
                final_edge_threshold=cost_threshold,
                median_cost_bps=median_cost_bps,
                p90_cost_bps=p90_cost_bps,
            )
    oos_trades: List[pd.DataFrame] = []
    
    # Initialize Brain Library integration (if database is available)
    brain_training = None
    try:
        if dsn:
            from ..brain.brain_library import BrainLibrary
            brain_library = BrainLibrary(dsn=dsn, use_pool=True)
            brain_training = BrainIntegratedTraining(brain_library, settings)
            logger.info("brain_library_integration_enabled", symbol=symbol)
    except Exception as e:
        logger.warning("brain_library_initialization_failed", symbol=symbol, error=str(e))
        # Continue without Brain Library if initialization fails
    
    logger.info(
        "model_training_config",
        symbol=symbol,
        n_estimators=model_config.n_estimators,
        learning_rate=model_config.learning_rate,
        max_depth=model_config.max_depth,
        early_stopping_rounds=model_config.early_stopping_rounds,
        total_splits=len(splits),
        total_samples=len(dataset),
        message="Starting intensive model training with configured hyperparameters",
    )
    
    # Print training configuration
    model_type = "Multi-Model Ensemble" if use_ensemble else "LightGBM"
    ensemble_info = f" ({', '.join(ensemble_techniques)})" if use_ensemble and ensemble_techniques else ""
    _print_status(f"Starting {model_type}{ensemble_info} training", sym=symbol, level="INFO")
    _print_status(f"Configuration: {model_config.n_estimators} estimators, LR={model_config.learning_rate}, Depth={model_config.max_depth}", sym=symbol, level="INFO")
    _print_status(f"Walk-forward validation: {len(splits)} splits ({settings.training.walk_forward.train_days} train / {settings.training.walk_forward.test_days} test days)", sym=symbol, level="INFO")
    
    # If no splits were created, log detailed diagnostics
    if len(splits) == 0:
        logger.error(
            "no_walk_forward_splits_created",
            symbol=symbol,
            ts_min=ts_min.isoformat() if hasattr(ts_min, 'isoformat') else str(ts_min),
            ts_max=ts_max.isoformat() if hasattr(ts_max, 'isoformat') else str(ts_max),
            ts_range_days=ts_range_days,
            train_days=settings.training.walk_forward.train_days,
            test_days=settings.training.walk_forward.test_days,
            required_range_days=settings.training.walk_forward.train_days + settings.training.walk_forward.test_days,
            message="No walk-forward splits created - check timestamp range and walk-forward settings",
        )
    
    total_training_start_time = time_module.time()
    ensemble_trainers = []  # Store ensemble trainers for final model
    split_metrics = []  # Store metrics for each split to show stability
    
    # Create progress bar for walk-forward splits
    split_iterator = enumerate(splits)
    if HAS_TQDM:
        split_iterator = tqdm(
            enumerate(splits),
            total=len(splits),
            desc=f"[{symbol}] Training splits",
            unit="split",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
    
    for split_idx, (train_mask, test_mask) in split_iterator:
        split_start_time = time_module.time()
        
        # CRITICAL: Recompute features for test set using only data up to split point
        # This prevents leakage from future data in rolling windows
        train_timestamps = dataset.loc[train_mask, "ts"]
        test_timestamps = dataset.loc[test_mask, "ts"]
        
        # Get training data (for training features - can use full dataset)
        X_train = dataset.loc[train_mask, feature_cols]
        y_train = dataset.loc[train_mask, "net_edge_bps"]

        # Sanity check labels before model training on this split.
        # Current pipeline treats net_edge_bps as a regression target.
        if False:  # Placeholder for future classification mode
            winners = int((y_train > 0).sum())
            losers = int((y_train <= 0).sum())
            total = int(len(y_train))
            win_rate = winners / total if total else 0.0

            logger.info(
                "label_sanity_check_split",
                symbol=symbol,
                split_idx=split_idx + 1,
                total=total,
                winners=winners,
                losers=losers,
                win_rate=win_rate,
            )

            if total == 0 or win_rate == 0.0 or win_rate == 1.0:
                logger.warning(
                    "labels_degenerate_split_skipped",
                    symbol=symbol,
                    split_idx=split_idx + 1,
                    total=total,
                    win_rate=win_rate,
                    message="Skipping split due to degenerate classification labels",
                )
                continue
        # Regression: require non-trivial variance
        var = float(np.var(y_train.values)) if len(y_train) > 0 else 0.0
        logger.info(
            "label_sanity_check_split_regression",
            symbol=symbol,
            split_idx=split_idx + 1,
            total=len(y_train),
            variance=var,
        )
        if var < 1e-6:
            logger.warning(
                "labels_low_variance_split_skipped",
                symbol=symbol,
                split_idx=split_idx + 1,
                variance=var,
                message="Skipping split due to near-zero target variance",
            )
            continue
        
        # For test set, recompute features using only data up to the split point
        # This ensures test features don't include future test data in rolling windows
        if not train_timestamps.empty and not test_timestamps.empty:
            split_point = train_timestamps.max()
            # Get all data up to and including the split point (end of training)
            # This is what would have been available at prediction time
            data_up_to_split = raw_frame.filter(pl.col("ts") <= split_point)
            
            # Also include test data for feature computation (but features will only use past data due to shift(1))
            # Actually, we need to include test data too for the test features to be computed
            # But the shift(1) ensures they only use past data
            data_for_test_features = raw_frame.filter(pl.col("ts") <= test_timestamps.max())
            
            # Recompute features for test set using only data up to test end
            # This ensures rolling windows don't include future test data
            test_feature_frame = recipe.build(data_for_test_features)
            
            # Convert to pandas and align with test mask
            test_feature_df = test_feature_frame.to_pandas()
            test_feature_df["ts"] = pd.to_datetime(test_feature_df["ts"])
            
            # Get test features from recomputed frame
            # CRITICAL: Only select columns that exist in the recomputed feature frame
            # (recipe.build() only creates features, not labels like edge_confidence or net_edge_bps)
            available_feature_cols = [col for col in feature_cols if col in test_feature_df.columns]
            if len(available_feature_cols) != len(feature_cols):
                missing_cols = set(feature_cols) - set(available_feature_cols)
                logger.warning(
                    "some_feature_cols_missing_in_recomputed_features",
                    symbol=symbol,
                    split_idx=split_idx + 1,
                    missing_cols=list(missing_cols),
                    available_cols_count=len(available_feature_cols),
                    expected_cols_count=len(feature_cols),
                    note="Recomputed features don't include label columns (edge_confidence, net_edge_bps) - this is expected"
                )
            
            test_ts_mask = test_feature_df["ts"].isin(test_timestamps)
            X_test_recomputed = test_feature_df.loc[test_ts_mask, available_feature_cols]
            
            # Use recomputed test features
            X_test = X_test_recomputed
            logger.debug(
                "test_features_recomputed",
                symbol=symbol,
                split_idx=split_idx + 1,
                split_point=split_point.isoformat() if hasattr(split_point, 'isoformat') else str(split_point),
                test_samples=len(X_test),
                note="Test features recomputed using only data up to test end - prevents leakage"
            )
        else:
            # Fallback: use pre-computed features (less ideal but faster)
            X_test = dataset.loc[test_mask, feature_cols]
        
        if len(y_train) < 100:
            logger.warning(
                "skipping_split_insufficient_data",
                symbol=symbol,
                split_idx=split_idx,
                train_samples=len(y_train),
                min_required=100,
            )
            continue
        
        # Create validation set from training data (20% for early stopping)
        train_size = int(0.8 * len(X_train))
        X_train_split = X_train.iloc[:train_size].copy()
        y_train_split = y_train.iloc[:train_size].copy()
        X_val_split = X_train.iloc[train_size:]
        y_val_split = y_train.iloc[train_size:]
        
        # Apply noise injection to training data if enabled (helps prevent overfitting)
        if noise_injection_std > 0:
            noise = np.random.normal(0, noise_injection_std, size=X_train_split.shape)
            X_train_split = X_train_split + noise
            logger.debug(
                "noise_injection_applied",
                symbol=symbol,
                split_idx=split_idx + 1,
                noise_std=noise_injection_std,
                message="Added Gaussian noise to training features to improve generalization"
            )
        
        logger.info(
            "training_split_start",
            symbol=symbol,
            split_idx=split_idx + 1,
            total_splits=len(splits),
            train_samples=len(X_train_split),
            val_samples=len(X_val_split),
            test_samples=test_mask.sum(),
            use_ensemble=use_ensemble,
            ensemble_method=ensemble_method if use_ensemble else "single_model",
        )
        
        # Print split progress
        progress_pct = ((split_idx + 1) / len(splits)) * 100
        _print_status(f"Split {split_idx + 1}/{len(splits)} ({progress_pct:.1f}%): Training on {len(X_train_split):,} samples, validating on {len(X_val_split):,}", sym=symbol, level="PROGRESS")
        
        # Use multi-model ensemble if enabled, otherwise single LightGBM
        if use_ensemble:
            logger.info(
                "training_multi_model_ensemble",
                symbol=symbol,
                split_idx=split_idx + 1,
                techniques=ensemble_techniques,
                ensemble_method=ensemble_method,
            )
            _print_status(f"Training ensemble: {', '.join(ensemble_techniques)}", sym=symbol, level="PROGRESS")
            
            # Train ensemble with hyperparameters and fixed weights
            ensemble_trainer, ensemble_results = train_multi_model_ensemble(
                X_train=X_train_split,
                y_train=y_train_split,
                X_val=X_val_split,
                y_val=y_val_split,
                regimes=None,  # Could add regime detection here
                techniques=ensemble_techniques,
                ensemble_method=ensemble_method,
                is_classification=False,
                hyperparams=ensemble_hyperparams,
                fixed_ensemble_weights=fixed_weights,
                use_fixed_weights=use_fixed,
            )
            
            ensemble_trainers.append(ensemble_trainer)
            
            # X_test already computed above with proper feature recomputation
            if X_test.empty:
                continue
            
            # Validate no data leakage: ensure test timestamps are strictly after train timestamps
            train_timestamps = dataset.loc[train_mask, "ts"]
            test_timestamps = dataset.loc[test_mask, "ts"]
            if not train_timestamps.empty and not test_timestamps.empty:
                max_train_ts = train_timestamps.max()
                min_test_ts = test_timestamps.min()
                if min_test_ts <= max_train_ts:
                    logger.warning(
                        "potential_data_leakage_detected",
                        symbol=symbol,
                        split_idx=split_idx + 1,
                        max_train_ts=max_train_ts.isoformat(),
                        min_test_ts=min_test_ts.isoformat(),
                        message="Test data overlaps with training data - potential lookahead bias!"
                    )
                else:
                    logger.debug(
                        "data_leakage_check_passed",
                        symbol=symbol,
                        split_idx=split_idx + 1,
                        gap_days=(min_test_ts - max_train_ts).days,
                        message="Test data is strictly after training data - no leakage"
                    )
            
            # Use ensemble prediction
            ensemble_result = ensemble_trainer.predict_ensemble(X_test, regime=None)
            predictions = ensemble_result.prediction
            confidence = ensemble_result.confidence * np.ones(len(predictions))  # Use ensemble confidence
            
            logger.info(
                "ensemble_training_complete",
                symbol=symbol,
                split_idx=split_idx + 1,
                num_models=len(ensemble_results),
                ensemble_confidence=ensemble_result.confidence,
                model_contributions=ensemble_result.model_contributions,
            )
            
            # Find best model
            best_model = max(ensemble_results.items(), key=lambda x: x[1].val_score if x[1].val_score is not None else -999)[0] if ensemble_results else "unknown"
            _print_status(f"Ensemble trained: {len(ensemble_results)} models, best={best_model}, confidence={ensemble_result.confidence:.3f}", sym=symbol, level="SUCCESS")
        else:
            # Train single LightGBM model with early stopping
            _print_status("Training LightGBM model...", sym=symbol, level="PROGRESS")
            import lightgbm as lgb  # type: ignore[reportMissingImports]
            model = LGBMRegressor(**hyperparams)
            
            callbacks = []
            if model_config.early_stopping_rounds > 0:
                callbacks.append(lgb.early_stopping(stopping_rounds=model_config.early_stopping_rounds, verbose=False))
            callbacks.append(lgb.log_evaluation(period=0))  # Disable verbose logging
            
            model.fit(
                X_train_split,
                y_train_split,
                eval_set=[(X_val_split, y_val_split)],
                callbacks=callbacks if callbacks else None,
            )
            
            # Log early stopping info if it was triggered
            if hasattr(model, 'best_iteration_') and model.best_iteration_ is not None:
                actual_iterations = model.best_iteration_ + 1
                logger.info(
                    "early_stopping_triggered",
                    symbol=symbol,
                    split_idx=split_idx + 1,
                    best_iteration=model.best_iteration_,
                    actual_iterations=actual_iterations,
                    total_iterations=model_config.n_estimators,
                    stopped_early=True,
                )
                _print_status(f"LightGBM trained: {actual_iterations}/{model_config.n_estimators} iterations (early stopped)", sym=symbol, level="SUCCESS")
            else:
                logger.info(
                    "training_completed_all_iterations",
                    symbol=symbol,
                    split_idx=split_idx + 1,
                    total_iterations=model_config.n_estimators,
                    stopped_early=False,
                )
                _print_status(f"LightGBM trained: {model_config.n_estimators} iterations (full training)", sym=symbol, level="SUCCESS")
            
            # X_test already computed above with proper feature recomputation
            if X_test.empty:
                continue
            
            # Validate no data leakage: ensure test timestamps are strictly after train timestamps
            train_timestamps = dataset.loc[train_mask, "ts"]
            test_timestamps = dataset.loc[test_mask, "ts"]
            if not train_timestamps.empty and not test_timestamps.empty:
                max_train_ts = train_timestamps.max()
                min_test_ts = test_timestamps.min()
                if min_test_ts <= max_train_ts:
                    logger.warning(
                        "potential_data_leakage_detected",
                        symbol=symbol,
                        split_idx=split_idx + 1,
                        max_train_ts=max_train_ts.isoformat(),
                        min_test_ts=min_test_ts.isoformat(),
                        message="Test data overlaps with training data - potential lookahead bias!"
                    )
                else:
                    logger.debug(
                        "data_leakage_check_passed",
                        symbol=symbol,
                        split_idx=split_idx + 1,
                        gap_days=(min_test_ts - max_train_ts).days,
                        message="Test data is strictly after training data - no leakage"
                    )
            
            predictions = model.predict(X_test)
            confidence = dataset.loc[test_mask, "edge_confidence"].to_numpy()
        
        split_training_time = time_module.time() - split_start_time
        
        y_test = dataset.loc[test_mask, "net_edge_bps"].to_numpy()
        timestamps = dataset.loc[test_mask, "ts"].to_numpy()
        
        # Trade funnel diagnostics: how many predictions survive the edge threshold?
        total_candidates = int(len(predictions))
        executed_mask = predictions >= cost_threshold
        executed_count = int(executed_mask.sum())

        logger.info(
            "trade_funnel_split",
            symbol=symbol,
            split_idx=split_idx + 1,
            total_candidates=total_candidates,
            above_edge_threshold=executed_count,
            edge_threshold_bps=cost_threshold,
        )

        trades = _simulate_trades(predictions, y_test, confidence, timestamps, cost_threshold)
        if not trades.empty:
            oos_trades.append(trades)
        
        logger.info(
            "training_split_complete",
            symbol=symbol,
            split_idx=split_idx + 1,
            training_time_seconds=split_training_time,
            test_trades=len(trades),
            use_ensemble=use_ensemble,
            message=f"Split {split_idx + 1}/{len(splits)} training completed",
        )
        
        # Calculate and store split metrics
        split_metric = {
            "split_idx": split_idx + 1,
            "trades": len(trades),
            "training_time": split_training_time,
        }
        if not trades.empty:
            split_metric.update({
                "sharpe": _compute_metrics(trades, costs.total_costs_bps, len(dataset)).get("sharpe", 0.0),
                "profit_factor": _compute_metrics(trades, costs.total_costs_bps, len(dataset)).get("profit_factor", 0.0),
                "hit_rate": _compute_metrics(trades, costs.total_costs_bps, len(dataset)).get("hit_rate", 0.0) * 100,
                "pnl_bps": trades["pnl_bps"].sum(),
                "avg_pnl": trades["pnl_bps"].mean(),
                "max_dd": _compute_metrics(trades, costs.total_costs_bps, len(dataset)).get("max_dd_bps", 0.0),
            })
            avg_pnl = trades["pnl_bps"].mean()
            win_rate = (trades["pnl_bps"] > 0).mean() * 100
            sharpe = split_metric["sharpe"]
            _print_status(f"Split {split_idx + 1}/{len(splits)} complete: {len(trades)} trades, Sharpe={sharpe:.2f}, Avg PnL={avg_pnl:.2f} bps, Win Rate={win_rate:.1f}% ({split_training_time:.1f}s)", sym=symbol, level="SUCCESS")
        else:
            split_metric.update({
                "sharpe": 0.0,
                "profit_factor": 0.0,
                "hit_rate": 0.0,
                "pnl_bps": 0.0,
                "avg_pnl": 0.0,
                "max_dd": 0.0,
            })
            _print_status(f"Split {split_idx + 1}/{len(splits)} complete: 0 trades generated ({split_training_time:.1f}s)", sym=symbol, level="WARN")
        
        split_metrics.append(split_metric)
    
    total_training_time = time_module.time() - total_training_start_time
    logger.info(
        "all_splits_training_complete",
        symbol=symbol,
        total_splits=len(splits),
        total_training_time_seconds=total_training_time,
        total_training_time_minutes=total_training_time / 60.0,
        message=f"All walk-forward splits trained in {total_training_time:.2f} seconds ({total_training_time / 60.0:.2f} minutes)",
    )

    combined_trades = pd.concat(oos_trades, ignore_index=True) if oos_trades else pd.DataFrame(columns=["timestamp", "pnl_bps", "prediction_bps", "confidence", "equity_curve"])  # type: ignore[call-overload]
    
    # Print detailed split-by-split results using _print_status for Ray compatibility
    _print_status("", sym=symbol, level="INFO")  # Empty line
    _print_status("=" * 80, sym=symbol, level="INFO")
    _print_status(f"📊 WALK-FORWARD VALIDATION RESULTS - {symbol}", sym=symbol, level="INFO")
    _print_status("=" * 80, sym=symbol, level="INFO")
    if split_metrics:
        header = f"{'Split':<8} {'Trades':<10} {'Sharpe':<10} {'Hit Rate':<12} {'PnL (bps)':<15} {'Max DD (bps)':<15}"
        _print_status(header, sym=symbol, level="INFO")
        _print_status("-" * 80, sym=symbol, level="INFO")
        for sm in split_metrics:
            row = f"{sm['split_idx']:<8} {sm['trades']:<10} {sm['sharpe']:<10.2f} {sm['hit_rate']:<12.1f}% {sm['pnl_bps']:<15.2f} {sm['max_dd']:<15.1f}"
            _print_status(row, sym=symbol, level="INFO")
        
        # Calculate stability metrics
        sharpe_values = [sm['sharpe'] for sm in split_metrics if sm['trades'] > 0]
        hit_rate_values = [sm['hit_rate'] for sm in split_metrics if sm['trades'] > 0]
        if sharpe_values:
            sharpe_mean = np.mean(sharpe_values)
            sharpe_std = np.std(sharpe_values)
            sharpe_cv = (sharpe_std / abs(sharpe_mean)) * 100 if sharpe_mean != 0 else 0
            hit_rate_mean = np.mean(hit_rate_values)
            hit_rate_std = np.std(hit_rate_values)
            
            _print_status("-" * 80, sym=symbol, level="INFO")
            _print_status("", sym=symbol, level="INFO")  # Empty line
            _print_status("📈 STABILITY ANALYSIS:", sym=symbol, level="INFO")
            _print_status(f"  Sharpe Ratio: Mean={sharpe_mean:.2f}, Std={sharpe_std:.2f}, CV={sharpe_cv:.1f}%", sym=symbol, level="INFO")
            _print_status(f"  Hit Rate: Mean={hit_rate_mean:.1f}%, Std={hit_rate_std:.1f}%", sym=symbol, level="INFO")
            
            # Determine stability
            is_stable = sharpe_cv < 50 and len([s for s in sharpe_values if s > 0]) >= len(sharpe_values) * 0.7
            stability_status = "✅ STABLE" if is_stable else "⚠️  VARIABLE"
            _print_status(f"  Stability: {stability_status}", sym=symbol, level="INFO")
            if not is_stable:
                _print_status("  ⚠️  Warning: Model performance varies significantly across time periods", sym=symbol, level="WARN")
                _print_status("     Consider: More training data, different features, or regime detection", sym=symbol, level="WARN")
        _print_status("=" * 80, sym=symbol, level="INFO")
        _print_status("", sym=symbol, level="INFO")  # Empty line
    
    # Print trade summary
    total_trades = len(combined_trades) if not combined_trades.empty else 0
    _print_status(f"Walk-forward training complete: {len(splits)} splits, {total_trades:,} total trades, {total_training_time/60:.1f} minutes", sym=symbol, level="SUCCESS")
    
    if not combined_trades.empty:
        winning_trades = (combined_trades["pnl_bps"] > 0).sum()
        losing_trades = (combined_trades["pnl_bps"] <= 0).sum()
        total_pnl = combined_trades["pnl_bps"].sum()
        avg_pnl = combined_trades["pnl_bps"].mean()
        _print_status(f"Trade Summary: {total_trades:,} trades ({winning_trades:,} wins, {losing_trades:,} losses), Total PnL={total_pnl:.2f} bps, Avg={avg_pnl:.2f} bps", sym=symbol, level="INFO")

        # Persist aggregated trade stats for meta tuning (symbol/mode level).
        # Approximate trades_per_day from OOS trades timestamp range.
        try:
            ts_min = combined_trades["timestamp"].min()
            ts_max = combined_trades["timestamp"].max()
            if isinstance(ts_min, pd.Timestamp) and isinstance(ts_max, pd.Timestamp):
                days_range = max((ts_max - ts_min).total_seconds() / 86400.0, 1.0)
            else:
                days_range = max(len(combined_trades) / 28800.0, 1.0)  # fallback heuristic
            trades_per_day = float(total_trades / days_range)

            if brain_training is not None:
                # For now, treat this training as "scalp" mode by default.
                training_mode = getattr(settings.training, "trading_mode", "scalp")
                try:
                    brain_training.brain.store_symbol_mode_trade_stats(  # type: ignore[attr-defined]
                        stat_date=run_date,
                        symbol=symbol,
                        mode=str(training_mode),
                        trades=int(total_trades),
                        trades_per_day=trades_per_day,
                    )
                except Exception as e:
                    logger.warning(
                        "symbol_mode_trade_stats_store_failed",
                        symbol=symbol,
                        mode=str(training_mode),
                        error=str(e),
                    )
        except Exception as e:
            logger.warning(
                "symbol_mode_trade_stats_compute_failed",
                symbol=symbol,
                error=str(e),
            )
    else:
        _print_status("WARNING: No trades generated during walk-forward validation", sym=symbol, level="WARN")
    
    metrics = _compute_metrics(combined_trades, costs.total_costs_bps, len(dataset))
    
    # Print key metrics with robustness warnings
    sharpe = metrics.get('sharpe', 0)
    hit_rate = metrics.get('hit_rate', 0) * 100
    pnl_bps = metrics.get('pnl_bps', 0)
    is_unrealistic = metrics.get('is_unrealistic', False)
    validation_warnings = metrics.get('validation_warnings', [])
    
    _print_status(f"AGGREGATED Performance: Sharpe={sharpe:.2f}, Hit Rate={hit_rate:.1f}%, PnL={pnl_bps:.2f} bps", sym=symbol, level="INFO")
    
    # Overfitting detection: Display validation warnings
    if is_unrealistic:
        _print_status("", sym=symbol, level="WARN")
        _print_status("🚨 UNREALISTIC METRICS DETECTED - MODEL LIKELY OVERFITTED OR DATA LEAKAGE:", sym=symbol, level="WARN")
        for warning in validation_warnings:
            _print_status(f"  ⚠️  {warning}", sym=symbol, level="WARN")
        _print_status("", sym=symbol, level="WARN")
        _print_status("  RECOMMENDATIONS:", sym=symbol, level="WARN")
        _print_status("  1. Verify no future data is used in feature generation", sym=symbol, level="WARN")
        _print_status("  2. Check label creation - labels should be forward-looking only", sym=symbol, level="WARN")
        _print_status("  3. Run with shuffle_candles=true to test robustness", sym=symbol, level="WARN")
        _print_status("  4. Shift validation window forward and retrain", sym=symbol, level="WARN")
        _print_status("  5. Review data leakage checks in walk-forward validation", sym=symbol, level="WARN")
        _print_status("", sym=symbol, level="WARN")
    
    # Overfitting detection: If Sharpe > 9, warn about potential overfitting
    if sharpe > 9.0:
        _print_status(f"⚠️  WARNING: Sharpe ratio ({sharpe:.2f}) is extremely high (>9). This likely indicates overfitting.", sym=symbol, level="WARN")
        _print_status("   Recommendation: Run with shuffle_candles=true to test robustness", sym=symbol, level="WARN")
    
    # Compare with paper trading results if available
    paper_trading_sharpe = None
    try:
        # Check if paper trading results exist for this symbol
        if dsn:
            import psycopg2  # type: ignore[reportMissingImports]
            conn = psycopg2.connect(dsn)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT sharpe_ratio, hit_rate, pnl_bps 
                FROM paper_trading_results 
                WHERE symbol = %s 
                ORDER BY created_at DESC 
                LIMIT 1
            """, (symbol,))
            result = cursor.fetchone()
            if result:
                paper_trading_sharpe, paper_hit_rate, paper_pnl = result
                _print_status("", sym=symbol, level="INFO")
                _print_status("📊 PAPER TRADING COMPARISON:", sym=symbol, level="INFO")
                _print_status(f"  Backtest Sharpe: {sharpe:.2f} | Paper Trading Sharpe: {paper_trading_sharpe:.2f}", sym=symbol, level="INFO")
                _print_status(f"  Backtest Hit Rate: {hit_rate:.1f}% | Paper Trading Hit Rate: {paper_hit_rate*100:.1f}%", sym=symbol, level="INFO")
                
                # Overfitting detection: If backtest Sharpe >> paper trading Sharpe, likely overfitting
                if paper_trading_sharpe and sharpe > paper_trading_sharpe * 2:
                    _print_status(f"  ⚠️  OVERFITTING DETECTED: Backtest Sharpe ({sharpe:.2f}) is >2x paper trading Sharpe ({paper_trading_sharpe:.2f})", sym=symbol, level="WARN")
                    _print_status("     Model may not generalize to live trading", sym=symbol, level="WARN")
                elif paper_trading_sharpe and sharpe < 5 and paper_trading_sharpe < 5:
                    _print_status(f"  ✅ Model performance is consistent between backtest and paper trading", sym=symbol, level="SUCCESS")
            cursor.close()
            conn.close()
    except Exception as e:
        logger.debug("paper_trading_comparison_unavailable", symbol=symbol, error=str(e))
    
    # Brain Library integration: Store final model metrics after all splits
    if brain_training:
        try:
            logger.info("training_final_model_for_brain_integration", symbol=symbol)
            final_model_start_time = time_module.time()
            
            # Train final model on all data - use ensemble if enabled
            brain_ensemble_trainer = None  # Initialize for use in testing
            if use_ensemble:
                # Train ensemble for brain integration with hyperparameters and fixed weights
                brain_ensemble_trainer, brain_ensemble_results = train_multi_model_ensemble(
                    X_train=dataset[feature_cols],
                    y_train=dataset["net_edge_bps"],
                    X_val=None,
                    y_val=None,
                    regimes=None,
                    techniques=ensemble_techniques,
                    ensemble_method=ensemble_method,
                    is_classification=False,
                    hyperparams=ensemble_hyperparams,
                    fixed_ensemble_weights=fixed_weights,
                    use_fixed_weights=use_fixed,
                )
                # For brain integration, use the best single model from ensemble
                # (Brain integration may not support ensemble directly)
                best_model_name = brain_ensemble_trainer.get_best_model()
                if best_model_name and best_model_name in brain_ensemble_trainer.models:
                    final_model = brain_ensemble_trainer.models[best_model_name]
                    model_type = "lightgbm" if "lightgbm" in best_model_name.lower() else "xgboost" if "xgboost" in best_model_name.lower() else "random_forest"
                else:
                    # Fallback to first model
                    final_model = list(brain_ensemble_trainer.models.values())[0]
                    model_type = "lightgbm"
            else:
                # Train single LightGBM model
                final_model = LGBMRegressor(**hyperparams)
                final_model.fit(dataset[feature_cols], dataset["net_edge_bps"])
                model_type = "lightgbm"
            
            final_model_training_time = time_module.time() - final_model_start_time
            logger.info(
                "final_model_training_complete",
                symbol=symbol,
                training_time_seconds=final_model_training_time,
                total_samples=len(dataset),
                use_ensemble=use_ensemble,
                model_type=model_type,
            )
            
            # Test final model on ALL unseen test windows (not just the last one)
            # This gives a more comprehensive evaluation of the final model
            if splits:
                all_test_trades = []
                for split_idx, (_, test_mask) in enumerate(splits):
                    final_X_test = dataset.loc[test_mask, feature_cols]
                    final_y_test = dataset.loc[test_mask, "net_edge_bps"]
                    final_confidence = dataset.loc[test_mask, "edge_confidence"].to_numpy()
                    final_timestamps = dataset.loc[test_mask, "ts"].to_numpy()
                    
                    if not final_X_test.empty:
                        # Get predictions from final model
                        if use_ensemble and brain_ensemble_trainer:
                            # Use the ensemble trainer for predictions
                            final_predictions = brain_ensemble_trainer.predict_ensemble(final_X_test, regime=None).prediction
                        else:
                            # Use single model
                            final_predictions = final_model.predict(final_X_test)
                        
                        # Simulate trades on this test window
                        test_trades = _simulate_trades(
                            final_predictions,
                            final_y_test.to_numpy(),
                            final_confidence,
                            final_timestamps,
                            cost_threshold
                        )
                        if not test_trades.empty:
                            all_test_trades.append(test_trades)
                            logger.debug(
                                "final_model_test_split",
                                symbol=symbol,
                                split_idx=split_idx + 1,
                                trades=len(test_trades),
                                message=f"Final model tested on split {split_idx + 1}"
                            )
                
                # Combine all test trades from final model
                if all_test_trades:
                    combined_final_trades = pd.concat(all_test_trades, ignore_index=True)
                    final_metrics = _compute_metrics(combined_final_trades, costs.total_costs_bps, len(dataset))
                    logger.info(
                        "final_model_evaluation_complete",
                        symbol=symbol,
                        total_test_splits=len(splits),
                        total_trades=len(combined_final_trades),
                        sharpe=final_metrics.get("sharpe", 0.0),
                        hit_rate=final_metrics.get("hit_rate", 0.0),
                        message="Final model evaluated on all unseen test windows"
                    )
                    _print_status(f"Final model tested on {len(splits)} unseen windows: {len(combined_final_trades)} trades, Sharpe={final_metrics.get('sharpe', 0):.2f}", sym=symbol, level="SUCCESS")
                
                # For brain integration, use the last test split (as before)
                last_train_mask, last_test_mask = splits[-1]
                final_X_test = dataset.loc[last_test_mask, feature_cols]
                final_y_test = dataset.loc[last_test_mask, "net_edge_bps"]
                
                if not final_X_test.empty:
                    brain_result = brain_training.train_with_brain_integration(
                        symbol=symbol,
                        X_train=dataset[feature_cols],
                        y_train=dataset["net_edge_bps"],
                        X_test=final_X_test,
                        y_test=final_y_test,
                        feature_names=feature_cols,
                        base_model=final_model,
                        model_type=model_type,
                    )
                    logger.info("brain_integration_final_complete", symbol=symbol, status=brain_result.get("status"))
        except Exception as e:
            logger.warning("brain_integration_final_failed", symbol=symbol, error=str(e))

    # Run validation pipeline if enabled
    validation_passed = True
    if settings.training.validation.enabled:
        try:
            from ..validation.validation_pipeline import ValidationPipeline
            from ..engine.walk_forward import WalkForwardResults  # type: ignore[reportMissingImports]

            # Create walk-forward results from metrics
            wf_results = WalkForwardResults(
                windows=[],
                total_windows=len(splits),
                test_sharpe=metrics["sharpe"],
                test_win_rate=metrics["hit_rate"],
                test_avg_pnl_bps=metrics["pnl_bps"] / max(metrics["trades_oos"], 1),
                sharpe_std=0.0,  # Would calculate from splits
                win_rate_std=0.0,
                train_test_sharpe_diff=0.0,  # Would calculate from train metrics
                train_test_wr_diff=0.0,
            )

            validation_pipeline = ValidationPipeline(settings=settings)
            validation_result = validation_pipeline.validate(
                walk_forward_results=wf_results,
                model_id=f"{symbol}-{run_date:%Y%m%d}",
                data=raw_frame,
                symbol=symbol,
            )
            validation_passed = validation_result.passed

            if not validation_passed:
                logger.error(
                    "validation_failed",
                    symbol=symbol,
                    blocking_issues=validation_result.blocking_issues,
                )
                
                # Notify Telegram about validation failure
                # Note: telegram_monitor is passed via closure in _train_symbol
                # We'll add it as a parameter to _train_symbol
        except Exception as e:
            logger.warning(
                "validation_pipeline_error",
                symbol=symbol,
                error=str(e),
                error_type=type(e).__name__,
            )
            # Don't block on validation errors - log and continue
            # But if it's a ValueError (hard block), re-raise
            if isinstance(e, ValueError):
                raise

    if metrics["trades_oos"] < settings.training.walk_forward.min_trades or not validation_passed:
        reason = "gate_trades"
        gate_results = {
            "sharpe_pass": False,
            "profit_factor_pass": False,
            "max_dd_pass": False,
            "hit_rate_pass": False,
            "trades_pass": False,
        }
        metrics.update({
            "recommended_edge_threshold_bps": cost_threshold, 
            "validation_window": f"{dataset['ts'].min().date()} to {dataset['ts'].max().date()}",
            "num_walk_forward_splits": len(splits) if 'splits' in locals() else 0,  # Number of times model was tested
        })
        return TrainingTaskResult(
            symbol=symbol,
            costs=costs,
            metrics=metrics,
            gate_results=gate_results,
            published=False,
            reason=reason,
            artifacts=None,
            model_id=str(uuid4()),
            run_id=f"{symbol}-{run_date:%Y%m%d}",
            pilot_contract=None,
            mechanic_contract=None,
            metrics_payload=None,
            model_params=hyperparams,
            feature_metadata={"columns": feature_cols},
        )

    # Calculate median values for display (needed regardless of unrealistic check)
    median_dd = median([abs(entry.get("max_dd_bps", 0.0)) for entry in history]) if history else abs(metrics["max_dd_bps"])
    median_hit = median([entry.get("hit_rate", 0.0) for entry in history]) if history else metrics["hit_rate"]
    
    # Check for unrealistic metrics (overfitting/data leakage) - reject even if gates pass
    is_unrealistic = metrics.get('is_unrealistic', False)
    if is_unrealistic:
        _print_status("", sym=symbol, level="WARN")
        _print_status("🚨 MODEL REJECTED: Unrealistic metrics indicate overfitting or data leakage", sym=symbol, level="WARN")
        _print_status("   Model will NOT be published despite passing gates", sym=symbol, level="WARN")
        _print_status("", sym=symbol, level="WARN")
        gate_results = {
            "sharpe_pass": False,
            "profit_factor_pass": False,
            "max_dd_pass": False,
            "hit_rate_pass": False,
            "trades_pass": metrics["trades_oos"] >= settings.training.walk_forward.min_trades,
            "unrealistic_metrics": True,  # Special flag for unrealistic metrics
        }
        published = False
        reason = "unrealistic_metrics_overfitting"
        failed = ["unrealistic_metrics"]
        logger.warning(
            "model_rejected_unrealistic_metrics",
            symbol=symbol,
            sharpe=metrics.get('sharpe', 0),
            profit_factor=metrics.get('profit_factor', 0),
            hit_rate=metrics.get('hit_rate', 0),
            max_dd=metrics.get('max_dd_bps', 0),
            warnings=metrics.get('validation_warnings', []),
            message="Model rejected due to unrealistic metrics indicating overfitting or data leakage"
        )
    else:
        # Gate checks with min/max thresholds
        min_sharpe = 0.7
        max_realistic_sharpe = 9.0
        min_pf = 1.1
        max_realistic_pf = 5.0
        
        gate_results = {
            "sharpe_pass": min_sharpe <= metrics["sharpe"] <= max_realistic_sharpe,
            "profit_factor_pass": min_pf <= metrics["profit_factor"] <= max_realistic_pf,
            "max_dd_pass": abs(metrics["max_dd_bps"]) <= 1.2 * median_dd if median_dd else True,
            "hit_rate_pass": metrics["hit_rate"] >= max(median_hit - 0.01, 0.0),
            "trades_pass": metrics["trades_oos"] >= settings.training.walk_forward.min_trades,
            "unrealistic_metrics": False,
        }
        published = all([v for k, v in gate_results.items() if k != "unrealistic_metrics"])
        failed = [name for name, passed in gate_results.items() if not passed and name != "unrealistic_metrics"]
        reason = "all_pass" if published else "gate_failure:" + ",".join(failed)
    
    # Print gate results with min/max thresholds
    min_sharpe = 0.7
    max_realistic_sharpe = 9.0
    min_pf = 1.1
    max_realistic_pf = 5.0
    
    sharpe_pass = min_sharpe <= metrics['sharpe'] <= max_realistic_sharpe
    pf_pass = min_pf <= metrics['profit_factor'] <= max_realistic_pf
    
    print(f"\n{'='*80}")
    print(f"📊 [{symbol}] GATE RESULTS")
    print(f"{'='*80}")
    print(f"  Sharpe Ratio:      {metrics['sharpe']:.2f} {'✅ PASS' if sharpe_pass else '❌ FAIL'} (min: {min_sharpe}, max: {max_realistic_sharpe})")
    print(f"  Profit Factor:     {metrics['profit_factor']:.2f} {'✅ PASS' if pf_pass else '❌ FAIL'} (min: {min_pf}, max: {max_realistic_pf})")
    print(f"  Hit Rate:          {metrics['hit_rate']*100:.1f}% {'✅ PASS' if gate_results['hit_rate_pass'] else '❌ FAIL'} (threshold: {max(median_hit - 0.01, 0.0)*100:.1f}%)")
    print(f"  Max Drawdown:      {metrics['max_dd_bps']:.2f} bps {'✅ PASS' if gate_results['max_dd_pass'] else '❌ FAIL'} (threshold: {1.2 * median_dd:.2f} bps)")
    print(f"  Trades:            {metrics['trades_oos']:,} {'✅ PASS' if gate_results['trades_pass'] else '❌ FAIL'} (threshold: {settings.training.walk_forward.min_trades:,})")
    if gate_results.get('unrealistic_metrics', False):
        print(f"  Unrealistic Metrics: ❌ FAIL (overfitting/data leakage detected)")
    print(f"{'='*80}")
    if published:
        print(f"✅ [{symbol}] ALL GATES PASSED - Model will be published")
    else:
        if gate_results.get('unrealistic_metrics', False):
            print(f"❌ [{symbol}] MODEL REJECTED - Unrealistic metrics indicate overfitting or data leakage")
        else:
            print(f"❌ [{symbol}] GATE FAILURE - Model rejected: {', '.join(failed)}")
    print(f"{'='*80}\n")

    logger.info("training_final_production_model", symbol=symbol)
    _print_status("Training final production model on all data...", sym=symbol, level="PROGRESS")
    final_production_start_time = time_module.time()
    
    # Train final model on all data for production use
    # Use ensemble if enabled, otherwise single model
    if use_ensemble:
        logger.info(
            "training_final_ensemble_model",
            symbol=symbol,
            techniques=ensemble_techniques,
            ensemble_method=ensemble_method,
        )
        
        # Train final ensemble on all data with hyperparameters and fixed weights
        final_ensemble_trainer, final_ensemble_results = train_multi_model_ensemble(
            X_train=dataset[feature_cols],
            y_train=dataset["net_edge_bps"],
            X_val=None,  # Use all data for final model
            y_val=None,
            regimes=None,
            techniques=ensemble_techniques,
            ensemble_method=ensemble_method,
            is_classification=False,
            hyperparams=ensemble_hyperparams,
            fixed_ensemble_weights=fixed_weights,
            use_fixed_weights=use_fixed,
        )
        
        # For serialization, we'll pickle the ensemble trainer
        # In production, use the ensemble trainer for predictions
        model_bytes = pickle.dumps(final_ensemble_trainer)
        
        # Get feature importances from best model or average
        feature_importances = []
        if final_ensemble_trainer.models:
            # Average feature importances across all models
            all_importances = []
            for model_name, model in final_ensemble_trainer.models.items():
                if hasattr(model, 'feature_importances_'):
                    all_importances.append(model.feature_importances_)
            
            if all_importances:
                feature_importances = np.mean(all_importances, axis=0).tolist()
            else:
                feature_importances = [0.0] * len(feature_cols)
        else:
            feature_importances = [0.0] * len(feature_cols)
        
        logger.info(
            "final_ensemble_model_training_complete",
            symbol=symbol,
            num_models=len(final_ensemble_results),
            ensemble_method=ensemble_method,
        )
        _print_status(f"Final ensemble model trained: {len(final_ensemble_results)} models", sym=symbol, level="SUCCESS")
    else:
        # Train single LightGBM model
        final_model = LGBMRegressor(**hyperparams)
        final_model.fit(dataset[feature_cols], dataset["net_edge_bps"])
        
        model_bytes = pickle.dumps(final_model)
        feature_importances = final_model.feature_importances_.tolist()
    
    final_production_training_time = time_module.time() - final_production_start_time
    logger.info(
        "final_production_model_training_complete",
        symbol=symbol,
        training_time_seconds=final_production_training_time,
        training_time_minutes=final_production_training_time / 60.0,
        total_samples=len(dataset),
        use_ensemble=use_ensemble,
        message=f"Final production model trained in {final_production_training_time:.2f} seconds",
    )

    validation_window = f"{dataset['ts'].min().date()} to {dataset['ts'].max().date()}"
    run_id = f"{mode}-{symbol}-{run_date:%Y%m%d}"

    metrics_payload = MetricsPayload(
        symbol=symbol,
        run_id=run_id,
        validation_window=validation_window,
        sharpe=float(metrics["sharpe"]),
        profit_factor=float(metrics["profit_factor"]),
        hit_rate_pct=float(metrics["hit_rate"]) * 100.0,
        max_drawdown_bps=float(metrics["max_dd_bps"]),
        trades=int(metrics["trades_oos"]),
        turnover_pct=float(metrics["turnover"]),
        pnl_bps=float(metrics["pnl_bps"]),
        costs={
            "fee_bps": costs.fee_bps,
            "spread_bps": costs.spread_bps,
            "slippage_bps": costs.slippage_bps,
            "total_costs_bps": costs.total_costs_bps,
        },
        gate_results=gate_results,
    )

    metrics.update(
        {
            "recommended_edge_threshold_bps": cost_threshold,
            "validation_window": validation_window,
            "num_walk_forward_splits": len(splits),  # Number of times model was tested
            "mode": mode,
        }
    )

    pilot_contract = PilotContract(
        date=run_date,
        symbol=symbol,
        recommended_edge_threshold_bps=cost_threshold,
        taker_extra_buffer_bps=settings.costs.taker_buffer_bps,
        max_trades_per_day_hint=40,
        cooldown_seconds_hint=900,
        costs_breakdown={
            "fee_bps": costs.fee_bps,
            "spread_bps": costs.spread_bps,
            "slippage_bps": costs.slippage_bps,
            "total_costs_bps": costs.total_costs_bps,
        },
        validation_summary={
            "sharpe": metrics["sharpe"],
            "profit_factor": metrics["profit_factor"],
            "hit_rate_pct": metrics["hit_rate"] * 100,
            "max_drawdown_bps": metrics["max_dd_bps"],
            "trades": metrics["trades_oos"],
            "publish_status": "published" if published else "rejected",
        },
        gate_flags=gate_results,
        notes=None,
    )

    mechanic_contract = MechanicContract(
        date=run_date,
        symbol=symbol,
        edge_threshold_bps=cost_threshold,
        entry_exit_horizons={"primary_minutes": 4, "alternates_minutes": [3, 5]},
        taker_policy_hint="post_only_preferred",
        taker_cross_buffer_bps=settings.costs.taker_buffer_bps,
        cooldown_seconds_hint=900,
        max_trades_hint=40,
        costs_components={
            "fee_bps": costs.fee_bps,
            "spread_bps": costs.spread_bps,
            "slippage_bps": costs.slippage_bps,
            "total_costs_bps": costs.total_costs_bps,
        },
        promotion_criteria_hint={
            "delta_sharpe_min": 0.15,
            "delta_profit_factor_min": 0.1,
            "cost_reduction_bps": 2.0,
        },
        confidence_model={
            "type": "logistic",
            "calibration_score": metrics["confidence_mean"],
        },
    )

    report_png = _render_equity_curve(combined_trades, symbol)
    report_summary = {
        "symbol": symbol,
        "run_id": run_id,
        "mode": mode,
        "metrics": metrics,
    }

    # Clean hyperparameters: remove NaN values and convert numpy types to native Python types
    def clean_for_json(obj: Any) -> Any:
        """Recursively clean object for JSON serialization."""
        import numpy as np
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items() if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))}
        elif isinstance(obj, (list, tuple)):
            return [clean_for_json(item) for item in obj if not (isinstance(item, float) and (np.isnan(item) or np.isinf(item)))]
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            val = float(obj)
            # Replace NaN and Inf with None
            if np.isnan(val) or np.isinf(val):
                return None
            return val
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return [clean_for_json(item) for item in obj.tolist()]
        elif obj is None:
            return None
        else:
            # Try to serialize, if it fails return string representation
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)
    
    cleaned_hyperparams = clean_for_json(hyperparams)
    cleaned_feature_importances = clean_for_json(feature_importances) if feature_importances else []
    
    config_payload = {
        "run_id": run_id,
        "symbol": symbol,
        "mode": mode,
        "feature_columns": feature_cols,
        "feature_importances": cleaned_feature_importances,
        "hyperparameters": cleaned_hyperparams,
        "training_window_days": settings.training.window_days,
        "walk_forward": settings.training.walk_forward.model_dump() if hasattr(settings.training.walk_forward, 'model_dump') else settings.training.walk_forward.dict() if hasattr(settings.training.walk_forward, 'dict') else {"train_days": settings.training.walk_forward.train_days, "test_days": settings.training.walk_forward.test_days, "min_trades": settings.training.walk_forward.min_trades},
    }

    bundle = None
    if published:
        _print_status("Creating model artifacts...", sym=symbol, level="PROGRESS")
        # Ensure all payloads are JSON-serializable
        try:
            config_json = json.dumps(config_payload, separators=(",", ":"), allow_nan=False)
        except (ValueError, TypeError) as e:
            logger.error("config_json_serialization_failed", error=str(e), message="Falling back to cleaned config")
            # Remove any remaining problematic values
            config_payload_clean = clean_for_json(config_payload)
            config_json = json.dumps(config_payload_clean, separators=(",", ":"), allow_nan=False)
        
        try:
            report_json = json.dumps(report_summary, separators=(",", ":"), allow_nan=False)
        except (ValueError, TypeError) as e:
            logger.error("report_json_serialization_failed", error=str(e), message="Falling back to cleaned report")
            report_summary_clean = clean_for_json(report_summary)
            report_json = json.dumps(report_summary_clean, separators=(",", ":"), allow_nan=False)
        
        bundle = ArtifactBundle(
            files={
                "model.bin": model_bytes,
                "config.json": config_json.encode("utf-8"),
                "metrics.json": metrics_payload.to_json().encode("utf-8"),
                "pilot_contract.json": pilot_contract.to_json().encode("utf-8"),
                "mechanic_contract.json": mechanic_contract.to_json().encode("utf-8"),
                "report_summary.json": report_json.encode("utf-8"),
                "report.png": report_png,
            }
        )
        _print_status(f"Artifacts created: {len(bundle.files)} files (model.bin, config.json, metrics.json, contracts, report)", sym=symbol, level="SUCCESS")
        
        # Export to Dropbox in organized structure if enabled
        # Note: feature_cols and cleaned_feature_importances are defined earlier in the function
        if settings.dropbox.enabled and settings.dropbox.access_token:
            try:
                from ..integrations.dropbox_sync import DropboxSync
                import tempfile
                
                dropbox_sync = DropboxSync(
                    access_token=settings.dropbox.access_token,
                    app_folder=settings.dropbox.app_folder,
                    enabled=True,
                    create_dated_folder=False,
                )
                
                # Save artifacts to temp files
                temp_dir = Path(tempfile.mkdtemp(prefix=f"huracan_{symbol}_"))
                artifact_files = {}
                
                # Save model
                model_path = temp_dir / "model.bin"
                model_path.write_bytes(model_bytes)
                artifact_files["model"] = model_path
                
                # Save metrics
                metrics_path = temp_dir / "training_metrics.json"
                metrics_path.write_bytes(metrics_payload.to_json().encode("utf-8"))
                artifact_files["metrics"] = metrics_path
                
                # Save config
                config_path = temp_dir / "config.json"
                config_path.write_bytes(config_json.encode("utf-8"))
                artifact_files["config"] = config_path
                
                # Save costs (create from costs object)
                costs_path = temp_dir / "costs.json"
                costs_json = json.dumps({
                    "maker_fee_bps": costs.maker_fee_bps,
                    "taker_fee_bps": costs.taker_fee_bps,
                    "slippage_bps": costs.slippage_bps,
                    "total_cost_bps": costs.total_cost_bps,
                    "symbol": symbol,
                }, indent=2)
                costs_path.write_bytes(costs_json.encode("utf-8"))
                artifact_files["costs"] = costs_path
                
                # Save contracts
                pilot_contract_path = temp_dir / "pilot_contract.json"
                pilot_contract_path.write_bytes(pilot_contract.to_json().encode("utf-8"))
                artifact_files["pilot_contract"] = pilot_contract_path
                
                mechanic_contract_path = temp_dir / "mechanic_contract.json"
                mechanic_contract_path.write_bytes(mechanic_contract.to_json().encode("utf-8"))
                artifact_files["mechanic_contract"] = mechanic_contract_path
                
                # Save report
                report_path = temp_dir / "report.png"
                report_path.write_bytes(report_png)
                artifact_files["report"] = report_path
                
                # Save report summary JSON
                report_summary_path = temp_dir / "report_summary.json"
                report_summary_path.write_bytes(report_json.encode("utf-8"))
                artifact_files["report_summary"] = report_summary_path
                
                # Save features metadata (if available)
                features_path = None
                if feature_cols:
                    features_metadata = {
                        "symbol": symbol,
                        "feature_columns": feature_cols,
                        "feature_importances": cleaned_feature_importances if 'cleaned_feature_importances' in locals() else {},
                        "num_features": len(feature_cols),
                        "run_date": run_date.isoformat(),
                    }
                    features_path = temp_dir / "features.json"
                    features_path.write_bytes(json.dumps(features_metadata, indent=2).encode("utf-8"))
                    artifact_files["features"] = features_path
                
                # Find candle data file
                candle_data_path = None
                symbol_safe = symbol.replace("/", "-")
                candles_dir = Path("data/candles")
                if candles_dir.exists():
                    # Look for most recent candle data file for this symbol
                    candle_files = sorted(candles_dir.glob(f"{symbol_safe}_*.parquet"), reverse=True)
                    if candle_files:
                        candle_data_path = candle_files[0]
                
                # Export all results
                export_results = dropbox_sync.export_coin_results(
                    symbol=symbol,
                    run_date=run_date,
                    model_path=artifact_files.get("model"),
                    metrics_path=artifact_files.get("metrics"),
                    costs_path=artifact_files.get("costs"),
                    features_path=features_path,
                    candle_data_path=candle_data_path,
                    additional_files={
                        "config": artifact_files.get("config"),
                        "pilot_contract": artifact_files.get("pilot_contract"),
                        "mechanic_contract": artifact_files.get("mechanic_contract"),
                        "report": artifact_files.get("report"),
                        "report_summary": artifact_files.get("report_summary"),
                    },
                )
                
                # Clean up temp files
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                
                files_uploaded = sum(1 for v in export_results.values() if v)
                _print_status(f"Results exported to Dropbox: {files_uploaded}/{len(export_results)} files", sym=symbol, level="SUCCESS")
                logger.info(
                    "coin_results_exported_to_dropbox",
                    symbol=symbol,
                    run_date=run_date.isoformat(),
                    files_uploaded=files_uploaded,
                    total_files=len(export_results),
                )
            except Exception as e:
                logger.warning("dropbox_export_failed_in_training", symbol=symbol, error=str(e))
                # Don't fail training if Dropbox export fails

    # Run RL training if enabled and database configured
    if dsn and settings.training.rl_agent.enabled:
        _print_status("Starting RL (Reinforcement Learning) training...", sym=symbol, level="PROGRESS")
        rl_metrics = _run_advanced_rl_training_for_symbol(
            symbol=symbol,
            settings=settings,
            exchange=exchange,
            dsn=dsn,
            lookback_days=settings.training.window_days,
        )
        if rl_metrics:
            _print_status(f"RL training complete: {rl_metrics.get('episodes', 0)} episodes trained", sym=symbol, level="SUCCESS")
            logger.info("rl_training_completed_for_symbol", symbol=symbol, rl_metrics=rl_metrics)
        else:
            _print_status("RL training skipped or failed", sym=symbol, level="WARN")
            logger.warning("rl_training_did_not_complete", symbol=symbol)
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"📊 [{symbol}] TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"  Status:           {'✅ PUBLISHED' if published else '❌ REJECTED'}")
    print(f"  Reason:           {reason}")
    print(f"  Total Trades:     {metrics['trades_oos']:,}")
    print(f"  Sharpe Ratio:     {metrics['sharpe']:.2f}")
    print(f"  Hit Rate:         {metrics['hit_rate']*100:.1f}%")
    print(f"  Profit Factor:    {metrics['profit_factor']:.2f}")
    print(f"  PnL (bps):        {metrics['pnl_bps']:.2f}")
    print(f"  Max Drawdown:     {metrics['max_dd_bps']:.2f} bps")
    if published:
        print(f"  Artifacts:        {len(bundle.files) if bundle else 0} files ready for upload")
    print(f"{'='*80}\n")

    return TrainingTaskResult(
        symbol=symbol,
        costs=costs,
        metrics=metrics,
        gate_results=gate_results,
        published=published,
        reason=reason,
        artifacts=bundle,
        model_id=str(uuid4()),
        run_id=run_id,
        pilot_contract=pilot_contract,
        mechanic_contract=mechanic_contract,
        metrics_payload=metrics_payload,
        model_params=hyperparams,
        feature_metadata={"columns": feature_cols, "feature_importances": feature_importances},
    )


def _run_advanced_rl_training_for_symbol(
    symbol: str,
    settings: EngineSettings,
    exchange: ExchangeClient,
    dsn: str,
    lookback_days: int = 365,
) -> Optional[Dict[str, Any]]:
    """Run advanced RL training for a single symbol if enabled.

    Uses enhanced RL pipeline or V2 pipeline based on settings.
    This executes shadow trading on historical data to learn patterns.
    Returns metrics dict if successful, None if skipped/failed.
    """
    if not settings.training.rl_agent.enabled:
        logger.info("rl_training_skipped", symbol=symbol, reason="rl_agent_disabled")
        return None

    if not settings.training.shadow_trading.enabled:
        logger.info("rl_training_skipped", symbol=symbol, reason="shadow_trading_disabled")
        return None

    # Get advanced config with safe fallback
    try:
        advanced_config = settings.training.advanced
        use_v2 = advanced_config.use_rl_v2_pipeline if advanced_config else False
        use_enhanced = advanced_config.use_enhanced_rl if advanced_config else False
        enable_advanced_rewards = advanced_config.enable_advanced_rewards if advanced_config else False
        enable_higher_order_features = advanced_config.enable_higher_order_features if advanced_config else False
        enable_granger_causality = advanced_config.enable_granger_causality if advanced_config else False
        enable_regime_prediction = advanced_config.enable_regime_prediction if advanced_config else False
        use_triple_barrier = advanced_config.use_triple_barrier if advanced_config else False
        use_meta_labeling = advanced_config.use_meta_labeling if advanced_config else False
    except (AttributeError, KeyError) as e:
        logger.warning("advanced_config_not_available_for_rl", symbol=symbol, error=str(e), message="Using default RL pipeline")
        advanced_config = None
        use_v2 = False
        use_enhanced = False
        enable_advanced_rewards = False
        enable_higher_order_features = False
        enable_granger_causality = False
        enable_regime_prediction = False
        use_triple_barrier = False
        use_meta_labeling = False
    
    logger.info(
        "===== STARTING ADVANCED RL TRAINING =====",
        symbol=symbol,
        lookback_days=lookback_days,
        operation="RL_SHADOW_TRADING",
        n_epochs=settings.training.rl_agent.n_epochs,
        epochs_per_update=settings.training.rl_agent.epochs_per_update,
        batch_size=settings.training.rl_agent.batch_size,
        use_v2_pipeline=use_v2,
        use_enhanced_pipeline=use_enhanced,
        enable_advanced_rewards=enable_advanced_rewards,
        enable_higher_order_features=enable_higher_order_features,
        enable_granger_causality=enable_granger_causality,
        enable_regime_prediction=enable_regime_prediction,
        use_triple_barrier=use_triple_barrier,
        use_meta_labeling=use_meta_labeling,
        message="RL training will process historical data and learn trading patterns",
    )

    rl_training_start_time = time_module.time()
    
    try:
        # Initialize appropriate RL pipeline
        if use_v2:
            # Use V2 pipeline (triple-barrier, meta-labeling, recency weighting)
            logger.info("using_rl_v2_pipeline", symbol=symbol)
            rl_pipeline = RLTrainingPipelineV2(
                settings=settings,
                dsn=dsn,
                use_v2_pipeline=True,
                trading_mode='scalp',  # Could be configurable
            )
        elif use_enhanced:
            # Use enhanced RL pipeline (Phase 1 improvements)
            logger.info("using_enhanced_rl_pipeline", symbol=symbol)
            rl_pipeline = EnhancedRLPipeline(
                settings=settings,
                dsn=dsn,
                enable_advanced_rewards=enable_advanced_rewards,
                enable_higher_order_features=enable_higher_order_features,
                enable_granger_causality=enable_granger_causality,
                enable_regime_prediction=enable_regime_prediction,
            )
        else:
            # Use basic RL pipeline
            logger.info("using_basic_rl_pipeline", symbol=symbol)
            rl_pipeline = RLTrainingPipeline(settings=settings, dsn=dsn)

        # Run shadow trading and learning on historical data
        logger.info(
            "rl_training_executing",
            symbol=symbol,
            pipeline_type="V2" if use_v2 else ("Enhanced" if use_enhanced else "Basic"),
            message="RL pipeline is now training on historical data - this may take several minutes",
        )
        
        metrics = rl_pipeline.train_on_symbol(
            symbol=symbol,
            exchange_client=exchange,
            lookback_days=lookback_days,
        )

        rl_training_time = time_module.time() - rl_training_start_time
        
        logger.info(
            "===== RL TRAINING COMPLETE =====",
            symbol=symbol,
            total_trades=metrics.get("total_trades", 0),
            winning_trades=metrics.get("winning_trades", 0),
            win_rate=metrics.get("win_rate", 0.0),
            avg_profit_gbp=metrics.get("avg_profit_gbp", 0.0),
            patterns_learned=metrics.get("patterns_learned", 0),
            training_time_seconds=rl_training_time,
            training_time_minutes=rl_training_time / 60.0,
            pipeline_type="V2" if use_v2 else ("Enhanced" if use_enhanced else "Basic"),
            message=f"RL training completed in {rl_training_time:.2f} seconds ({rl_training_time / 60.0:.2f} minutes)",
        )

        return metrics

    except Exception as exc:
        logger.exception(
            "rl_training_failed",
            symbol=symbol,
            error=str(exc),
            error_type=type(exc).__name__,
            pipeline_type="V2" if use_v2 else ("Enhanced" if use_enhanced else "Basic"),
        )
        return None
