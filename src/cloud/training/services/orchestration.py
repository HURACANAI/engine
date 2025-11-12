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


def _walk_forward_masks(ts: pd.Series, train_days: int, test_days: int) -> List[tuple[np.ndarray, np.ndarray]]:
    if ts.empty:
        return []
    splits: List[tuple[np.ndarray, np.ndarray]] = []
    start = ts.min()
    end = ts.max()
    current = start
    while True:
        train_start = current
        train_end = train_start + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)
        if test_end > end:
            break
        train_mask = (ts >= train_start) & (ts < train_end)
        test_mask = (ts >= train_end) & (ts < test_end)
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            current += timedelta(days=test_days)
            if current >= end:
                break
            continue
        splits.append((train_mask.to_numpy(), test_mask.to_numpy()))
        current += timedelta(days=test_days)
        if current + timedelta(days=train_days + test_days) > end + timedelta(days=1):
            break
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
    if equity.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max
    return float(drawdown.min())


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
    sharpe = (mean_pnl / std_pnl * np.sqrt(pnl.size)) if std_pnl > 0 else 0.0
    positives = pnl[pnl > 0].sum()
    negatives = pnl[pnl < 0].sum()
    profit_factor = float(positives / max(abs(negatives), 1e-9))
    hit_rate = float((pnl > 0).mean()) if pnl.size else 0.0
    max_dd = _max_drawdown(trades["equity_curve"].to_numpy())
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
    ) -> None:
        self._settings = settings
        self._exchange = exchange_client
        self._universe_selector = universe_selector
        self._registry = model_registry
        self._notifier = notifier
        self._artifact_publisher = artifact_publisher
        self._telegram_monitor = telegram_monitor
        self._learning_tracker = learning_tracker
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
                    # Add timeout to prevent indefinite hanging (30 minutes per coin)
                    # Note: This will raise ray.exceptions.GetTimeoutError if timeout is exceeded
                    logger.info(
                        "waiting_for_training_task",
                        symbol=symbol,
                        timeout_seconds=1800,
                        message="Task may take time downloading historical data from exchange",
                    )
                    result = ray.get(task, timeout=1800.0)  # 30 minutes timeout
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
                        timeout_seconds=1800,
                        message="Task exceeded 30 minute timeout - likely stuck on API call or data download",
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
                                "error": "Task exceeded 30 minute timeout - likely stuck on API call or data download",
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
                        reason="training_timeout: Task exceeded 30 minute timeout",
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
                print(f"❌ [{result.symbol}] Failed to publish artifacts: {e}")
                logger.error("artifact_publish_failed", symbol=result.symbol, error=str(e), error_type=type(e).__name__)
        
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
    
    logger.info(
        "downloading_historical_data",
        symbol=symbol,
        start_at=start_at.isoformat(),
        end_at=end_at.isoformat(),
        window_days=settings.training.window_days,
        message="This may take several minutes depending on data size",
    )
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
            label_config = ScalpLabelConfig()  # Use scalp config for 4-minute horizon
            cost_estimator = CostEstimator(
                taker_fee_bps=taker_fee,
                spread_bps=spread_bps,
                slippage_bps=costs.slippage_bps,
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
            
            # Convert labeled trades to DataFrame format expected by training
            if not labeled_trades:
                logger.warning(
                    "no_labeled_trades_from_v2_labeling",
                    symbol=symbol,
                    message="V2 labeling produced no trades",
                )
                dataset = pd.DataFrame()
            else:
                # Convert LabeledTrade objects to DataFrame
                trade_data = []
                for trade in labeled_trades:
                    # Get the original row index for this trade
                    # We'll need to match trade.entry_time to feature_frame timestamps
                    trade_data.append({
                        "net_edge_bps": trade.pnl_net_bps,
                        "edge_confidence": 1.0 if trade.meta_label == 1 else 0.0,
                        "trade_entry_time": trade.entry_time,
                        "trade_exit_time": trade.exit_time,
                        "trade_pnl_bps": trade.pnl_net_bps,
                    })
                
                # For now, use basic labeling as fallback if V2 produces no trades
                # This is a temporary workaround - V2 labeling needs more integration
                logger.warning(
                    "v2_labeling_integration_incomplete",
                    symbol=symbol,
                    message="V2 labeling not fully integrated with training pipeline, using basic labeling",
                )
                use_v2_labeling = False  # Fall back to basic labeling
                
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
            label_builder = LabelBuilder(LabelingConfig(horizon_minutes=4))
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
        # V2 labeling path (if we implement full integration)
        dataset = pd.DataFrame()
    
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

    excluded = {"ts", "open", "high", "low", "close", "volume", "vwap", "net_edge_bps"}
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
        "verbose": model_config.verbose,
    }

    ts_series = dataset["ts"]
    
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
    
    splits = _walk_forward_masks(ts_series, settings.training.walk_forward.train_days, settings.training.walk_forward.test_days)
    cost_threshold = cost_model.recommended_edge_threshold(costs)
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
        X_train = dataset.loc[train_mask, feature_cols]
        y_train = dataset.loc[train_mask, "net_edge_bps"]
        
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
        X_train_split = X_train.iloc[:train_size]
        y_train_split = y_train.iloc[:train_size]
        X_val_split = X_train.iloc[train_size:]
        y_val_split = y_train.iloc[train_size:]
        
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
            
            # Train ensemble
            ensemble_trainer, ensemble_results = train_multi_model_ensemble(
                X_train=X_train_split,
                y_train=y_train_split,
                X_val=X_val_split,
                y_val=y_val_split,
                regimes=None,  # Could add regime detection here
                techniques=ensemble_techniques,
                ensemble_method=ensemble_method,
                is_classification=False,
            )
            
            ensemble_trainers.append(ensemble_trainer)
            
            # Get best model for predictions (or use ensemble)
            X_test = dataset.loc[test_mask, feature_cols]
            if X_test.empty:
                continue
            
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
            
            X_test = dataset.loc[test_mask, feature_cols]
            if X_test.empty:
                continue
            predictions = model.predict(X_test)
            confidence = dataset.loc[test_mask, "edge_confidence"].to_numpy()
        
        split_training_time = time_module.time() - split_start_time
        
        y_test = dataset.loc[test_mask, "net_edge_bps"].to_numpy()
        timestamps = dataset.loc[test_mask, "ts"].to_numpy()
        
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
        
        # Print split completion with metrics
        if not trades.empty:
            avg_pnl = trades["pnl_bps"].mean()
            win_rate = (trades["pnl_bps"] > 0).mean() * 100
            _print_status(f"Split {split_idx + 1}/{len(splits)} complete: {len(trades)} trades, Avg PnL={avg_pnl:.2f} bps, Win Rate={win_rate:.1f}% ({split_training_time:.1f}s)", sym=symbol, level="SUCCESS")
        else:
            _print_status(f"Split {split_idx + 1}/{len(splits)} complete: 0 trades generated ({split_training_time:.1f}s)", sym=symbol, level="WARN")
    
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
    
    # Print trade summary
    total_trades = len(combined_trades) if not combined_trades.empty else 0
    _print_status(f"Walk-forward training complete: {len(splits)} splits, {total_trades:,} total trades, {total_training_time/60:.1f} minutes", sym=symbol, level="SUCCESS")
    
    if not combined_trades.empty:
        winning_trades = (combined_trades["pnl_bps"] > 0).sum()
        losing_trades = (combined_trades["pnl_bps"] <= 0).sum()
        total_pnl = combined_trades["pnl_bps"].sum()
        avg_pnl = combined_trades["pnl_bps"].mean()
        _print_status(f"Trade Summary: {total_trades:,} trades ({winning_trades:,} wins, {losing_trades:,} losses), Total PnL={total_pnl:.2f} bps, Avg={avg_pnl:.2f} bps", sym=symbol, level="INFO")
    else:
        _print_status("WARNING: No trades generated during walk-forward validation", sym=symbol, level="WARN")
    
    metrics = _compute_metrics(combined_trades, costs.total_costs_bps, len(dataset))
    
    # Print key metrics
    _print_status(f"Model Performance: Sharpe={metrics.get('sharpe', 0):.2f}, Hit Rate={metrics.get('hit_rate', 0)*100:.1f}%, PnL={metrics.get('pnl_bps', 0):.2f} bps", sym=symbol, level="INFO")
    
    # Brain Library integration: Store final model metrics after all splits
    if brain_training:
        try:
            logger.info("training_final_model_for_brain_integration", symbol=symbol)
            final_model_start_time = time_module.time()
            
            # Train final model on all data - use ensemble if enabled
            if use_ensemble:
                # Train ensemble for brain integration
                brain_ensemble_trainer, brain_ensemble_results = train_multi_model_ensemble(
                    X_train=dataset[feature_cols],
                    y_train=dataset["net_edge_bps"],
                    X_val=None,
                    y_val=None,
                    regimes=None,
                    techniques=ensemble_techniques,
                    ensemble_method=ensemble_method,
                    is_classification=False,
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
            
            # Use last test split for evaluation
            if splits:
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
        metrics.update({"recommended_edge_threshold_bps": cost_threshold, "validation_window": f"{dataset['ts'].min().date()} to {dataset['ts'].max().date()}"})
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

    median_dd = median([abs(entry.get("max_dd_bps", 0.0)) for entry in history]) if history else abs(metrics["max_dd_bps"])
    median_hit = median([entry.get("hit_rate", 0.0) for entry in history]) if history else metrics["hit_rate"]
    gate_results = {
        "sharpe_pass": metrics["sharpe"] >= 0.7,
        "profit_factor_pass": metrics["profit_factor"] >= 1.1,
        "max_dd_pass": abs(metrics["max_dd_bps"]) <= 1.2 * median_dd if median_dd else True,
        "hit_rate_pass": metrics["hit_rate"] >= max(median_hit - 0.01, 0.0),
        "trades_pass": metrics["trades_oos"] >= settings.training.walk_forward.min_trades,
    }
    published = all(gate_results.values())
    failed = [name for name, passed in gate_results.items() if not passed]
    reason = "all_pass" if published else "gate_failure:" + ",".join(failed)
    
    # Print gate results
    print(f"\n{'='*80}")
    print(f"📊 [{symbol}] GATE RESULTS")
    print(f"{'='*80}")
    print(f"  Sharpe Ratio:      {metrics['sharpe']:.2f} {'✅ PASS' if gate_results['sharpe_pass'] else '❌ FAIL'} (threshold: 0.7)")
    print(f"  Profit Factor:     {metrics['profit_factor']:.2f} {'✅ PASS' if gate_results['profit_factor_pass'] else '❌ FAIL'} (threshold: 1.1)")
    print(f"  Hit Rate:          {metrics['hit_rate']*100:.1f}% {'✅ PASS' if gate_results['hit_rate_pass'] else '❌ FAIL'} (threshold: {max(median_hit - 0.01, 0.0)*100:.1f}%)")
    print(f"  Max Drawdown:      {metrics['max_dd_bps']:.2f} bps {'✅ PASS' if gate_results['max_dd_pass'] else '❌ FAIL'} (threshold: {1.2 * median_dd:.2f} bps)")
    print(f"  Trades:            {metrics['trades_oos']:,} {'✅ PASS' if gate_results['trades_pass'] else '❌ FAIL'} (threshold: {settings.training.walk_forward.min_trades:,})")
    print(f"{'='*80}")
    if published:
        print(f"✅ [{symbol}] ALL GATES PASSED - Model will be published")
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
        
        # Train final ensemble on all data
        final_ensemble_trainer, final_ensemble_results = train_multi_model_ensemble(
            X_train=dataset[feature_cols],
            y_train=dataset["net_edge_bps"],
            X_val=None,  # Use all data for final model
            y_val=None,
            regimes=None,
            techniques=ensemble_techniques,
            ensemble_method=ensemble_method,
            is_classification=False,
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
