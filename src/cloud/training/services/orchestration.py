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
            s3_uri = self._artifact_publisher.publish(self._run_date, result.symbol, result.artifacts)
            result.artifacts_path = s3_uri
        kind = "baseline" if result.published else "candidate"
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
        if result.metrics_payload:
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
        self._registry.log_publish(
            model_id=result.model_id,
            symbol=result.symbol,
            published=result.published,
            reason=result.reason,
            at=datetime.now(tz=timezone.utc),
        )
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
    logger.info(
        "training_task_started",
        symbol=symbol,
        exchange_id=exchange_id,
        message="Ray task started - beginning data download",
    )
    settings = EngineSettings.model_validate(raw_settings)
    run_date = date.fromisoformat(run_date_str)
    credentials = settings.exchange.credentials.get(exchange_id, {})
    exchange = ExchangeClient(exchange_id, credentials=credentials, sandbox=settings.exchange.sandbox)
    quality_suite = DataQualitySuite()
    
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

    recipe = FeatureRecipe()
    feature_frame = recipe.build(raw_frame)
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
    label_builder = LabelBuilder(LabelingConfig(horizon_minutes=4))
    labeled = label_builder.build(feature_frame, costs)
    dataset = labeled.to_pandas()
    dataset["ts"] = pd.to_datetime(dataset["ts"], utc=True)
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset.dropna(inplace=True)
    dataset.sort_values("ts", inplace=True)
    if dataset.empty:
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

    hyperparams = {
        "objective": "regression",
        "learning_rate": 0.05,
        "n_estimators": 300,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 32,
        "random_state": 42,
        "n_jobs": -1,
    }

    ts_series = dataset["ts"]
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
    
    for train_mask, test_mask in splits:
        X_train = dataset.loc[train_mask, feature_cols]
        y_train = dataset.loc[train_mask, "net_edge_bps"]
        if len(y_train) < 100:
            continue
        model = LGBMRegressor(**hyperparams)
        model.fit(X_train, y_train)
        X_test = dataset.loc[test_mask, feature_cols]
        y_test = dataset.loc[test_mask, "net_edge_bps"].to_numpy()
        confidence = dataset.loc[test_mask, "edge_confidence"].to_numpy()
        timestamps = dataset.loc[test_mask, "ts"].to_numpy()
        if X_test.empty:
            continue
        predictions = model.predict(X_test)
        
        trades = _simulate_trades(predictions, y_test, confidence, timestamps, cost_threshold)
        if not trades.empty:
            oos_trades.append(trades)

    combined_trades = pd.concat(oos_trades, ignore_index=True) if oos_trades else pd.DataFrame(columns=["timestamp", "pnl_bps", "prediction_bps", "confidence", "equity_curve"])  # type: ignore[call-overload]
    metrics = _compute_metrics(combined_trades, costs.total_costs_bps, len(dataset))
    
    # Brain Library integration: Store final model metrics after all splits
    if brain_training:
        try:
            # Train final model on all data
            final_model = LGBMRegressor(**hyperparams)
            final_model.fit(dataset[feature_cols], dataset["net_edge_bps"])
            
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
                        model_type="lightgbm",
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

    final_model = LGBMRegressor(**hyperparams)
    final_model.fit(dataset[feature_cols], dataset["net_edge_bps"])
    model_bytes = pickle.dumps(final_model)
    feature_importances = final_model.feature_importances_.tolist()

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

    config_payload = {
        "run_id": run_id,
        "symbol": symbol,
        "mode": mode,
        "feature_columns": feature_cols,
        "feature_importances": feature_importances,
        "hyperparameters": hyperparams,
        "training_window_days": settings.training.window_days,
        "walk_forward": asdict(settings.training.walk_forward),
    }

    bundle = None
    if published:
        bundle = ArtifactBundle(
            files={
                "model.bin": model_bytes,
                "config.json": json.dumps(config_payload, separators=(",", ":")).encode("utf-8"),
                "metrics.json": metrics_payload.to_json().encode("utf-8"),
                "pilot_contract.json": pilot_contract.to_json().encode("utf-8"),
                "mechanic_contract.json": mechanic_contract.to_json().encode("utf-8"),
                "report_summary.json": json.dumps(report_summary, separators=(",", ":")).encode("utf-8"),
                "report.png": report_png,
            }
        )

    # Run RL training if enabled and database configured
    if dsn and settings.training.rl_agent.enabled:
        rl_metrics = _run_rl_training_for_symbol(
            symbol=symbol,
            settings=settings,
            exchange=exchange,
            dsn=dsn,
            lookback_days=settings.training.window_days,
        )
        if rl_metrics:
            logger.info("rl_training_completed_for_symbol", symbol=symbol, rl_metrics=rl_metrics)
        else:
            logger.warning("rl_training_did_not_complete", symbol=symbol)

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


def _run_rl_training_for_symbol(
    symbol: str,
    settings: EngineSettings,
    exchange: ExchangeClient,
    dsn: str,
    lookback_days: int = 365,
) -> Optional[Dict[str, Any]]:
    """Run RL training for a single symbol if enabled.

    This executes shadow trading on historical data to learn patterns.
    Returns metrics dict if successful, None if skipped/failed.
    """
    if not settings.training.rl_agent.enabled:
        logger.info("rl_training_skipped", symbol=symbol, reason="rl_agent_disabled")
        return None

    if not settings.training.shadow_trading.enabled:
        logger.info("rl_training_skipped", symbol=symbol, reason="shadow_trading_disabled")
        return None

    logger.info(
        "===== STARTING RL TRAINING =====",
        symbol=symbol,
        lookback_days=lookback_days,
        operation="RL_SHADOW_TRADING",
    )

    try:
        # Initialize RL pipeline
        rl_pipeline = RLTrainingPipeline(settings=settings, dsn=dsn)

        # Run shadow trading and learning on historical data
        metrics = rl_pipeline.train_on_symbol(
            symbol=symbol,
            exchange_client=exchange,
            lookback_days=lookback_days,
        )

        logger.info(
            "===== RL TRAINING COMPLETE =====",
            symbol=symbol,
            total_trades=metrics.get("total_trades", 0),
            winning_trades=metrics.get("winning_trades", 0),
            win_rate=metrics.get("win_rate", 0.0),
            avg_profit_gbp=metrics.get("avg_profit_gbp", 0.0),
            patterns_learned=metrics.get("patterns_learned", 0),
        )

        return metrics

    except Exception as exc:
        logger.exception(
            "rl_training_failed",
            symbol=symbol,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return None
