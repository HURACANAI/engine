"""Orchestration helpers integrating APScheduler and Ray."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timezone, timedelta
from io import BytesIO
from statistics import median
from typing import Any, Dict, List, Optional
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import ray
import structlog
from lightgbm import LGBMRegressor

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
from shared.contracts.mechanic import MechanicContract
from shared.contracts.metrics import MetricsPayload
from shared.contracts.pilot import PilotContract
from shared.features.recipe import FeatureRecipe


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
        return pd.DataFrame(columns=["timestamp", "pnl_bps", "prediction_bps", "confidence", "equity_curve"])
    trades = pd.DataFrame(
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
    ) -> None:
        self._settings = settings
        self._exchange = exchange_client
        self._universe_selector = universe_selector
        self._registry = model_registry
        self._notifier = notifier
        self._artifact_publisher = artifact_publisher
        self._run_date = datetime.now(tz=timezone.utc).date()

    def run(self) -> List[TrainingTaskResult]:
        universe = self._universe_selector.select()
        rows = list(universe.iter_rows(named=True))
        logger.info("universe_selected", count=len(rows), symbols=[row["symbol"] for row in rows])
        tasks = [
            self._submit_task(
                row=row,
                history=self._registry.fetch_recent_metrics(row["symbol"], limit=10),
            )
            for row in rows
        ]
        ray_results = ray.get(tasks) if tasks else []
        results: List[TrainingTaskResult] = []
        for result in ray_results:
            results.append(result)
            self._finalize_result(result)
        self._notifier.send_summary(results, run_date=self._run_date)
        return results

    def _submit_task(self, row: Dict[str, Any], history: List[Dict[str, Any]]) -> "ray.ObjectRef[TrainingTaskResult]":
        symbol = row["symbol"]
        return _train_symbol.remote(  # type: ignore[attr-defined]
            symbol,
            self._settings.model_dump(mode="python"),
            self._exchange.exchange_id,
            row,
            history,
            self._run_date.isoformat(),
            self._settings.mode,
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


@ray.remote  # type: ignore[misc]
def _train_symbol(
    symbol: str,
    raw_settings: Dict[str, Any],
    exchange_id: str,
    universe_row: Dict[str, Any],
    history: List[Dict[str, Any]],
    run_date_str: str,
    mode: str,
) -> TrainingTaskResult:
    settings = EngineSettings.model_validate(raw_settings)
    run_date = date.fromisoformat(run_date_str)
    credentials = settings.exchange.credentials.get(exchange_id, {})
    exchange = ExchangeClient(exchange_id, credentials=credentials, sandbox=settings.exchange.sandbox)
    quality_suite = DataQualitySuite()
    loader = CandleDataLoader(exchange_client=exchange, quality_suite=quality_suite)
    start_at, end_at = _window_bounds(run_date, settings.scheduler.daily_run_time_utc, settings.training.window_days)
    query = CandleQuery(symbol=symbol, start_at=start_at, end_at=end_at)
    raw_frame = loader.load(query)
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

    combined_trades = pd.concat(oos_trades, ignore_index=True) if oos_trades else pd.DataFrame(columns=["timestamp", "pnl_bps", "prediction_bps", "confidence", "equity_curve"])
    metrics = _compute_metrics(combined_trades, costs.total_costs_bps, len(dataset))

    if metrics["trades_oos"] < settings.training.walk_forward.min_trades:
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
