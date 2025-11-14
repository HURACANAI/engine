"""
Meta Tuner for Huracan trading engine.

This module provides a `run_meta_tuning()` entrypoint that:
- Reads model performance metrics and realized trade stats from the Brain Library
  and trade logs.
- Tunes per-symbol, per-mode configuration parameters (edge thresholds, TP/SL
  levels, and position size multipliers).
- Persists those dynamic parameters to Postgres in the `symbol_mode_config` table.

Discovered integration points (for future readers):
- Triple-barrier + v2 labeling:
  - `cloud.engine.labeling.triple_barrier.TripleBarrierLabeler`
  - `cloud.engine.labeling.meta_labeler.MetaLabeler`
  - Training-time integration: `cloud.training.services.orchestration._train_symbol`
    (v2 labeling and `net_edge_bps` construction).
- Walk-forward training and splits:
  - `cloud.training.services.orchestration._train_symbol`
  - `cloud.engine.walk_forward.WalkForwardEngine` and
    `cloud.training.validation.walk_forward_purged.WalkForwardPurgedValidator`
- Brain Library and Postgres schema:
  - `cloud.training.brain.brain_library.BrainLibrary`
  - Uses `DatabaseConnectionPool` in `cloud.training.database.pool`
- Hamilton live trading:
  - `cloud.training.hamilton.interface.HamiltonModelLoader` and
    `cloud.training.hamilton.interface.HamiltonRankingTable`
  - Live execution and sizing logic integrate via the Hamilton interface and
    execution/risk modules under `cloud.training.execution` and `cloud.training.risk`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import structlog

from ..brain.brain_library import BrainLibrary

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Meta-tuning configuration (all magic numbers live here for easy tweaking)
# ---------------------------------------------------------------------------

@dataclass
class ModeTuningConfig:
    """Per-mode tuning targets and clamps."""

    target_min_trades_per_day: float
    target_max_trades_per_day: float
    sharpe_floor_increase: float
    sharpe_floor_decrease: float
    max_edge_bps: float = 50.0
    min_edge_bps: float = 0.0
    edge_step_bps: float = 1.0
    size_increase_factor: float = 1.1
    size_decrease_factor: float = 0.5
    max_size_multiplier: float = 3.0
    min_size_multiplier: float = 0.0


# Default configs per trading mode
DEFAULT_MODE_CONFIGS: Dict[str, ModeTuningConfig] = {
    "scalp": ModeTuningConfig(
        target_min_trades_per_day=10.0,
        target_max_trades_per_day=80.0,
        sharpe_floor_increase=0.1,
        sharpe_floor_decrease=-0.2,
    ),
    "swing": ModeTuningConfig(
        target_min_trades_per_day=1.0,
        target_max_trades_per_day=10.0,
        sharpe_floor_increase=0.2,
        sharpe_floor_decrease=0.0,
    ),
}

# Lookback windows
METRICS_LOOKBACK_DAYS = 30
TRADES_LOOKBACK_DAYS = 30


# ---------------------------------------------------------------------------
# Symbol/mode config DTO
# ---------------------------------------------------------------------------

@dataclass
class SymbolModeConfig:
    symbol: str
    mode: str
    edge_threshold_bps: float
    tp_atr_k: float
    sl_atr_k: float
    min_tp_bps: float
    size_multiplier: float
    tp_bps: float = 0.0
    sl_bps: float = 0.0
    fee_bps: float = 0.0
    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    window_days: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


def _config_from_dict(data: Dict[str, Any], symbol: str, mode: str) -> SymbolModeConfig:
    return SymbolModeConfig(
        symbol=symbol,
        mode=mode,
        edge_threshold_bps=float(data.get("edge_threshold_bps", 5.0)),
        tp_atr_k=float(data.get("tp_atr_k", 2.0)),
        sl_atr_k=float(data.get("sl_atr_k", 1.0)),
        min_tp_bps=float(data.get("min_tp_bps", 10.0)),
        size_multiplier=float(data.get("size_multiplier", 1.0)),
        tp_bps=float(data.get("tp_bps", 0.0)),
        sl_bps=float(data.get("sl_bps", 0.0)),
        fee_bps=float(data.get("fee_bps", 0.0)),
        spread_bps=float(data.get("spread_bps", 0.0)),
        slippage_bps=float(data.get("slippage_bps", 0.0)),
        window_days=data.get("window_days"),
        created_at=data.get("created_at"),
        updated_at=data.get("updated_at"),
    )


def _load_symbol_mode_config(
    brain: BrainLibrary, symbol: str, mode: str
) -> Optional[SymbolModeConfig]:
    row = brain.get_symbol_mode_config(symbol, mode)
    if not row:
        return None
    return _config_from_dict(row, symbol, mode)


def _upsert_symbol_mode_config(brain: BrainLibrary, cfg: SymbolModeConfig) -> None:
    brain.upsert_symbol_mode_config(
        cfg.symbol,
        cfg.mode,
        {
            "edge_threshold_bps": cfg.edge_threshold_bps,
            "tp_atr_k": cfg.tp_atr_k,
            "sl_atr_k": cfg.sl_atr_k,
            "min_tp_bps": cfg.min_tp_bps,
            "size_multiplier": cfg.size_multiplier,
            "tp_bps": cfg.tp_bps,
            "sl_bps": cfg.sl_bps,
            "fee_bps": cfg.fee_bps,
            "spread_bps": cfg.spread_bps,
            "slippage_bps": cfg.slippage_bps,
            "window_days": cfg.window_days,
        },
    )


# ---------------------------------------------------------------------------
# Metric readers (placeholders wired for future extension)
# ---------------------------------------------------------------------------


def _get_recent_trade_stats(
    brain: BrainLibrary, symbol: str, mode: str, lookback_days: int
) -> Dict[str, float]:
    """
    Return realized trade statistics for a symbol/mode over the lookback window.

    Expected keys:
        trades_per_day
        avg_fee_bps
        avg_slippage_bps
        avg_spread_bps

    Implementation:
        - Uses symbol_mode_trade_stats aggregated by training/evaluation code.
        - Falls back to zeros if no rows are found for the lookback window.
    """
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=lookback_days)

    total_trades = 0
    days_covered = 0

    with brain._get_connection() as conn:  # type: ignore[attr-defined]
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT stat_date, trades, trades_per_day
                  FROM symbol_mode_trade_stats
                 WHERE symbol = %s
                   AND mode = %s
                   AND stat_date BETWEEN %s AND %s
                """,
                (symbol, mode, start_date, end_date),
            )
            for row in cur.fetchall():
                stat_date, trades, trades_per_day = row
                total_trades += int(trades)
                days_covered += 1

    trades_per_day = (
        total_trades / float(days_covered) if days_covered > 0 else 0.0
    )

    logger.info(
        "recent_trade_stats_loaded",
        symbol=symbol,
        mode=mode,
        trades_per_day=trades_per_day,
        total_trades=total_trades,
        days_covered=days_covered,
        lookback_days=lookback_days,
    )

    return {
        "trades_per_day": trades_per_day,
        # Placeholders for cost components until we persist realized costs per trade.
        "avg_fee_bps": 0.0,
        "avg_slippage_bps": 0.0,
        "avg_spread_bps": 0.0,
    }


def _get_recent_volatility_stats(
    brain: BrainLibrary, symbol: str, mode: str, lookback_days: int
) -> Dict[str, float]:
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=lookback_days)
    rows = brain.load_symbol_mode_stats(
        "symbol_mode_vol_stats", symbol, mode, start_date, end_date
    )
    if not rows:
        return {"atr_bps": 0.0, "realized_vol_bps": 0.0}

    atr_values = [float(row["atr_bps"]) for row in rows if row.get("atr_bps") is not None]
    vol_values = [
        float(row["realized_vol_bps"]) for row in rows if row.get("realized_vol_bps") is not None
    ]
    atr_bps = sum(atr_values) / len(atr_values) if atr_values else 0.0
    realized_vol_bps = sum(vol_values) / len(vol_values) if vol_values else 0.0
    logger.info(
        "recent_volatility_stats_loaded",
        symbol=symbol,
        mode=mode,
        atr_bps=atr_bps,
        realized_vol_bps=realized_vol_bps,
        samples=len(rows),
    )
    return {"atr_bps": atr_bps, "realized_vol_bps": realized_vol_bps}


def _get_recent_cost_stats(
    brain: BrainLibrary, symbol: str, mode: str, lookback_days: int
) -> Dict[str, float]:
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=lookback_days)
    rows = brain.load_symbol_mode_stats(
        "symbol_mode_cost_stats", symbol, mode, start_date, end_date
    )
    if not rows:
        return {
            "avg_fee_bps": 0.0,
            "avg_slippage_bps": 0.0,
            "avg_spread_bps": 0.0,
            "median_total_costs_bps": 0.0,
        }
    fee = sum(float(row["avg_fee_bps"]) for row in rows) / len(rows)
    slippage = sum(float(row["avg_slippage_bps"]) for row in rows) / len(rows)
    spread = sum(float(row["avg_spread_bps"]) for row in rows) / len(rows)
    median_costs = sum(float(row["median_total_costs_bps"]) for row in rows) / len(rows)
    logger.info(
        "recent_cost_stats_loaded",
        symbol=symbol,
        mode=mode,
        avg_fee_bps=fee,
        avg_slippage_bps=slippage,
        avg_spread_bps=spread,
        median_total_costs_bps=median_costs,
        samples=len(rows),
    )
    return {
        "avg_fee_bps": fee,
        "avg_slippage_bps": slippage,
        "avg_spread_bps": spread,
        "median_total_costs_bps": median_costs,
    }


def _get_recent_model_metrics(
    brain: BrainLibrary, symbol: str, mode: str, lookback_days: int
) -> Dict[str, float]:
    """
    Return recent model performance metrics from model_metrics/model_comparisons.

    Expected keys:
        sharpe
        hit_rate
        max_drawdown

    This implementation uses `model_metrics` as a coarse proxy; more detailed
    per-mode metrics can be added later by extending training-time writes.
    """
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=lookback_days)

    sharpe_values: List[float] = []
    hit_values: List[float] = []
    max_dd_values: List[float] = []

    with brain._get_connection() as conn:  # type: ignore[attr-defined]
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT sharpe_ratio, hit_ratio, max_drawdown
                  FROM model_metrics
                 WHERE symbol = %s
                   AND evaluation_date BETWEEN %s AND %s
                """,
                (symbol, start_date, end_date),
            )
            for row in cur.fetchall():
                if row[0] is not None:
                    sharpe_values.append(float(row[0]))
                if row[1] is not None:
                    hit_values.append(float(row[1]))
                if row[2] is not None:
                    max_dd_values.append(float(row[2]))

    sharpe = sum(sharpe_values) / len(sharpe_values) if sharpe_values else 0.0
    hit_rate = sum(hit_values) / len(hit_values) if hit_values else 0.0
    max_dd = min(max_dd_values) if max_dd_values else 0.0

    logger.info(
        "recent_model_metrics_loaded",
        symbol=symbol,
        mode=mode,
        sharpe=sharpe,
        hit_rate=hit_rate,
        max_drawdown=max_dd,
        lookback_days=lookback_days,
    )

    return {
        "sharpe": sharpe,
        "hit_rate": hit_rate,
        "max_drawdown": max_dd,
    }


def _get_v2_label_distribution(
    brain: BrainLibrary, symbol: str, mode: str, lookback_days: int
) -> Dict[str, float]:
    """
    Return distribution stats for net_edge_bps from the latest training run.

    NOTE: At the moment, detailed per-mode net_edge_bps distributions are not
    materialized in Brain Library tables. This function is a placeholder that
    should be wired to the training pipeline when that data is persisted.
    """
    logger.info(
        "net_edge_distribution_placeholder_used",
        symbol=symbol,
        mode=mode,
        lookback_days=lookback_days,
    )
    return {
        "median_net_edge_bps": 0.0,
        "p90_net_edge_bps": 0.0,
        "max_net_edge_bps": 0.0,
    }


# ---------------------------------------------------------------------------
# Tuning rules
# ---------------------------------------------------------------------------


def _tune_edge_threshold(
    symbol: str,
    mode: str,
    cfg: SymbolModeConfig,
    mode_cfg: ModeTuningConfig,
    trades_per_day: float,
    sharpe: float,
) -> float:
    old_edge = cfg.edge_threshold_bps
    new_edge = old_edge

    if trades_per_day < mode_cfg.target_min_trades_per_day and sharpe >= mode_cfg.sharpe_floor_increase:
        new_edge = max(
            mode_cfg.min_edge_bps,
            old_edge - mode_cfg.edge_step_bps,
        )
        if new_edge != old_edge:
            logger.info(
                "edge_threshold_tuned",
                symbol=symbol,
                mode=mode,
                old_edge=old_edge,
                new_edge=new_edge,
                reason="low_trade_count",
                trades_per_day=trades_per_day,
                sharpe=sharpe,
            )
    elif trades_per_day > mode_cfg.target_max_trades_per_day and sharpe <= mode_cfg.sharpe_floor_decrease:
        new_edge = min(
            mode_cfg.max_edge_bps,
            old_edge + mode_cfg.edge_step_bps,
        )
        if new_edge != old_edge:
            logger.info(
                "edge_threshold_tuned",
                symbol=symbol,
                mode=mode,
                old_edge=old_edge,
                new_edge=new_edge,
                reason="high_trade_count_low_sharpe",
                trades_per_day=trades_per_day,
                sharpe=sharpe,
            )

    return new_edge


def _tune_size_multiplier(
    symbol: str,
    mode: str,
    cfg: SymbolModeConfig,
    mode_cfg: ModeTuningConfig,
    sharpe: float,
    max_drawdown: float,
) -> float:
    old_size = cfg.size_multiplier
    new_size = old_size

    # Simple risk-based adjustment: grow when Sharpe > 0 and DD acceptable, cut when Sharpe < 0.
    if sharpe > mode_cfg.sharpe_floor_increase and max_drawdown >= -5.0:
        new_size = min(
            mode_cfg.max_size_multiplier,
            old_size * mode_cfg.size_increase_factor,
        )
        if new_size != old_size:
            logger.info(
                "size_multiplier_updated",
                symbol=symbol,
                mode=mode,
                old_size=old_size,
                new_size=new_size,
                reason="sharpe_positive_drawdown_ok",
                sharpe=sharpe,
                max_drawdown=max_drawdown,
            )
    elif sharpe < 0.0:
        new_size = max(
            mode_cfg.min_size_multiplier,
            old_size * mode_cfg.size_decrease_factor,
        )
        if new_size != old_size:
            logger.info(
                "size_multiplier_updated",
                symbol=symbol,
                mode=mode,
                old_size=old_size,
                new_size=new_size,
                reason="sharpe_negative",
                sharpe=sharpe,
                max_drawdown=max_drawdown,
            )

    return new_size


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_meta_tuning(brain: BrainLibrary, symbols_and_modes: List[Tuple[str, str]]) -> None:
    """
    Run meta-parameter tuning for the given list of (symbol, mode) pairs.

    This function is designed to be called:
    - After a full training cycle across symbols/modes.
    - From a daily scheduled job.

    It is safe by design: failures are logged and tuning for a given
    symbol/mode is skipped without raising.
    """
    logger.info(
        "meta_tuning_started",
        total_pairs=len(symbols_and_modes),
        lookback_days_metrics=METRICS_LOOKBACK_DAYS,
        lookback_days_trades=TRADES_LOOKBACK_DAYS,
    )

    for symbol, mode in symbols_and_modes:
        mode_cfg = DEFAULT_MODE_CONFIGS.get(
            mode,
            ModeTuningConfig(
                target_min_trades_per_day=5.0,
                target_max_trades_per_day=50.0,
                sharpe_floor_increase=0.1,
                sharpe_floor_decrease=-0.2,
            ),
        )

        try:
            current_cfg = _load_symbol_mode_config(brain, symbol, mode)
            if current_cfg is None:
                # Seed with conservative defaults
                current_cfg = SymbolModeConfig(
                    symbol=symbol,
                    mode=mode,
                    edge_threshold_bps=5.0,
                    tp_atr_k=2.0,
                    sl_atr_k=1.0,
                    min_tp_bps=10.0,
                    size_multiplier=1.0,
                    tp_bps=10.0,
                    sl_bps=5.0,
                    fee_bps=0.0,
                    spread_bps=0.0,
                    slippage_bps=0.0,
                )
                logger.info(
                    "symbol_mode_config_seeded_with_defaults",
                    symbol=symbol,
                    mode=mode,
                    edge_threshold_bps=current_cfg.edge_threshold_bps,
                    tp_atr_k=current_cfg.tp_atr_k,
                    sl_atr_k=current_cfg.sl_atr_k,
                    min_tp_bps=current_cfg.min_tp_bps,
                    size_multiplier=current_cfg.size_multiplier,
                )

            trade_stats = _get_recent_trade_stats(
                brain, symbol, mode, TRADES_LOOKBACK_DAYS
            )
            metrics = _get_recent_model_metrics(
                brain, symbol, mode, METRICS_LOOKBACK_DAYS
            )
            vol_stats = _get_recent_volatility_stats(
                brain, symbol, mode, METRICS_LOOKBACK_DAYS
            )
            cost_stats = _get_recent_cost_stats(
                brain, symbol, mode, METRICS_LOOKBACK_DAYS
            )

            trades_per_day = trade_stats.get("trades_per_day", 0.0)
            sharpe = metrics.get("sharpe", 0.0)
            max_dd = metrics.get("max_drawdown", 0.0)
            atr_bps = vol_stats.get("atr_bps", 0.0)
            median_costs_bps = cost_stats.get("median_total_costs_bps", 0.0)

            # If we really have no data, skip safely.
            if trades_per_day == 0.0 and sharpe == 0.0:
                logger.info(
                    "meta_tuning_skipped_insufficient_data",
                    symbol=symbol,
                    mode=mode,
                    trades_per_day=trades_per_day,
                    sharpe=sharpe,
                )
                continue

            # Tune edge threshold and size multiplier according to rules.
            new_edge = _tune_edge_threshold(
                symbol,
                mode,
                current_cfg,
                mode_cfg,
                trades_per_day=trades_per_day,
                sharpe=sharpe,
            )
            new_size = _tune_size_multiplier(
                symbol,
                mode,
                current_cfg,
                mode_cfg,
                sharpe=sharpe,
                max_drawdown=max_dd,
            )

            # TP/SL tuning using ATR stats
            if atr_bps > 0:
                candidate_tp = max(current_cfg.min_tp_bps, atr_bps * current_cfg.tp_atr_k)
                candidate_sl = max(1.0, atr_bps * current_cfg.sl_atr_k)
                if abs(candidate_tp - current_cfg.tp_bps) > 1e-6 or abs(candidate_sl - current_cfg.sl_bps) > 1e-6:
                    logger.info(
                        "tp_sl_updated",
                        symbol=symbol,
                        mode=mode,
                        old_tp_bps=current_cfg.tp_bps,
                        new_tp_bps=candidate_tp,
                        old_sl_bps=current_cfg.sl_bps,
                        new_sl_bps=candidate_sl,
                        atr_bps=atr_bps,
                    )
                    current_cfg.tp_bps = candidate_tp
                    current_cfg.sl_bps = candidate_sl

            # Ensure TP minimum stays above costs
            if median_costs_bps > 0 and current_cfg.min_tp_bps < median_costs_bps + 1.0:
                old_min_tp = current_cfg.min_tp_bps
                current_cfg.min_tp_bps = median_costs_bps + 1.0
                logger.info(
                    "min_tp_adjusted_for_costs",
                    symbol=symbol,
                    mode=mode,
                    old_min_tp_bps=old_min_tp,
                    new_min_tp_bps=current_cfg.min_tp_bps,
                    median_costs_bps=median_costs_bps,
                )

            # Cost overrides using realized stats
            new_fee = cost_stats.get("avg_fee_bps", 0.0)
            new_spread = cost_stats.get("avg_spread_bps", 0.0)
            new_slippage = cost_stats.get("avg_slippage_bps", 0.0)
            if (
                new_fee
                and new_spread
                and (
                    abs(new_fee - current_cfg.fee_bps) > 1e-6
                    or abs(new_spread - current_cfg.spread_bps) > 1e-6
                    or abs(new_slippage - current_cfg.slippage_bps) > 1e-6
                )
            ):
                logger.info(
                    "cost_model_updated",
                    symbol=symbol,
                    mode=mode,
                    old_fee_bps=current_cfg.fee_bps,
                    new_fee_bps=new_fee,
                    old_spread_bps=current_cfg.spread_bps,
                    new_spread_bps=new_spread,
                    old_slippage_bps=current_cfg.slippage_bps,
                    new_slippage_bps=new_slippage,
                )
                current_cfg.fee_bps = new_fee
                current_cfg.spread_bps = new_spread
                current_cfg.slippage_bps = new_slippage

            current_cfg.edge_threshold_bps = new_edge
            current_cfg.size_multiplier = new_size

            _upsert_symbol_mode_config(brain, current_cfg)

        except Exception as exc:
            logger.error(
                "meta_tuning_failed_for_symbol_mode",
                symbol=symbol,
                mode=mode,
                error=str(exc),
            )
            # Skip this symbol/mode and continue.

    logger.info("meta_tuning_completed")


