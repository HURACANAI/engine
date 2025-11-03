"""Model registry service for persisting engine outputs into Postgres."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError


@dataclass
class RegistryConfig:
    dsn: str


class ModelRegistry:
    """Creates and updates model, metrics, and publish records."""

    def __init__(self, config: RegistryConfig) -> None:
        self._engine: Engine = create_engine(config.dsn, future=True)

    def upsert_model(
        self,
        *,
        model_id: str,
        symbol: str,
        kind: str,
        created_at: datetime,
        s3_path: str,
        params: Dict[str, Any],
        features: Dict[str, Any],
        notes: str | None = None,
    ) -> None:
        statement = text(
            """
            INSERT INTO models (model_id, symbol, kind, created_at, s3_path, params, features, notes)
            VALUES (:model_id, :symbol, :kind, :created_at, :s3_path, :params::jsonb, :features::jsonb, :notes)
            ON CONFLICT (model_id)
            DO UPDATE SET
                symbol = EXCLUDED.symbol,
                kind = EXCLUDED.kind,
                s3_path = EXCLUDED.s3_path,
                params = EXCLUDED.params,
                features = EXCLUDED.features,
                notes = EXCLUDED.notes
            """
        )
        payload = {
            "model_id": model_id,
            "symbol": symbol,
            "kind": kind,
            "created_at": created_at,
            "s3_path": s3_path,
            "params": json.dumps(params),
            "features": json.dumps(features),
            "notes": notes,
        }
        self._execute(statement, payload)

    def upsert_metrics(self, *, model_id: str, metrics: Dict[str, Any]) -> None:
        statement = text(
            """
            INSERT INTO model_metrics (
                model_id, sharpe, profit_factor, hit_rate, max_dd_bps, pnl_bps, trades_oos,
                turnover, fee_bps, spread_bps, slippage_bps, total_costs_bps, validation_window
            )
            VALUES (
                :model_id, :sharpe, :profit_factor, :hit_rate, :max_dd_bps, :pnl_bps, :trades_oos,
                :turnover, :fee_bps, :spread_bps, :slippage_bps, :total_costs_bps, :validation_window
            )
            ON CONFLICT (model_id)
            DO UPDATE SET
                sharpe = EXCLUDED.sharpe,
                profit_factor = EXCLUDED.profit_factor,
                hit_rate = EXCLUDED.hit_rate,
                max_dd_bps = EXCLUDED.max_dd_bps,
                pnl_bps = EXCLUDED.pnl_bps,
                trades_oos = EXCLUDED.trades_oos,
                turnover = EXCLUDED.turnover,
                fee_bps = EXCLUDED.fee_bps,
                spread_bps = EXCLUDED.spread_bps,
                slippage_bps = EXCLUDED.slippage_bps,
                total_costs_bps = EXCLUDED.total_costs_bps,
                validation_window = EXCLUDED.validation_window
            """
        )
        payload = {"model_id": model_id, **metrics}
        self._execute(statement, payload)

    def log_publish(self, *, model_id: str, symbol: str, published: bool, reason: str, at: datetime) -> None:
        statement = text(
            """
            INSERT INTO publish_log (model_id, symbol, published, reason, at)
            VALUES (:model_id, :symbol, :published, :reason, :at)
            ON CONFLICT (model_id)
            DO UPDATE SET
                published = EXCLUDED.published,
                reason = EXCLUDED.reason,
                at = EXCLUDED.at
            """
        )
        payload = {
            "model_id": model_id,
            "symbol": symbol,
            "published": published,
            "reason": reason,
            "at": at,
        }
        self._execute(statement, payload)

    def _execute(self, statement: Any, payload: Dict[str, Any]) -> None:
        try:
            with self._engine.begin() as connection:
                connection.execute(statement, payload)
        except SQLAlchemyError as exc:
            raise RuntimeError("Failed to persist registry payload") from exc
