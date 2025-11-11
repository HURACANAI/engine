"""
Database Models

Database tables: models, model_metrics, promotions, live_trades, daily_equity
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

logger = structlog.get_logger(__name__)


@dataclass
class ModelRecord:
    """Model record in database."""
    model_id: str
    parent_id: Optional[str]
    kind: str  # "baseline", "challenger", "champion"
    created_at: datetime
    s3_path: str
    features_used: List[str]
    params: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "parent_id": self.parent_id,
            "kind": self.kind,
            "created_at": self.created_at.isoformat(),
            "s3_path": self.s3_path,
            "features_used": self.features_used,
            "params": self.params,
        }


@dataclass
class ModelMetrics:
    """Model metrics in database."""
    model_id: str
    sharpe: float
    hit_rate: float
    drawdown: float
    net_bps: float
    window: str  # "train", "validation", "test"
    cost_bps: float
    promoted: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "sharpe": self.sharpe,
            "hit_rate": self.hit_rate,
            "drawdown": self.drawdown,
            "net_bps": self.net_bps,
            "window": self.window,
            "cost_bps": self.cost_bps,
            "promoted": self.promoted,
        }


@dataclass
class Promotion:
    """Promotion record in database."""
    from_model_id: str
    to_model_id: str
    reason: str
    at: datetime
    snapshot: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "from_model_id": self.from_model_id,
            "to_model_id": self.to_model_id,
            "reason": self.reason,
            "at": self.at.isoformat(),
            "snapshot": self.snapshot,
        }


@dataclass
class LiveTrade:
    """Live trade record in database."""
    trade_id: str
    time: datetime
    symbol: str
    side: str  # "buy", "sell"
    size: float
    entry: float
    exit: float
    fees: float
    net_pnl: float
    model_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trade_id": self.trade_id,
            "time": self.time.isoformat(),
            "symbol": self.symbol,
            "side": self.side,
            "size": self.size,
            "entry": self.entry,
            "exit": self.exit,
            "fees": self.fees,
            "net_pnl": self.net_pnl,
            "model_id": self.model_id,
        }


@dataclass
class DailyEquity:
    """Daily equity record in database."""
    date: str  # YYYY-MM-DD
    nav: float  # Net Asset Value
    max_dd: float  # Maximum drawdown
    turnover: float
    fees_bps: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date,
            "nav": self.nav,
            "max_dd": self.max_dd,
            "turnover": self.turnover,
            "fees_bps": self.fees_bps,
        }


class DatabaseClient:
    """Database client for model registry and metrics."""

    def __init__(self, connection_string: str):
        """Initialize database client.

        Args:
            connection_string: Database connection string (e.g., "postgresql://user:pass@host:port/dbname")
        """
        self.connection_string = connection_string
        try:
            self._engine: Engine = create_engine(connection_string, future=True, pool_pre_ping=True)
            logger.info("database_client_initialized", connection_string=connection_string[:20] + "...")
        except SQLAlchemyError as e:
            logger.error("database_connection_failed", error=str(e))
            raise ConnectionError(f"Failed to connect to database: {e}") from e
    
    def save_model(self, model: ModelRecord) -> bool:
        """Save model record to database.

        Args:
            model: Model record

        Returns:
            True if successful

        Raises:
            RuntimeError: If database operation fails
        """
        statement = text(
            """
            INSERT INTO models (model_id, parent_id, kind, created_at, s3_path, features_used, params)
            VALUES (:model_id, :parent_id, :kind, :created_at, :s3_path, :features_used::jsonb, :params::jsonb)
            ON CONFLICT (model_id)
            DO UPDATE SET
                parent_id = EXCLUDED.parent_id,
                kind = EXCLUDED.kind,
                s3_path = EXCLUDED.s3_path,
                features_used = EXCLUDED.features_used,
                params = EXCLUDED.params
            """
        )
        payload = {
            "model_id": model.model_id,
            "parent_id": model.parent_id,
            "kind": model.kind,
            "created_at": model.created_at,
            "s3_path": model.s3_path,
            "features_used": json.dumps(model.features_used),
            "params": json.dumps(model.params),
        }
        try:
            with self._engine.begin() as connection:
                connection.execute(statement, payload)
            logger.info("model_saved", model_id=model.model_id, kind=model.kind)
            return True
        except SQLAlchemyError as e:
            logger.error("model_save_failed", model_id=model.model_id, error=str(e))
            raise RuntimeError(f"Failed to save model {model.model_id}: {e}") from e
    
    def save_metrics(self, metrics: ModelMetrics) -> bool:
        """Save model metrics to database.

        Args:
            metrics: Model metrics

        Returns:
            True if successful

        Raises:
            RuntimeError: If database operation fails
        """
        statement = text(
            """
            INSERT INTO model_metrics (model_id, sharpe, hit_rate, drawdown, net_bps, window, cost_bps, promoted)
            VALUES (:model_id, :sharpe, :hit_rate, :drawdown, :net_bps, :window, :cost_bps, :promoted)
            ON CONFLICT (model_id, window)
            DO UPDATE SET
                sharpe = EXCLUDED.sharpe,
                hit_rate = EXCLUDED.hit_rate,
                drawdown = EXCLUDED.drawdown,
                net_bps = EXCLUDED.net_bps,
                cost_bps = EXCLUDED.cost_bps,
                promoted = EXCLUDED.promoted
            """
        )
        payload = {
            "model_id": metrics.model_id,
            "sharpe": metrics.sharpe,
            "hit_rate": metrics.hit_rate,
            "drawdown": metrics.drawdown,
            "net_bps": metrics.net_bps,
            "window": metrics.window,
            "cost_bps": metrics.cost_bps,
            "promoted": metrics.promoted,
        }
        try:
            with self._engine.begin() as connection:
                connection.execute(statement, payload)
            logger.info("metrics_saved", model_id=metrics.model_id, sharpe=metrics.sharpe)
            return True
        except SQLAlchemyError as e:
            logger.error("metrics_save_failed", model_id=metrics.model_id, error=str(e))
            raise RuntimeError(f"Failed to save metrics for {metrics.model_id}: {e}") from e
    
    def save_promotion(self, promotion: Promotion) -> bool:
        """Save promotion record to database.

        Args:
            promotion: Promotion record

        Returns:
            True if successful

        Raises:
            RuntimeError: If database operation fails
        """
        statement = text(
            """
            INSERT INTO promotions (from_model_id, to_model_id, reason, at, snapshot)
            VALUES (:from_model_id, :to_model_id, :reason, :at, :snapshot::jsonb)
            """
        )
        payload = {
            "from_model_id": promotion.from_model_id,
            "to_model_id": promotion.to_model_id,
            "reason": promotion.reason,
            "at": promotion.at,
            "snapshot": json.dumps(promotion.snapshot),
        }
        try:
            with self._engine.begin() as connection:
                connection.execute(statement, payload)
            logger.info("promotion_saved", from_model_id=promotion.from_model_id, to_model_id=promotion.to_model_id)
            return True
        except SQLAlchemyError as e:
            logger.error("promotion_save_failed", from_model_id=promotion.from_model_id, error=str(e))
            raise RuntimeError(f"Failed to save promotion: {e}") from e
    
    def save_live_trade(self, trade: LiveTrade) -> bool:
        """Save live trade record to database.

        Args:
            trade: Live trade record

        Returns:
            True if successful

        Raises:
            RuntimeError: If database operation fails
        """
        statement = text(
            """
            INSERT INTO live_trades (trade_id, time, symbol, side, size, entry, exit, fees, net_pnl, model_id)
            VALUES (:trade_id, :time, :symbol, :side, :size, :entry, :exit, :fees, :net_pnl, :model_id)
            ON CONFLICT (trade_id)
            DO UPDATE SET
                exit = EXCLUDED.exit,
                fees = EXCLUDED.fees,
                net_pnl = EXCLUDED.net_pnl
            """
        )
        payload = {
            "trade_id": trade.trade_id,
            "time": trade.time,
            "symbol": trade.symbol,
            "side": trade.side,
            "size": trade.size,
            "entry": trade.entry,
            "exit": trade.exit,
            "fees": trade.fees,
            "net_pnl": trade.net_pnl,
            "model_id": trade.model_id,
        }
        try:
            with self._engine.begin() as connection:
                connection.execute(statement, payload)
            logger.info("live_trade_saved", trade_id=trade.trade_id, symbol=trade.symbol)
            return True
        except SQLAlchemyError as e:
            logger.error("live_trade_save_failed", trade_id=trade.trade_id, error=str(e))
            raise RuntimeError(f"Failed to save live trade {trade.trade_id}: {e}") from e

    def save_daily_equity(self, equity: DailyEquity) -> bool:
        """Save daily equity record to database.

        Args:
            equity: Daily equity record

        Returns:
            True if successful

        Raises:
            RuntimeError: If database operation fails
        """
        statement = text(
            """
            INSERT INTO daily_equity (date, nav, max_dd, turnover, fees_bps)
            VALUES (:date, :nav, :max_dd, :turnover, :fees_bps)
            ON CONFLICT (date)
            DO UPDATE SET
                nav = EXCLUDED.nav,
                max_dd = EXCLUDED.max_dd,
                turnover = EXCLUDED.turnover,
                fees_bps = EXCLUDED.fees_bps
            """
        )
        payload = {
            "date": equity.date,
            "nav": equity.nav,
            "max_dd": equity.max_dd,
            "turnover": equity.turnover,
            "fees_bps": equity.fees_bps,
        }
        try:
            with self._engine.begin() as connection:
                connection.execute(statement, payload)
            logger.info("daily_equity_saved", date=equity.date, nav=equity.nav)
            return True
        except SQLAlchemyError as e:
            logger.error("daily_equity_save_failed", date=equity.date, error=str(e))
            raise RuntimeError(f"Failed to save daily equity for {equity.date}: {e}") from e

