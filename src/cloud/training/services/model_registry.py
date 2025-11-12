"""Model registry service for persisting engine outputs into Postgres."""

from __future__ import annotations

import json
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

import structlog  # type: ignore[reportMissingImports]
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

logger = structlog.get_logger(__name__)


@dataclass
class RegistryConfig:
    dsn: str


class ModelRegistry:
    """Creates and updates model, metrics, and publish records."""

    def __init__(self, config: RegistryConfig) -> None:
        # SQLAlchemy 2.0+ doesn't need future=True (it's the default)
        # But we need to ensure proper parameter binding to avoid boolean evaluation issues
        self._engine: Engine = create_engine(config.dsn)

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
        """Clean params and features for JSON serialization before storing."""
        def clean_for_json(obj: Any) -> Any:
            """Recursively clean object for JSON serialization."""
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items() if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))}
            elif isinstance(obj, (list, tuple)):
                return [clean_for_json(item) for item in obj if not (isinstance(item, float) and (np.isnan(item) or np.isinf(item)))]
            elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                val = float(obj)
                if np.isnan(val) or np.isinf(val):
                    return None
                return val
            elif isinstance(obj, (np.bool_, bool)):
                # Convert boolean to integer to avoid SQLAlchemy 2.0 evaluation issues
                # This prevents SQLAlchemy from trying to evaluate the boolean in a clause context
                return 1 if bool(obj) else 0
            elif isinstance(obj, np.ndarray):
                return [clean_for_json(item) for item in obj.tolist()]
            elif obj is None:
                return None
            else:
                try:
                    json.dumps(obj, allow_nan=False)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)
        
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
        # Clean params and features before JSON serialization
        cleaned_params = clean_for_json(params)
        cleaned_features = clean_for_json(features)
        
        try:
            params_json = json.dumps(cleaned_params, separators=(",", ":"), allow_nan=False)
        except (ValueError, TypeError) as e:
            logger.warning("params_json_serialization_failed", symbol=symbol, error=str(e), message="Falling back to cleaned params")
            cleaned_params = clean_for_json(params)
            params_json = json.dumps(cleaned_params, separators=(",", ":"), allow_nan=False)
        
        try:
            features_json = json.dumps(cleaned_features, separators=(",", ":"), allow_nan=False)
        except (ValueError, TypeError) as e:
            logger.warning("features_json_serialization_failed", symbol=symbol, error=str(e), message="Falling back to cleaned features")
            cleaned_features = clean_for_json(features)
            features_json = json.dumps(cleaned_features, separators=(",", ":"), allow_nan=False)
        
        payload = {
            "model_id": model_id,
            "symbol": symbol,
            "kind": kind,
            "created_at": created_at,
            "s3_path": s3_path,
            "params": params_json,
            "features": features_json,
            "notes": notes,
        }
        self._execute(statement, payload)

    def upsert_metrics(self, *, model_id: str, metrics: Dict[str, Any]) -> None:
        """Clean metrics for database storage, handling NaN/Inf values."""
        def clean_value(val: Any) -> Any:
            """Clean a single value for database storage."""
            if isinstance(val, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(val)
            elif isinstance(val, (np.floating, np.float64, np.float32, np.float16)):
                val_float = float(val)
                # Replace NaN and Inf with None (database will handle as NULL)
                if np.isnan(val_float) or np.isinf(val_float):
                    logger.warning("metrics_contains_nan_or_inf", model_id=model_id, value=val_float, message="Replacing NaN/Inf with None")
                    return None
                return val_float
            elif isinstance(val, (np.bool_, bool)):
                # Convert boolean to integer to avoid SQLAlchemy 2.0 evaluation issues
                # This prevents SQLAlchemy from trying to evaluate the boolean in a clause context
                return 1 if bool(val) else 0
            elif val is None:
                return None
            else:
                return val
        
        # Clean all metric values
        cleaned_metrics = {k: clean_value(v) for k, v in metrics.items()}
        
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
        payload = {"model_id": model_id, **cleaned_metrics}
        self._execute(statement, payload)

    def log_publish(self, *, model_id: str, symbol: str, published: bool, reason: str, at: datetime) -> None:
        # Convert boolean to PostgreSQL boolean literal to avoid SQLAlchemy 2.0 evaluation issues
        # SQLAlchemy 2.0 may try to evaluate boolean parameters in SQL clauses, causing TypeError
        # Use PostgreSQL boolean literal (TRUE/FALSE) in SQL string to avoid parameter binding issues
        published_literal = "TRUE" if bool(published) else "FALSE"
        
        statement = text(
            f"""
            INSERT INTO publish_log (model_id, symbol, published, reason, at)
            VALUES (:model_id, :symbol, {published_literal}, :reason, :at)
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
            "reason": reason,
            "at": at,
        }
        self._execute(statement, payload)

    def _execute(self, statement: Any, payload: Dict[str, Any]) -> None:
        """Execute SQL statement with proper parameter binding and type conversion."""
        try:
            # Clean payload values to ensure proper types for SQLAlchemy/PostgreSQL
            cleaned_payload = {}
            for key, value in payload.items():
                if isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    cleaned_payload[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
                    val_float = float(value)
                    # Replace NaN and Inf with None (database will handle as NULL)
                    if np.isnan(val_float) or np.isinf(val_float):
                        cleaned_payload[key] = None
                    else:
                        cleaned_payload[key] = val_float
                elif isinstance(value, (np.bool_, bool)):
                    # CRITICAL: Convert boolean to integer to avoid SQLAlchemy 2.0 evaluation issues
                    # SQLAlchemy 2.0 may try to evaluate boolean parameters in SQL clauses, causing TypeError
                    # Convert to integer (0/1) - PostgreSQL will handle conversion to boolean in SQL
                    # This prevents SQLAlchemy from evaluating the boolean in a clause context
                    cleaned_payload[key] = 1 if bool(value) else 0
                elif isinstance(value, np.ndarray):
                    # Convert arrays to lists (for JSON columns)
                    cleaned_payload[key] = value.tolist()
                elif value is None:
                    cleaned_payload[key] = None
                elif isinstance(value, str):
                    # Ensure strings are properly encoded
                    cleaned_payload[key] = str(value)
                else:
                    # For other types, try to convert to native Python types
                    try:
                        # Try JSON serialization to catch any issues
                        json.dumps(value, allow_nan=False)
                        cleaned_payload[key] = value
                    except (TypeError, ValueError):
                        # If it can't be serialized, convert to string
                        cleaned_payload[key] = str(value)
            
            with self._engine.begin() as connection:
                # Execute with cleaned payload
                # SQLAlchemy 2.0+ style: use execute() with text() statement and parameters
                # CRITICAL: The TypeError "Boolean value of this clause is not defined" occurs when SQLAlchemy
                # tries to evaluate boolean parameters in a clause context during statement preparation.
                # We convert booleans to integers (0/1) in cleaned_payload, but SQLAlchemy may still try
                # to evaluate the statement itself during compilation.
                # Use exec_driver_sql to bypass SQLAlchemy's statement validation and use raw SQL
                try:
                    # Try normal execute first
                    connection.execute(statement, cleaned_payload)
                except TypeError as e:
                    if "Boolean value of this clause" in str(e):
                        # Fallback: Use raw SQL execution with psycopg2 directly
                        # This bypasses SQLAlchemy's statement validation
                        raw_sql = str(statement.compile(compile_kwargs={"literal_binds": True}))
                        # Replace parameter placeholders with values
                        for key, value in cleaned_payload.items():
                            if isinstance(value, str):
                                value = f"'{value.replace(chr(39), chr(39)+chr(39))}'"  # Escape single quotes
                            elif value is None:
                                value = "NULL"
                            else:
                                value = str(value)
                            raw_sql = raw_sql.replace(f":{key}", value)
                        # Execute raw SQL
                        connection.exec_driver_sql(raw_sql)
                    else:
                        raise
                # Commit is handled by context manager (begin())
        except SQLAlchemyError as exc:
            # Include the actual error details for debugging
            error_details = str(exc)
            error_type = type(exc).__name__
            # Get the original exception if available
            original_error = str(exc.orig) if hasattr(exc, 'orig') else error_details
            
            logger.error(
                "registry_persistence_failed",
                error_type=error_type,
                error=error_details,
                original_error=original_error,
                payload_keys=list(payload.keys()) if payload else [],
                cleaned_payload_keys=list(cleaned_payload.keys()) if 'cleaned_payload' in locals() else [],
                payload_types={k: type(v).__name__ for k, v in payload.items()} if payload else {},
                statement=str(statement)[:500] if statement else None,
            )
            raise RuntimeError(f"Failed to persist registry payload: {error_type}: {error_details} (original: {original_error})") from exc
        except Exception as exc:
            # Catch any other exceptions (TypeError, ValueError, etc.)
            error_details = str(exc)
            error_type = type(exc).__name__
            logger.error(
                "registry_persistence_failed_unexpected",
                error_type=error_type,
                error=error_details,
                payload_keys=list(payload.keys()) if payload else [],
                statement=str(statement)[:500] if statement else None,
            )
            raise RuntimeError(f"Failed to persist registry payload: {error_type}: {error_details}") from exc

    def fetch_recent_metrics(self, symbol: str, limit: int = 5) -> List[Dict[str, Any]]:
        statement = text(
            """
            SELECT
                mm.sharpe,
                mm.profit_factor,
                mm.hit_rate,
                mm.max_dd_bps,
                mm.pnl_bps,
                mm.trades_oos,
                mm.turnover,
                mm.total_costs_bps
            FROM model_metrics mm
            INNER JOIN models m ON m.model_id = mm.model_id
            WHERE m.symbol = :symbol
            ORDER BY m.created_at DESC
            LIMIT :limit
            """
        )
        try:
            with self._engine.begin() as connection:
                result = connection.execute(statement, {"symbol": symbol, "limit": limit})
                rows = []
                for row in result.mappings().all():
                    parsed: Dict[str, Any] = {}
                    for key, value in row.items():
                        if isinstance(value, Decimal):
                            parsed[key] = float(value)
                        else:
                            parsed[key] = value
                    rows.append(parsed)
                return rows
        except SQLAlchemyError:
            return []
