"""Model registry service for persisting engine outputs into Postgres."""

from __future__ import annotations

import json
import re
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
        
        # Use raw SQL string to avoid SQLAlchemy's text() compilation
        # This bypasses the boolean evaluation error in SQLAlchemy 2.0.44
        sql = """
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
        self._execute_raw_sql(sql, payload)

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
        
        # Use raw SQL string to avoid SQLAlchemy's text() compilation
        # This bypasses the boolean evaluation error in SQLAlchemy 2.0.44
        sql = """
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
        payload = {"model_id": model_id, **cleaned_metrics}
        self._execute_raw_sql(sql, payload)

    def log_publish(self, *, model_id: str, symbol: str, published: bool, reason: str, at: datetime) -> None:
        # Use raw SQL string to avoid SQLAlchemy's text() compilation
        # Cast integer to boolean in SQL to handle PostgreSQL boolean type
        # This bypasses the boolean evaluation error in SQLAlchemy 2.0.44
        sql = """
            INSERT INTO publish_log (model_id, symbol, published, reason, at)
            VALUES (:model_id, :symbol, :published::boolean, :reason, :at)
            ON CONFLICT (model_id)
            DO UPDATE SET
                published = EXCLUDED.published,
                reason = EXCLUDED.reason,
                at = EXCLUDED.at
        """
        
        payload = {
            "model_id": model_id,
            "symbol": symbol,
            "published": 1 if bool(published) else 0,  # Convert boolean to integer
            "reason": reason,
            "at": at,
        }
        self._execute_raw_sql(sql, payload)

    def _execute_raw_sql(self, sql: str, payload: Dict[str, Any]) -> None:
        """Execute raw SQL string with parameter binding, bypassing SQLAlchemy's text() compilation.
        
        CRITICAL: SQLAlchemy 2.0.44 has a bug where it tries to evaluate boolean parameters
        in a clause context during statement compilation, causing "Boolean value of this clause
        is not defined" TypeError. This method completely bypasses SQLAlchemy's statement compilation
        by using exec_driver_sql() with raw SQL strings and psycopg2 parameter binding.
        
        Args:
            sql: Raw SQL string with named parameters (:param_name)
            payload: Dictionary of parameter values
        """
        try:
            # Clean payload values to ensure proper types for PostgreSQL
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
                    # Convert boolean to integer (0/1) - PostgreSQL will handle conversion
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
            
            # Convert named parameters (:param) to psycopg2 style (%s) with proper ordering
            # Pattern: :param_name (but not ::type which is a SQL cast)
            # Match :param_name but exclude :: which is a SQL cast operator (not a parameter)
            # Approach: find all :word patterns, skip those that are part of :: casts
            param_pattern = r':([a-zA-Z_][a-zA-Z0-9_]*)'
            all_matches = list(re.finditer(param_pattern, sql))
            param_names = []
            processed_positions = set()  # Track positions we've already processed (to skip cast types)
            
            for match in all_matches:
                param_name = match.group(1)
                start_pos = match.start()
                end_pos = match.end()
                
                # Skip if we've already processed this position (part of a cast)
                if start_pos in processed_positions:
                    continue
                
                # Check if followed by : (which would be ::, a SQL cast)
                if end_pos < len(sql) and sql[end_pos] == ':':
                    # This is :param::type - skip the parameter and mark cast type positions
                    # Find where the cast type ends (next space, comma, paren, etc.)
                    cast_start = end_pos + 1  # After ::
                    cast_end = cast_start
                    while cast_end < len(sql) and (sql[cast_end].isalnum() or sql[cast_end] == '_'):
                        cast_end += 1
                    # Mark the cast type positions (including the : before it and the type name)
                    # This prevents the regex from matching :boolean as a parameter
                    for pos in range(end_pos, cast_end):  # From first : to end of cast type
                        processed_positions.add(pos)
                    # Skip this parameter (it's part of a cast expression)
                    continue
                
                # Valid parameter (not part of a cast)
                param_names.append(param_name)
            
            # Build parameter list in order of appearance in SQL
            param_list = []
            seen_params = set()
            for param_name in param_names:
                if param_name in cleaned_payload and param_name not in seen_params:
                    param_list.append(cleaned_payload[param_name])
                    seen_params.add(param_name)
            
            # Replace named parameters with %s placeholders (psycopg2 style)
            # Handle parameter replacement carefully to preserve SQL casts (e.g., ::boolean)
            sql_psycopg = sql
            for param_name in param_names:
                if param_name in cleaned_payload:
                    # Replace parameter placeholder, but be careful with SQL casts
                    # If parameter has a cast (e.g., :published::boolean), replace the parameter name only
                    # Pattern: :param_name or :param_name::type
                    pattern = f':{param_name}'
                    if f'{pattern}::boolean' in sql_psycopg:
                        # Parameter has boolean cast - replace parameter but keep cast
                        sql_psycopg = sql_psycopg.replace(pattern, '%s', 1)
                        # For boolean columns, convert integer to boolean in SQL
                        # PostgreSQL will cast %s::boolean correctly when we pass integer
                        # But we need to ensure the cast is in the right place
                        pass  # Cast is already in SQL, just replace parameter
                    else:
                        # No cast, simple replacement
                        sql_psycopg = sql_psycopg.replace(pattern, '%s', 1)
            
            # Execute using exec_driver_sql to bypass SQLAlchemy's statement compilation
            # This directly uses the database driver (psycopg2) without SQLAlchemy's validation
            with self._engine.begin() as connection:
                try:
                    # Use exec_driver_sql to bypass SQLAlchemy's compilation entirely
                    # This executes raw SQL with psycopg2 parameter binding
                    connection.exec_driver_sql(sql_psycopg, tuple(param_list))
                    logger.debug("registry_execution_success", sql_preview=sql_psycopg[:100], param_count=len(param_list))
                except Exception as e:
                    error_msg = str(e)
                    error_type = type(e).__name__
                    logger.error(
                        "registry_execution_failed",
                        error_type=error_type,
                        error=error_msg,
                        sql_preview=sql_psycopg[:200],
                        param_count=len(param_list),
                        payload_keys=list(payload.keys()),
                        cleaned_payload_keys=list(cleaned_payload.keys()),
                    )
                    raise RuntimeError(f"Failed to execute registry SQL: {error_type}: {error_msg}") from e
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
                sql_preview=sql[:200] if sql else None,
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
                sql_preview=sql[:200] if sql else None,
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
