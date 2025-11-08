"""
Unified Model Registry Implementation

Single source of truth for model lifecycle management.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import psycopg2
import psycopg2.extras
import structlog

logger = structlog.get_logger(__name__)


class ModelStatus(str, Enum):
    """Model publication status"""
    PUBLISH = "publish"
    SHADOW = "shadow"
    REJECT = "reject"
    PENDING = "pending"


@dataclass
class ModelRecord:
    """Complete model record"""
    model_id: str
    symbol: str
    version: str
    created_at: datetime
    updated_at: datetime

    # Lineage
    feature_set_id: Optional[str] = None
    run_manifest_id: Optional[str] = None
    parent_model_id: Optional[str] = None

    # Gate results
    gate_verdict: Optional[Dict[str, Any]] = None
    meta_weight: Optional[float] = None
    publish_status: ModelStatus = ModelStatus.PENDING

    # Metrics
    sharpe_ratio: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    brier_score: Optional[float] = None
    win_rate: Optional[float] = None
    trades_oos: Optional[int] = None

    # Artifacts
    s3_artifacts_uri: Optional[str] = None
    onnx_path: Optional[str] = None
    metadata_path: Optional[str] = None
    model_params: Optional[Dict[str, Any]] = None
    feature_metadata: Optional[Dict[str, Any]] = None

    # Lifecycle
    published_at: Optional[datetime] = None
    deprecated_at: Optional[datetime] = None
    deprecated_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling UUID and datetime"""
        d = asdict(self)
        d['publish_status'] = self.publish_status.value
        return d


class UnifiedModelRegistry:
    """
    Unified Model Registry - Single Source of Truth

    Consolidates model tracking from scattered registries.
    Provides complete lifecycle management from training to production.

    Example:
        registry = UnifiedModelRegistry(dsn="postgresql://...")

        # Register new model
        model_id = registry.register_model(
            symbol="BTC",
            version="2025-11-08_02-00",
            feature_set_id="fs_abc123",
            run_manifest_id="run_xyz789",
            metrics={"sharpe_ratio": 1.5, "max_drawdown_pct": 12.0}
        )

        # Update with gate verdict
        registry.update_gate_verdict(
            model_id=model_id,
            verdict={"status": "PUBLISH", "meta_weight": 0.15}
        )

        # Get publishable models for Hamilton
        models = registry.get_publishable_models(symbol="BTC", limit=5)
    """

    def __init__(self, dsn: str):
        """
        Initialize registry

        Args:
            dsn: PostgreSQL connection string
        """
        self.dsn = dsn
        self._conn: Optional[psycopg2.extensions.connection] = None

    def _connect(self) -> psycopg2.extensions.connection:
        """Get database connection"""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.dsn)
            psycopg2.extras.register_uuid()
        return self._conn

    def register_model(
        self,
        symbol: str,
        version: str,
        feature_set_id: Optional[str] = None,
        run_manifest_id: Optional[str] = None,
        parent_model_id: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        artifacts_uri: Optional[str] = None,
        onnx_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        model_params: Optional[Dict[str, Any]] = None,
        feature_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register a new model

        Args:
            symbol: Trading symbol (e.g., "BTC")
            version: Model version (e.g., "2025-11-08_02-00")
            feature_set_id: Feature set identifier
            run_manifest_id: Run manifest identifier
            parent_model_id: Parent model (for lineage)
            metrics: Performance metrics dict
            artifacts_uri: S3 URI for artifacts
            onnx_path: Path to ONNX model
            metadata_path: Path to metadata JSON
            model_params: Model parameters
            feature_metadata: Feature metadata

        Returns:
            model_id (UUID string)
        """
        conn = self._connect()
        model_id = str(uuid4())

        metrics = metrics or {}

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO unified_models (
                    model_id, symbol, version,
                    feature_set_id, run_manifest_id, parent_model_id,
                    sharpe_ratio, max_drawdown_pct, brier_score, win_rate, trades_oos,
                    s3_artifacts_uri, onnx_path, metadata_path,
                    model_params, feature_metadata,
                    publish_status
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    'pending'
                )
                RETURNING model_id
                """,
                (
                    model_id, symbol, version,
                    feature_set_id, run_manifest_id, parent_model_id,
                    metrics.get('sharpe_ratio'), metrics.get('max_drawdown_pct'),
                    metrics.get('brier_score'), metrics.get('win_rate'),
                    metrics.get('trades_oos'),
                    artifacts_uri, onnx_path, metadata_path,
                    json.dumps(model_params) if model_params else None,
                    json.dumps(feature_metadata) if feature_metadata else None,
                )
            )
            conn.commit()

        logger.info(
            "model_registered",
            model_id=model_id,
            symbol=symbol,
            version=version,
            feature_set_id=feature_set_id,
        )

        return model_id

    def update_gate_verdict(
        self,
        model_id: str,
        verdict: Dict[str, Any],
        meta_weight: Optional[float] = None,
        publish_status: Optional[ModelStatus] = None,
    ) -> None:
        """
        Update model with gate evaluation results

        Args:
            model_id: Model UUID
            verdict: Full gate verdict dict
            meta_weight: Computed meta weight
            publish_status: Publish, shadow, or reject
        """
        conn = self._connect()

        # Extract from verdict if not provided
        if meta_weight is None:
            meta_weight = verdict.get('meta_weight')
        if publish_status is None:
            status_str = verdict.get('status', 'pending')
            publish_status = ModelStatus(status_str.lower())

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE unified_models
                SET gate_verdict = %s,
                    meta_weight = %s,
                    publish_status = %s,
                    published_at = CASE
                        WHEN %s = 'publish' AND published_at IS NULL THEN NOW()
                        ELSE published_at
                    END
                WHERE model_id = %s
                """,
                (
                    json.dumps(verdict),
                    meta_weight,
                    publish_status.value,
                    publish_status.value,
                    model_id,
                )
            )

            # Also log to gate_evaluations table
            cur.execute(
                """
                INSERT INTO gate_evaluations (
                    model_id, evaluation_data, final_verdict, meta_weight,
                    passed_gates, failed_gates, warnings
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    model_id,
                    json.dumps(verdict),
                    publish_status.value,
                    meta_weight,
                    verdict.get('passed_gates', []),
                    verdict.get('failed_gates', []),
                    verdict.get('warnings', []),
                )
            )

            conn.commit()

        logger.info(
            "gate_verdict_updated",
            model_id=model_id,
            status=publish_status.value,
            meta_weight=meta_weight,
        )

    def get_publishable_models(
        self,
        symbol: Optional[str] = None,
        limit: int = 10,
        min_meta_weight: float = 0.0,
    ) -> List[ModelRecord]:
        """
        Get publishable models for Hamilton

        Args:
            symbol: Filter by symbol (None = all)
            limit: Max models to return
            min_meta_weight: Minimum meta weight threshold

        Returns:
            List of ModelRecord objects
        """
        conn = self._connect()

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            query = """
                SELECT * FROM unified_models
                WHERE publish_status = 'publish'
                  AND deprecated_at IS NULL
                  AND meta_weight >= %s
            """
            params = [min_meta_weight]

            if symbol:
                query += " AND symbol = %s"
                params.append(symbol)

            query += " ORDER BY meta_weight DESC, created_at DESC LIMIT %s"
            params.append(limit)

            cur.execute(query, params)
            rows = cur.fetchall()

        models = []
        for row in rows:
            # Convert row to ModelRecord
            model = ModelRecord(
                model_id=str(row['model_id']),
                symbol=row['symbol'],
                version=row['version'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                feature_set_id=row.get('feature_set_id'),
                run_manifest_id=row.get('run_manifest_id'),
                parent_model_id=str(row['parent_model_id']) if row.get('parent_model_id') else None,
                gate_verdict=row.get('gate_verdict'),
                meta_weight=row.get('meta_weight'),
                publish_status=ModelStatus(row['publish_status']),
                sharpe_ratio=row.get('sharpe_ratio'),
                max_drawdown_pct=row.get('max_drawdown_pct'),
                brier_score=row.get('brier_score'),
                win_rate=row.get('win_rate'),
                trades_oos=row.get('trades_oos'),
                s3_artifacts_uri=row.get('s3_artifacts_uri'),
                onnx_path=row.get('onnx_path'),
                metadata_path=row.get('metadata_path'),
                model_params=row.get('model_params'),
                feature_metadata=row.get('feature_metadata'),
                published_at=row.get('published_at'),
                deprecated_at=row.get('deprecated_at'),
                deprecated_reason=row.get('deprecated_reason'),
            )
            models.append(model)

        return models

    def get_model(self, model_id: str) -> Optional[ModelRecord]:
        """Get single model by ID"""
        conn = self._connect()

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM unified_models WHERE model_id = %s",
                (model_id,)
            )
            row = cur.fetchone()

        if not row:
            return None

        return ModelRecord(
            model_id=str(row['model_id']),
            symbol=row['symbol'],
            version=row['version'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            feature_set_id=row.get('feature_set_id'),
            run_manifest_id=row.get('run_manifest_id'),
            parent_model_id=str(row['parent_model_id']) if row.get('parent_model_id') else None,
            gate_verdict=row.get('gate_verdict'),
            meta_weight=row.get('meta_weight'),
            publish_status=ModelStatus(row['publish_status']),
            sharpe_ratio=row.get('sharpe_ratio'),
            max_drawdown_pct=row.get('max_drawdown_pct'),
            brier_score=row.get('brier_score'),
            win_rate=row.get('win_rate'),
            trades_oos=row.get('trades_oos'),
            s3_artifacts_uri=row.get('s3_artifacts_uri'),
            onnx_path=row.get('onnx_path'),
            metadata_path=row.get('metadata_path'),
            model_params=row.get('model_params'),
            feature_metadata=row.get('feature_metadata'),
            published_at=row.get('published_at'),
            deprecated_at=row.get('deprecated_at'),
            deprecated_reason=row.get('deprecated_reason'),
        )

    def deprecate_model(
        self,
        model_id: str,
        reason: str,
    ) -> None:
        """
        Deprecate a model (soft delete)

        Args:
            model_id: Model UUID
            reason: Deprecation reason
        """
        conn = self._connect()

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE unified_models
                SET deprecated_at = NOW(),
                    deprecated_reason = %s
                WHERE model_id = %s
                """,
                (reason, model_id)
            )
            conn.commit()

        logger.info("model_deprecated", model_id=model_id, reason=reason)

    def record_performance(
        self,
        model_id: str,
        live_trades: int,
        live_pnl_gbp: float,
        live_sharpe: Optional[float] = None,
        live_win_rate: Optional[float] = None,
        avg_slippage_bps: Optional[float] = None,
        regime: Optional[str] = None,
    ) -> None:
        """
        Record live performance data from Hamilton

        Args:
            model_id: Model UUID
            live_trades: Number of live trades
            live_pnl_gbp: Live PnL in GBP
            live_sharpe: Live Sharpe ratio
            live_win_rate: Live win rate
            avg_slippage_bps: Average slippage in bps
            regime: Market regime
        """
        conn = self._connect()

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO model_performance (
                    model_id, live_trades, live_pnl_gbp, live_sharpe,
                    live_win_rate, avg_slippage_bps, regime
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    model_id, live_trades, live_pnl_gbp, live_sharpe,
                    live_win_rate, avg_slippage_bps, regime
                )
            )
            conn.commit()

    def get_model_lineage(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Get full lineage (ancestors and descendants)

        Args:
            model_id: Model UUID

        Returns:
            List of lineage records (generation, path)
        """
        conn = self._connect()

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM v_model_lineage
                WHERE %s = ANY(lineage_path)
                ORDER BY generation
                """,
                (model_id,)
            )
            rows = cur.fetchall()

        return [dict(row) for row in rows]

    def close(self):
        """Close database connection"""
        if self._conn and not self._conn.closed:
            self._conn.close()
