"""
Walk-Forward Testing Database Schema

Database tables for storing walk-forward testing results.

Tables:
1. trades: Trade records
2. predictions: Prediction records
3. features: Feature snapshots at decision time
4. attribution: Trade attribution records
5. regimes: Regime labels
6. models: Model metadata
7. data_provenance: Data versioning and provenance

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import json

import structlog

logger = structlog.get_logger(__name__)


# ============================================================================
# Database Table Definitions (SQL-like structure for documentation)
# ============================================================================

"""
CREATE TABLE trades (
    trade_id TEXT PRIMARY KEY,
    pred_id TEXT,
    timestamp_open TIMESTAMP,
    timestamp_close TIMESTAMP,
    symbol TEXT,
    side TEXT,  -- 'long' or 'short'
    size REAL,
    entry_price REAL,
    exit_price REAL,
    stop REAL,
    target REAL,
    fees REAL,
    slippage REAL,
    funding_cost REAL,
    pnl REAL,
    pnl_after_costs REAL,
    exit_reason TEXT,
    mfe REAL,  -- Maximum Favorable Excursion
    mae REAL,  -- Maximum Adverse Excursion
    time_in_trade_seconds REAL,
    risk_preset TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE predictions (
    pred_id TEXT PRIMARY KEY,
    timestamp TIMESTAMP,
    symbol TEXT,
    model_id TEXT,
    model_version TEXT,
    data_version TEXT,
    signal_type TEXT,  -- 'long', 'short', etc.
    predicted_label REAL,
    predicted_confidence REAL,
    horizon INTEGER,
    features_hash TEXT,
    features JSON,  -- All engineered features
    regime TEXT,
    volatility_bucket TEXT,
    trend_bucket TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE features (
    trade_id TEXT PRIMARY KEY,
    timestamp TIMESTAMP,
    features JSON,  -- All engineered features at decision time
    z_scores JSON,  -- Z-scores of features
    feature_hash TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE attribution (
    trade_id TEXT PRIMARY KEY,
    method TEXT,  -- 'shap', 'permutation', etc.
    top_features JSON,  -- List of (feature_name, importance) tuples
    shap_values JSON,
    error_type TEXT,  -- 'direction_wrong', 'timing_late', 'stop_too_tight', etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE regimes (
    timestamp TIMESTAMP,
    symbol TEXT,
    regime_label TEXT,  -- 'bull', 'bear', 'sideways', etc.
    vol_bucket TEXT,  -- 'high', 'medium', 'low'
    trend_bucket TEXT,  -- 'upward', 'downward', 'neutral'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (timestamp, symbol)
);

CREATE TABLE models (
    model_id TEXT PRIMARY KEY,
    code_hash TEXT,
    params JSON,
    training_span_start TIMESTAMP,
    training_span_end TIMESTAMP,
    metrics_out_of_sample JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE data_provenance (
    dataset_id TEXT PRIMARY KEY,
    sources JSON,  -- List of data sources
    candle_latency REAL,  -- Latency in seconds
    adjustments JSON,  -- Data adjustments applied
    survivorship_flag BOOLEAN,  -- Whether survivorship bias is present
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


@dataclass
class TradeRecord:
    """Trade record for database"""
    trade_id: str
    pred_id: str
    timestamp_open: datetime
    timestamp_close: Optional[datetime]
    symbol: str
    side: str
    size: float
    entry_price: float
    exit_price: Optional[float]
    stop: Optional[float]
    target: Optional[float]
    fees: float
    slippage: float
    funding_cost: float
    pnl: float
    pnl_after_costs: float
    exit_reason: Optional[str]
    mfe: Optional[float]
    mae: Optional[float]
    time_in_trade_seconds: Optional[float]
    risk_preset: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            "trade_id": self.trade_id,
            "pred_id": self.pred_id,
            "timestamp_open": self.timestamp_open.isoformat(),
            "timestamp_close": self.timestamp_close.isoformat() if self.timestamp_close else None,
            "symbol": self.symbol,
            "side": self.side,
            "size": self.size,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "stop": self.stop,
            "target": self.target,
            "fees": self.fees,
            "slippage": self.slippage,
            "funding_cost": self.funding_cost,
            "pnl": self.pnl,
            "pnl_after_costs": self.pnl_after_costs,
            "exit_reason": self.exit_reason,
            "mfe": self.mfe,
            "mae": self.mae,
            "time_in_trade_seconds": self.time_in_trade_seconds,
            "risk_preset": self.risk_preset
        }


@dataclass
class PredictionRecord:
    """Prediction record for database"""
    pred_id: str
    timestamp: datetime
    symbol: str
    model_id: str
    model_version: str
    data_version: str
    signal_type: str
    predicted_label: Optional[float]
    predicted_confidence: float
    horizon: int
    features_hash: str
    features: Dict[str, float]
    regime: Optional[str]
    volatility_bucket: Optional[str]
    trend_bucket: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            "pred_id": self.pred_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "data_version": self.data_version,
            "signal_type": self.signal_type,
            "predicted_label": self.predicted_label,
            "predicted_confidence": self.predicted_confidence,
            "horizon": self.horizon,
            "features_hash": self.features_hash,
            "features": json.dumps(self.features),
            "regime": self.regime,
            "volatility_bucket": self.volatility_bucket,
            "trend_bucket": self.trend_bucket
        }


@dataclass
class FeatureSnapshot:
    """Feature snapshot for database"""
    trade_id: str
    timestamp: datetime
    features: Dict[str, float]
    z_scores: Dict[str, float]
    feature_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            "trade_id": self.trade_id,
            "timestamp": self.timestamp.isoformat(),
            "features": json.dumps(self.features),
            "z_scores": json.dumps(self.z_scores),
            "feature_hash": self.feature_hash
        }


@dataclass
class AttributionRecord:
    """Attribution record for database"""
    trade_id: str
    method: str
    top_features: List[tuple]
    shap_values: Optional[Dict[str, float]]
    error_type: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            "trade_id": self.trade_id,
            "method": self.method,
            "top_features": json.dumps(self.top_features),
            "shap_values": json.dumps(self.shap_values) if self.shap_values else None,
            "error_type": self.error_type
        }


@dataclass
class RegimeRecord:
    """Regime record for database"""
    timestamp: datetime
    symbol: str
    regime_label: str
    vol_bucket: str
    trend_bucket: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "regime_label": self.regime_label,
            "vol_bucket": self.vol_bucket,
            "trend_bucket": self.trend_bucket
        }


@dataclass
class ModelRecord:
    """Model record for database"""
    model_id: str
    code_hash: str
    params: Dict[str, Any]
    training_span_start: datetime
    training_span_end: datetime
    metrics_out_of_sample: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            "model_id": self.model_id,
            "code_hash": self.code_hash,
            "params": json.dumps(self.params),
            "training_span_start": self.training_span_start.isoformat(),
            "training_span_end": self.training_span_end.isoformat(),
            "metrics_out_of_sample": json.dumps(self.metrics_out_of_sample)
        }


@dataclass
class DataProvenanceRecord:
    """Data provenance record for database"""
    dataset_id: str
    sources: List[str]
    candle_latency: float
    adjustments: Dict[str, Any]
    survivorship_flag: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            "dataset_id": self.dataset_id,
            "sources": json.dumps(self.sources),
            "candle_latency": self.candle_latency,
            "adjustments": json.dumps(self.adjustments),
            "survivorship_flag": self.survivorship_flag
        }

