"""
Per-Coin Training Contracts

Contracts for per-coin model training, artifacts, and routing.
Implements the required contracts for per-coin baseline training.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RunManifest:
    """Manifest per run with per-coin artifacts and metrics."""
    
    run_id: str
    utc_started: datetime
    utc_finished: Optional[datetime] = None
    engine_version: str = "1.0.0"
    symbols_trained: List[str] = field(default_factory=list)
    artifacts_map: Dict[str, str] = field(default_factory=dict)  # SYMBOL -> model path
    metrics_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # SYMBOL -> metrics
    costs_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # SYMBOL -> cost object
    feature_recipe_hash_map: Dict[str, str] = field(default_factory=dict)  # SYMBOL -> hash
    status: str = "ok"  # "ok" or "failed"
    failure_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["utc_started"] = self.utc_started.isoformat()
        if self.utc_finished:
            data["utc_finished"] = self.utc_finished.isoformat()
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RunManifest:
        """Create from dictionary."""
        data = data.copy()
        data["utc_started"] = datetime.fromisoformat(data["utc_started"])
        if data.get("utc_finished"):
            data["utc_finished"] = datetime.fromisoformat(data["utc_finished"])
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> RunManifest:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class ChampionPointer:
    """Champion pointer per coin with model paths and metadata."""
    
    date: str  # YYYYMMDD format
    run_id: str
    models: Dict[str, str]  # SYMBOL -> absolute model path
    default_costs_bps: float = 15.0
    code_hash: Optional[str] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.updated_at:
            data["updated_at"] = self.updated_at.isoformat()
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChampionPointer:
        """Create from dictionary."""
        data = data.copy()
        if data.get("updated_at"):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> ChampionPointer:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class FeatureRecipe:
    """Feature recipe per coin with configuration and hash."""
    
    symbol: str
    timeframes: List[str] = field(default_factory=lambda: ["1h"])
    indicators: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # indicator_name -> params
    fill_rules: Dict[str, str] = field(default_factory=lambda: {"strategy": "forward_fill"})
    normalization: Dict[str, Any] = field(default_factory=lambda: {"type": "standard", "scaler": "StandardScaler"})
    window_sizes: Dict[str, int] = field(default_factory=lambda: {"lookback": 100, "prediction": 1})
    hash: Optional[str] = None
    
    def compute_hash(self) -> str:
        """Compute hash of feature recipe for reproducibility."""
        data = {
            "symbol": self.symbol,
            "timeframes": sorted(self.timeframes),
            "indicators": {k: v for k, v in sorted(self.indicators.items())},
            "fill_rules": self.fill_rules,
            "normalization": self.normalization,
            "window_sizes": self.window_sizes,
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Ensure hash is computed
        if not self.hash:
            self.hash = self.compute_hash()
            data["hash"] = self.hash
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FeatureRecipe:
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> FeatureRecipe:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class PerCoinMetrics:
    """Metrics per coin with cost-aware performance metrics."""
    
    symbol: str
    sample_size: int
    gross_pnl_pct: float
    net_pnl_pct: float
    sharpe: float
    hit_rate: float  # 0.0 to 1.0
    max_drawdown_pct: float
    avg_trade_bps: float
    validation_windows: List[str] = field(default_factory=list)
    costs_bps_used: Dict[str, float] = field(default_factory=dict)  # Cost object copy
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PerCoinMetrics:
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> PerCoinMetrics:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class CostModel:
    """Cost model per coin with trading fees and execution costs."""
    
    symbol: str
    taker_fee_bps: float
    maker_fee_bps: float
    median_spread_bps: float
    slippage_bps_per_sigma: float
    min_notional: float
    step_size: float
    last_updated_utc: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["last_updated_utc"] = self.last_updated_utc.isoformat()
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CostModel:
        """Create from dictionary."""
        data = data.copy()
        data["last_updated_utc"] = datetime.fromisoformat(data["last_updated_utc"])
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> CostModel:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def total_cost_bps(self, order_type: str = "taker") -> float:
        """Calculate total cost in basis points."""
        fee_bps = self.taker_fee_bps if order_type == "taker" else self.maker_fee_bps
        return fee_bps + self.median_spread_bps + self.slippage_bps_per_sigma


@dataclass
class Heartbeat:
    """Engine heartbeat with current status and progress."""
    
    utc_timestamp: datetime
    phase: str  # "loading", "training", "validating", "publishing", "complete", "failed"
    current_symbol: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0
    last_error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["utc_timestamp"] = self.utc_timestamp.isoformat()
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Heartbeat:
        """Create from dictionary."""
        data = data.copy()
        data["utc_timestamp"] = datetime.fromisoformat(data["utc_timestamp"])
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> Heartbeat:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class FailureReport:
    """Failure report with error details and suggestions."""
    
    run_id: str
    step: str
    exception_type: str
    message: str
    last_files_written: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    utc_timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.utc_timestamp is None:
            self.utc_timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.utc_timestamp:
            data["utc_timestamp"] = self.utc_timestamp.isoformat()
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FailureReport:
        """Create from dictionary."""
        data = data.copy()
        if data.get("utc_timestamp"):
            data["utc_timestamp"] = datetime.fromisoformat(data["utc_timestamp"])
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> FailureReport:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

