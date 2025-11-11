"""
Standard Event Schema - Observability events.

Defines standard event schema for all trading decisions and system events.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
import hashlib
import json
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class EngineVoteEvent:
    """Individual engine vote."""
    engine_id: str
    side: int  # -1, 0, +1
    probability: float
    reliability: float
    raw_score: float


@dataclass
class TradingDecisionEvent:
    """
    Standard trading decision event.
    
    Logged for every decision point.
    """
    timestamp: datetime
    symbol: str
    features_hash: str  # Hash of features used
    engine_votes: List[EngineVoteEvent]
    consensus_score: float
    action: str  # "buy", "sell", "wait"
    size_usd: float
    price: float
    expected_edge_bps: float
    realized_pnl: Optional[float] = None  # Filled after trade closes
    regime: str = "unknown"
    costs_bps: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        data = asdict(self)
        # Convert datetime to ISO string
        data['timestamp'] = self.timestamp.isoformat()
        # Convert engine votes
        data['engine_votes'] = [asdict(v) for v in self.engine_votes]
        return data


@dataclass
class HealthCheckEvent:
    """System health check event."""
    timestamp: datetime
    component: str
    status: str  # "healthy", "degraded", "down"
    latency_ms: Optional[float] = None
    error_rate: Optional[float] = None
    data_gaps: Optional[int] = None
    model_load_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class TradeExecutionEvent:
    """Trade execution event."""
    timestamp: datetime
    symbol: str
    side: str  # "buy", "sell"
    size_usd: float
    price: float
    venue: str
    order_type: str
    execution_id: str
    slippage_bps: float
    fees_bps: float
    latency_ms: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class PnLAttributionEvent:
    """PnL attribution event (daily summary)."""
    timestamp: datetime
    date: str  # YYYY-MM-DD
    total_pnl_usd: float
    pnl_by_engine: Dict[str, float]
    pnl_by_regime: Dict[str, float]
    pnl_by_venue: Dict[str, float]
    cost_breakdown: Dict[str, float]  # fees, slippage, funding, etc.
    num_trades: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class EventLogger:
    """
    Standard event logger for observability.
    
    Logs all events with consistent schema.
    """
    
    def __init__(self) -> None:
        """Initialize event logger."""
        self.logger = structlog.get_logger(__name__)
        self.events: List[Dict[str, Any]] = []
        self.max_events_in_memory = 10000
    
    def log_decision(self, event: TradingDecisionEvent) -> None:
        """Log trading decision event."""
        event_dict = event.to_dict()
        self.logger.info("trading_decision", **event_dict)
        self._store_event("trading_decision", event_dict)
    
    def log_health_check(self, event: HealthCheckEvent) -> None:
        """Log health check event."""
        event_dict = event.to_dict()
        self.logger.info("health_check", **event_dict)
        self._store_event("health_check", event_dict)
    
    def log_execution(self, event: TradeExecutionEvent) -> None:
        """Log trade execution event."""
        event_dict = event.to_dict()
        self.logger.info("trade_execution", **event_dict)
        self._store_event("trade_execution", event_dict)
    
    def log_pnl_attribution(self, event: PnLAttributionEvent) -> None:
        """Log PnL attribution event."""
        event_dict = event.to_dict()
        self.logger.info("pnl_attribution", **event_dict)
        self._store_event("pnl_attribution", event_dict)
    
    def _store_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Store event in memory (for recent events)."""
        self.events.append({
            "type": event_type,
            "data": event_data,
            "logged_at": datetime.now().isoformat()
        })
        
        # Limit memory usage
        if len(self.events) > self.max_events_in_memory:
            self.events.pop(0)
    
    def get_recent_events(self, event_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent events.
        
        Args:
            event_type: Filter by event type (optional)
            limit: Maximum number of events
        
        Returns:
            List of recent events
        """
        events = self.events
        if event_type:
            events = [e for e in events if e["type"] == event_type]
        
        return events[-limit:]
    
    def hash_features(self, features: Dict[str, float]) -> str:
        """
        Create hash of features for tracking.
        
        Args:
            features: Feature dictionary
        
        Returns:
            Feature hash string
        """
        # Sort features for consistent hashing
        sorted_features = json.dumps(features, sort_keys=True)
        return hashlib.sha256(sorted_features.encode()).hexdigest()[:16]


# Global event logger instance
_event_logger: Optional[EventLogger] = None


def get_event_logger() -> EventLogger:
    """Get global event logger instance."""
    global _event_logger
    if _event_logger is None:
        _event_logger = EventLogger()
    return _event_logger

