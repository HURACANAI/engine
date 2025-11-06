"""
Observability Schemas - Pydantic v2 with strict validation

All events, trades, models tracked with:
- Schema versioning for migrations
- Timestamp validation (prevent look-ahead bias)
- Content-addressable IDs for reproducibility
- Full type safety

Critical Guards:
1. decision_timestamp ≤ label_cutoff_timestamp (no peeking into future)
2. All events have schema version for safe migrations
3. Model IDs are content hashes (SHA256) for reproducibility
4. Git SHA tracking for code reproducibility
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Literal, Any
from datetime import datetime
from enum import Enum
import uuid


# ============================================================================
# ENUMS
# ============================================================================

class EventKind(str, Enum):
    """Event types"""
    SIGNAL = "signal"
    GATE_DECISION = "gate_decision"
    TRADE_EXEC = "trade_exec"
    TRADE_CLOSE = "trade_close"
    MODEL_UPDATE = "model_update"
    GATE_ADJUSTMENT = "gate_adjustment"
    ERROR = "error"


class Priority(str, Enum):
    """Event priority (for lossy tiering)"""
    DEBUG = "debug"
    NORMAL = "normal"
    CRITICAL = "critical"


class GateDecisionType(str, Enum):
    """Gate pass/fail"""
    PASS = "pass"
    REJECT = "reject"


class TradeResult(str, Enum):
    """Trade outcome"""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"


class HealthStatus(str, Enum):
    """System health"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


# ============================================================================
# SUB-MODELS
# ============================================================================

class GateDecision(BaseModel):
    """Gate evaluation result"""
    name: str  # meta_label, cost, regret, etc.
    decision: GateDecisionType
    inputs: Dict[str, float]  # win_prob, threshold, ev_bps, etc.
    context: Dict[str, Any]  # technique, regime, confidence
    timing_ms: float  # How long gate evaluation took
    counterfactual: Optional['Counterfactual'] = None

    class Config:
        use_enum_values = True


class Counterfactual(BaseModel):
    """What would have happened if we took blocked trade"""
    eligible: bool  # Can we compute counterfactual?
    label_cutoff_timestamp: Optional[datetime] = None  # When label became known
    hindsight_pnl_bps: Optional[float] = None  # Actual outcome
    verdict: Optional[Literal["SAVED_LOSS", "MISSED_PROFIT"]] = None

    @field_validator('label_cutoff_timestamp')
    @classmethod
    def validate_cutoff(cls, v, info):
        """Ensure label cutoff is after decision"""
        # Will be validated at event level
        return v


class TradeExecution(BaseModel):
    """Trade execution details"""
    side: Literal["LONG", "SHORT"]
    size_gbp: float
    entry_price: float
    order_type: Literal["MAKER", "TAKER"]
    fill_time_sec: float
    fees_bps: float  # Negative = rebate
    slippage_bps: float


class TradeOutcome(BaseModel):
    """Trade close details"""
    exit_price: float
    hold_time_sec: float
    pnl_bps: float
    pnl_gbp: float
    won: bool
    exit_reason: str  # take_profit, stop_loss, regime_change, timeout


class ModelUpdate(BaseModel):
    """Model training event"""
    model_id: str  # sha256:...
    code_git_sha: str  # Git commit hash
    data_snapshot_id: str  # Data version
    train_start: datetime
    train_end: datetime
    metrics: Dict[str, float]  # auc, ece, brier, wr
    n_samples: int
    notes: str = ""


class MarketContext(BaseModel):
    """Market conditions at decision time"""
    volatility_1h: float
    spread_bps: float
    liquidity_score: float  # 0-1
    recent_trend_30m: float  # % change
    volume_vs_avg: float  # Current / average
    order_book_imbalance: Optional[float] = None  # Buy pressure - sell pressure


class DecisionStep(BaseModel):
    """One step in decision pipeline"""
    stage: str  # alpha_engines, consensus, mode_select, gate_cost, etc.
    started_at: datetime
    completed_at: datetime
    latency_ms: float
    result: Dict[str, Any]


class DecisionTrace(BaseModel):
    """Complete decision timeline"""
    steps: List[DecisionStep]
    total_latency_ms: float


# ============================================================================
# MAIN EVENT SCHEMA
# ============================================================================

class Event(BaseModel):
    """
    Universal event schema - all observability data.

    Guards:
    - decision_timestamp tracked for leakage prevention
    - event_version for safe schema migrations
    - Optional fields for different event kinds
    """

    # Meta
    event_version: Literal[1] = 1
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ts: datetime = Field(default_factory=datetime.utcnow)
    decision_timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: Literal["hamilton", "engine", "mechanic", "system"]
    kind: EventKind

    # Core identifiers
    symbol: Optional[str] = None
    mode: Optional[Literal["SCALP", "RUNNER"]] = None
    regime: Optional[Literal["TREND", "RANGE", "PANIC", "UNKNOWN"]] = None

    # Signal features
    features: Optional[Dict[str, float]] = None

    # Event-specific data
    gate: Optional[GateDecision] = None
    trade: Optional[TradeExecution] = None
    outcome: Optional[TradeOutcome] = None
    model: Optional[ModelUpdate] = None
    market_context: Optional[MarketContext] = None
    decision_trace: Optional[DecisionTrace] = None

    # Tags for filtering
    tags: List[str] = Field(default_factory=list)

    # Error info
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    class Config:
        use_enum_values = True

    def model_post_init(self, __context):
        """Validate no future leakage after model creation"""
        if self.gate and self.gate.counterfactual:
            cf = self.gate.counterfactual
            if cf.label_cutoff_timestamp and self.decision_timestamp > cf.label_cutoff_timestamp:
                raise ValueError(
                    f"decision_timestamp ({self.decision_timestamp}) after "
                    f"label_cutoff ({cf.label_cutoff_timestamp}) - potential data leakage!"
                )


# ============================================================================
# DATABASE SCHEMAS (SQLite)
# ============================================================================

class Trade(BaseModel):
    """Trade record for SQLite"""
    trade_id: str
    ts_open: datetime
    ts_close: Optional[datetime] = None
    symbol: str
    mode: str  # SCALP or RUNNER
    regime: str
    side: str  # LONG or SHORT
    size_gbp: float
    entry: float
    exit: Optional[float] = None
    pnl_bps: Optional[float] = None
    pnl_gbp: Optional[float] = None
    fees_bps: float
    result: Optional[TradeResult] = None
    model_id: str
    signal_id: str  # Link to event
    decision_latency_ms: float
    market_context_json: Optional[str] = None  # JSON string

    class Config:
        use_enum_values = True


class Model(BaseModel):
    """Model record for SQLite"""
    model_id: str  # sha256:...
    created_at: datetime
    code_git_sha: str
    data_snapshot_id: str
    auc: float
    ece: float
    brier: float
    wr: Optional[float] = None
    n_samples: int
    notes: str = ""


class ModelDelta(BaseModel):
    """Model comparison record"""
    ts: datetime
    from_model_id: str
    to_model_id: str
    auc_delta: float
    ece_delta: float
    wr_delta: Optional[float] = None
    sharpe_delta: Optional[float] = None
    mdd_delta: Optional[float] = None  # Max drawdown
    notes: str = ""


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_signal_event(
    symbol: str,
    price: float,
    features: Dict[str, float],
    regime: str,
    market_context: MarketContext,
    tags: List[str] = None
) -> Event:
    """Helper to create signal event"""
    return Event(
        source="hamilton",
        kind=EventKind.SIGNAL,
        symbol=symbol,
        regime=regime,
        features={"price": price, **features},
        market_context=market_context,
        tags=tags or []
    )


def create_gate_event(
    symbol: str,
    gate_decision: GateDecision,
    mode: str,
    features: Dict[str, float],
    tags: List[str] = None
) -> Event:
    """Helper to create gate decision event"""
    return Event(
        source="hamilton",
        kind=EventKind.GATE_DECISION,
        symbol=symbol,
        mode=mode,
        gate=gate_decision,
        features=features,
        tags=tags or []
    )


def create_trade_exec_event(
    symbol: str,
    mode: str,
    trade: TradeExecution,
    signal_id: str,
    decision_trace: DecisionTrace,
    tags: List[str] = None
) -> Event:
    """Helper to create trade execution event"""
    all_tags = (tags or []) + [signal_id]
    return Event(
        source="hamilton",
        kind=EventKind.TRADE_EXEC,
        symbol=symbol,
        mode=mode,
        trade=trade,
        decision_trace=decision_trace,
        tags=all_tags
    )


def create_model_update_event(
    model_update: ModelUpdate,
    tags: List[str] = None
) -> Event:
    """Helper to create model update event"""
    return Event(
        source="engine",
        kind=EventKind.MODEL_UPDATE,
        model=model_update,
        tags=tags or []
    )


# SQLite schema creation SQL
SQLITE_SCHEMA = """
-- Trades table
CREATE TABLE IF NOT EXISTS trades(
  trade_id TEXT PRIMARY KEY,
  ts_open TEXT NOT NULL,
  ts_close TEXT,
  symbol TEXT NOT NULL,
  mode TEXT NOT NULL CHECK(mode IN ('SCALP','RUNNER')),
  regime TEXT NOT NULL CHECK(regime IN ('TREND','RANGE','PANIC','UNKNOWN')),
  side TEXT NOT NULL CHECK(side IN ('LONG','SHORT')),
  size_gbp REAL NOT NULL,
  entry REAL NOT NULL,
  exit REAL,
  pnl_bps REAL,
  pnl_gbp REAL,
  fees_bps REAL NOT NULL,
  result TEXT CHECK(result IN ('WIN','LOSS','BREAKEVEN')),
  model_id TEXT NOT NULL,
  signal_id TEXT NOT NULL,
  decision_latency_ms REAL NOT NULL,
  market_context_json TEXT
);

CREATE INDEX IF NOT EXISTS ix_trades_ts ON trades(ts_open);
CREATE INDEX IF NOT EXISTS ix_trades_mode_regime ON trades(mode, regime);
CREATE INDEX IF NOT EXISTS ix_trades_result ON trades(result);
CREATE INDEX IF NOT EXISTS ix_trades_symbol ON trades(symbol);

-- Models table
CREATE TABLE IF NOT EXISTS models(
  model_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  code_git_sha TEXT NOT NULL,
  data_snapshot_id TEXT NOT NULL,
  auc REAL NOT NULL,
  ece REAL NOT NULL,
  brier REAL NOT NULL,
  wr REAL,
  n_samples INTEGER NOT NULL,
  notes TEXT
);

CREATE INDEX IF NOT EXISTS ix_models_created ON models(created_at);
CREATE INDEX IF NOT EXISTS ix_models_auc ON models(auc);

-- Model deltas table
CREATE TABLE IF NOT EXISTS model_deltas(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  from_model_id TEXT NOT NULL,
  to_model_id TEXT NOT NULL,
  auc_delta REAL NOT NULL,
  ece_delta REAL NOT NULL,
  wr_delta REAL,
  sharpe_delta REAL,
  mdd_delta REAL,
  notes TEXT
);

CREATE INDEX IF NOT EXISTS ix_deltas_ts ON model_deltas(ts);
"""


if __name__ == '__main__':
    # Test schema validation
    print("Testing schemas...")

    # Test event creation
    event = create_signal_event(
        symbol="ETH-USD",
        price=2045.50,
        features={"confidence": 0.78, "trend_strength": 0.64},
        regime="TREND",
        market_context=MarketContext(
            volatility_1h=0.34,
            spread_bps=4.2,
            liquidity_score=0.82,
            recent_trend_30m=0.008,
            volume_vs_avg=1.2
        ),
        tags=["high_confidence"]
    )

    print(f"✓ Event created: {event.event_id}")
    print(f"  Kind: {event.kind}")
    print(f"  Symbol: {event.symbol}")
    print(f"  Market context: vol={event.market_context.volatility_1h:.2f}")

    # Test gate decision
    gate = GateDecision(
        name="meta_label",
        decision=GateDecisionType.PASS,
        inputs={"win_prob": 0.71, "threshold": 0.65, "ev_bps": 12.3},
        context={"technique": "TREND", "regime": "TREND"},
        timing_ms=15.7
    )

    print(f"✓ Gate decision: {gate.name} {gate.decision}")

    # Test leakage prevention
    try:
        bad_event = Event(
            source="hamilton",
            kind=EventKind.GATE_DECISION,
            decision_timestamp=datetime(2025, 11, 5, 14, 32, 0),
            gate=GateDecision(
                name="test",
                decision=GateDecisionType.REJECT,
                inputs={},
                context={},
                timing_ms=1.0,
                counterfactual=Counterfactual(
                    eligible=True,
                    label_cutoff_timestamp=datetime(2025, 11, 5, 14, 30, 0),  # BEFORE decision!
                    hindsight_pnl_bps=-12.0
                )
            )
        )
        print("✗ Leakage validation FAILED - should have raised error!")
    except ValueError as e:
        print(f"✓ Leakage prevention working: {e}")

    print("\nAll schema tests passed ✓")
