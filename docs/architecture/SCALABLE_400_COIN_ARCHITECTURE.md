# Scalable 400-Coin Engine Architecture

**Version:** 3.0  
**Date:** 2025-01-XX  
**Status:** Design & Implementation Plan

---

## 🎯 Executive Summary

This document defines the architecture for a production-ready, scalable trading engine capable of:
- Training 400+ Binance coins without hard limits
- Distributed, asynchronous training using Ray/Dask on RunPod GPUs
- Automatic model building and versioning for every coin, regime, and timeframe
- Real-time cost modeling and liquidity filtering
- Consensus voting with 23 engines, reliability weights, and correlation penalties
- Regime gating and risk presets
- Comprehensive observability with Prometheus/Grafana
- Complete separation of training (Engine) from execution (Hamilton)

---

## 📐 Core Design Principles

### 1. Separation of Concerns
- **Engine (Training)**: Trains models without capital limits, builds Brain Library
- **Hamilton (Execution)**: Trades with strict capital limits, uses Brain Library models
- **Complete isolation**: Training never affects live trading

### 2. Scalability First
- **Horizontal scaling**: Ray/Dask for distributed training
- **Async I/O**: All file operations and network calls are async
- **Configuration-driven**: All limits and thresholds via YAML
- **Throttling**: Build for 400 coins, throttle via config

### 3. Modularity & Dependency Injection
- **Type-hinted**: All functions have complete type hints
- **Dependency injection**: No hard dependencies, all injected
- **Testable**: 80%+ test coverage with pytest-asyncio

### 4. Observability
- **Structured logging**: structlog for all events
- **DecisionEvent logs**: Every model action logged with structured data
- **Prometheus metrics**: PnL by engine, latency, error rates
- **Grafana dashboards**: Real-time monitoring

---

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING LAYER (Engine)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Coin Selector│  │ Distributed  │  │ Model        │          │
│  │ & Liquidity  │→ │ Training     │→ │ Versioning   │          │
│  │ Filter       │  │ Orchestrator │  │ & Registry   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                  │                  │                  │
│         └──────────────────┼──────────────────┘                  │
│                            ↓                                     │
│                    ┌──────────────┐                              │
│                    │ Brain Library│                              │
│                    │ (PostgreSQL) │                              │
│                    └──────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ↓ (Model Pull)
┌─────────────────────────────────────────────────────────────────┐
│                  EXECUTION LAYER (Hamilton)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Model        │  │ Consensus    │  │ Risk         │          │
│  │ Loader       │→ │ Engine       │→ │ Manager      │          │
│  │ (Brain Lib)  │  │ (23 Engines) │  │ (Presets)    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                  │                  │                  │
│         └──────────────────┼──────────────────┘                  │
│                            ↓                                     │
│                    ┌──────────────┐                              │
│                    │ Cost Model   │                              │
│                    │ (Real-time)  │                              │
│                    └──────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Component Architecture

### 1. Distributed Training Orchestrator

**Location**: `src/cloud/training/orchestrator/distributed_trainer.py`

**Responsibilities**:
- Coordinate Ray/Dask cluster for distributed training
- Manage GPU allocation per coin/regime/timeframe
- Handle async training jobs with retries
- Track training progress and metrics

**Key Features**:
- Async task queue for 400+ coins
- Automatic GPU allocation and cleanup
- Progress tracking and reporting
- Failure recovery and retry logic

**Configuration**:
```yaml
training:
  distributed:
    backend: "ray"  # or "dask"
    num_workers: 8
    gpus_per_worker: 1
    max_concurrent_training_jobs: 16
    retry_attempts: 3
    timeout_seconds: 3600
```

### 2. Coin Selection & Liquidity Filter

**Location**: `src/cloud/training/services/coin_selector.py`

**Responsibilities**:
- Fetch all available Binance coins (400+)
- Rank by daily liquidity metrics
- Filter by spread, volume, age
- Dynamic daily ranking updates

**Key Features**:
- Real-time liquidity ranking
- Spread threshold filtering
- Volume-based selection
- Age and stability checks

**Configuration**:
```yaml
coin_selection:
  min_daily_volume_usd: 1000000
  max_spread_bps: 8
  min_age_days: 30
  ranking_method: "liquidity_score"  # or "volume", "spread"
  update_frequency: "daily"
  max_coins: 400  # No hard limit, but configurable throttle
```

### 3. Model Versioning & Registry

**Location**: `src/cloud/training/services/model_versioning.py`

**Responsibilities**:
- Version models per coin/regime/timeframe
- Store in Brain Library with metadata
- Track performance metrics
- Daily best model selection

**Key Features**:
- Semantic versioning (coin-regime-timeframe-version)
- Automatic best model selection
- Performance comparison
- Rollback capability

**Model Naming Convention**:
```
{coin}_{regime}_{timeframe}_v{version}_{timestamp}
Example: BTC_TREND_1h_v1_20250115_020000
```

### 4. Enhanced Consensus Engine

**Location**: `src/cloud/training/consensus/enhanced_consensus.py`

**Responsibilities**:
- Combine 23 engine votes with reliability weights
- Apply correlation penalties for similar engines
- Adaptive thresholds based on regime
- Confidence calculation

**Key Features**:
- Reliability-weighted voting
- Correlation matrix for engine diversity
- Regime-specific thresholds
- Consensus confidence scoring

**Configuration**:
```yaml
consensus:
  num_engines: 23
  min_agreement_threshold: 0.6
  reliability_decay_factor: 0.95
  correlation_penalty_threshold: 0.8
  adaptive_thresholds:
    TREND: 0.5
    RANGE: 0.55
    PANIC: 0.65
    ILLIQUID: 0.7
```

### 5. Regime Gating System

**Location**: `src/cloud/training/regime/regime_gate.py`

**Responsibilities**:
- Detect current market regime
- Enable/disable engines per regime
- Track engine performance by regime
- Weekly leaderboard updates

**Key Features**:
- Hard gates: Only approved engines per regime
- Soft gates: Weight adjustment based on regime
- Performance tracking by regime
- Automatic engine enablement/disablement

**Regime-Engine Mapping**:
```yaml
regime_gates:
  TREND:
    enabled_engines: ["trend_v1", "trend_v2", "breakout", "momentum"]
    disabled_engines: ["mean_reversion", "range_trading"]
  RANGE:
    enabled_engines: ["mean_reversion", "range_trading", "support_resistance"]
    disabled_engines: ["trend_v1", "trend_v2"]
  PANIC:
    enabled_engines: ["scalping", "volatility", "liquidity"]
    disabled_engines: ["trend_v1", "trend_v2", "breakout"]
  ILLIQUID:
    enabled_engines: []  # No trading in illiquid markets
```

### 6. Real-Time Cost Model

**Location**: `src/cloud/training/costs/realtime_cost_model.py`

**Responsibilities**:
- Calculate fees per venue (Binance, Coinbase, etc.)
- Model spreads in real-time
- Estimate slippage based on order size
- Calculate funding costs for perps
- Safety margin enforcement

**Key Features**:
- Venue-specific fee schedules
- Real-time spread updates via WebSocket
- Slippage modeling based on participation rate
- Funding cost tracking
- Edge-after-cost calculation

**Configuration**:
```yaml
cost_model:
  update_interval_seconds: 60
  spread_source: "websocket"  # or "api", "static"
  fee_schedule:
    binance:
      maker_bps: 2.0
      taker_bps: 4.0
      funding_bps_per_8h: 1.0
    coinbase:
      maker_bps: 5.0
      taker_bps: 5.0
  min_edge_after_cost_bps: 3.0
  slippage_model:
    participation_rate: 0.1
    impact_factor: 0.5
```

### 7. Risk Preset System

**Location**: `src/cloud/training/risk/risk_presets.py`

**Responsibilities**:
- Define risk presets (conservative, balanced, aggressive)
- Enforce limits in all simulations
- Per-trade risk calculation
- Daily loss limits

**Key Features**:
- Three preset levels
- Configurable per-preset limits
- Automatic enforcement in backtests
- Real-time risk monitoring

**Configuration**:
```yaml
risk_presets:
  conservative:
    per_trade_risk_pct: 0.5
    daily_loss_limit_pct: 1.5
    max_leverage: 1.5
    min_confidence: 0.65
  balanced:
    per_trade_risk_pct: 1.0
    daily_loss_limit_pct: 2.5
    max_leverage: 2.0
    min_confidence: 0.55
  aggressive:
    per_trade_risk_pct: 1.5
    daily_loss_limit_pct: 4.0
    max_leverage: 3.0
    min_confidence: 0.50
```

### 8. Walk-Forward Validation

**Location**: `src/cloud/training/validation/walk_forward.py`

**Responsibilities**:
- Purged cross-validation to prevent data leakage
- Multiple walk-forward windows
- Out-of-sample testing
- Performance tracking across windows

**Key Features**:
- Purge gaps between train/test sets
- Multiple validation windows
- OOS performance tracking
- Leakage detection

**Configuration**:
```yaml
validation:
  walk_forward:
    train_days: 20
    test_days: 5
    purge_days: 2
    min_windows: 5
    min_test_trades: 100
```

### 9. Shadow Testing System

**Location**: `src/cloud/training/deployment/shadow_tester.py`

**Responsibilities**:
- Run new models in shadow mode
- Compare with baseline models
- Statistical significance testing
- Automatic promotion/rejection

**Key Features**:
- Shadow trading for 2-4 weeks
- Performance comparison
- Statistical tests (t-test, Sharpe comparison)
- Automatic promotion criteria

**Configuration**:
```yaml
shadow_testing:
  enabled: true
  min_duration_days: 14
  min_trades: 100
  promotion_criteria:
    min_sharpe_improvement: 0.2
    min_win_rate_improvement: 0.05
    statistical_significance: 0.95
```

### 10. DecisionEvent Logging

**Location**: `src/cloud/training/observability/decision_logger.py`

**Responsibilities**:
- Log every model decision with structured data
- Track engine votes and consensus
- Store costs and expected edge
- Link to trade outcomes

**Key Features**:
- Structured JSON logs
- Complete decision context
- Performance attribution
- Queryable event store

**Event Schema**:
```python
@dataclass
class DecisionEvent:
    timestamp: datetime
    symbol: str
    regime: str
    engine_votes: List[EngineVote]
    consensus_score: float
    action: str  # "buy", "sell", "hold"
    expected_edge_bps: float
    costs_bps: float
    net_edge_bps: float
    risk_preset: str
    metadata: Dict[str, Any]
```

### 11. Prometheus Metrics

**Location**: `src/cloud/training/observability/prometheus_metrics.py`

**Responsibilities**:
- Expose metrics for Prometheus scraping
- Track PnL by engine, regime, venue
- Monitor latency and error rates
- System health metrics

**Key Metrics**:
- `engine_pnl_usd{engine, symbol, regime}`
- `engine_latency_ms{engine, operation}`
- `engine_error_rate{engine, error_type}`
- `consensus_confidence{regime}`
- `cost_model_spread_bps{venue, symbol}`
- `training_job_duration_seconds{coin, regime}`

### 12. Grafana Dashboards

**Location**: `engine/observability/grafana/dashboards/`

**Dashboard Panels**:
- PnL by Engine (time series)
- Latency Percentiles (histogram)
- Error Rate by Component (bar chart)
- Consensus Confidence Distribution (heatmap)
- Cost Breakdown by Venue (pie chart)
- Training Job Status (table)

---

## 🔄 Data Flow

### Training Flow

```
1. Daily Coin Selection
   ↓
2. Liquidity Ranking & Filtering
   ↓
3. Distributed Training Jobs (Ray/Dask)
   ├─ Coin 1 → GPU 1 → Train (regime, timeframe)
   ├─ Coin 2 → GPU 2 → Train (regime, timeframe)
   └─ ... (400 coins in parallel)
   ↓
4. Walk-Forward Validation
   ↓
5. Model Evaluation (Sharpe, drawdown, hit rate, edge-after-cost)
   ↓
6. Model Versioning & Storage (Brain Library)
   ↓
7. Best Model Selection
   ↓
8. Shadow Testing (if new model)
   ↓
9. Daily Push to Brain Library (for Hamilton)
```

### Execution Flow (Hamilton)

```
1. Load Models from Brain Library
   ↓
2. Fetch Real-Time Market Data
   ↓
3. Regime Detection
   ↓
4. Regime Gating (enable/disable engines)
   ↓
5. Run 23 Engines (parallel)
   ↓
6. Consensus Voting (with reliability weights)
   ↓
7. Cost Model Check (edge-after-cost)
   ↓
8. Risk Preset Enforcement
   ↓
9. DecisionEvent Logging
   ↓
10. Trade Execution (if passes all checks)
```

---

## 📁 File Structure

```
engine/
├── src/
│   └── cloud/
│       └── training/
│           ├── orchestrator/
│           │   ├── distributed_trainer.py      # Ray/Dask orchestration
│           │   └── training_coordinator.py     # Job coordination
│           ├── services/
│           │   ├── coin_selector.py            # Liquidity-based selection
│           │   ├── model_versioning.py         # Version management
│           │   └── daily_model_pusher.py       # Brain Library push
│           ├── consensus/
│           │   ├── enhanced_consensus.py       # 23-engine voting
│           │   └── reliability_tracker.py      # Engine reliability
│           ├── regime/
│           │   ├── regime_gate.py              # Engine enablement
│           │   └── regime_detector.py          # Regime classification
│           ├── costs/
│           │   ├── realtime_cost_model.py      # Real-time costs
│           │   └── venue_config.py             # Exchange configs
│           ├── risk/
│           │   ├── risk_presets.py             # Preset definitions
│           │   └── risk_enforcer.py            # Limit enforcement
│           ├── validation/
│           │   ├── walk_forward.py             # Purged CV
│           │   └── leakage_detector.py         # Data leakage checks
│           ├── deployment/
│           │   └── shadow_tester.py            # Shadow testing
│           └── observability/
│               ├── decision_logger.py          # DecisionEvent logging
│               ├── prometheus_metrics.py       # Metrics export
│               └── event_store.py              # Event storage
├── config/
│   ├── scalable_engine.yaml                    # Main config
│   ├── risk_presets.yaml                       # Risk configurations
│   └── consensus_config.yaml                   # Consensus settings
└── tests/
    └── test_scalable_engine/                   # Comprehensive tests
```

---

## 🔧 Configuration Examples

### Main Configuration (`config/scalable_engine.yaml`)

```yaml
engine:
  max_coins: 400  # No hard limit, but configurable
  active_coins: 20  # Start small, scale up
  max_concurrent_trades: 500

training:
  distributed:
    backend: "ray"
    num_workers: 8
    gpus_per_worker: 1
    max_concurrent_jobs: 16
  
  model_versioning:
    storage_path: "/models/trained"
    brain_library_dsn: "postgresql://..."
    daily_push_enabled: true
    push_time_utc: "03:00"

coin_selection:
  min_daily_volume_usd: 1000000
  max_spread_bps: 8
  ranking_update_frequency: "daily"

consensus:
  num_engines: 23
  reliability_decay: 0.95
  correlation_penalty_threshold: 0.8

regime_gates:
  enabled: true
  leaderboard_refresh_days: 7

cost_model:
  real_time_updates: true
  update_interval_seconds: 60
  min_edge_after_cost_bps: 3.0

risk:
  preset: "balanced"  # conservative, balanced, aggressive
  enforce_in_simulations: true

observability:
  prometheus:
    enabled: true
    port: 9090
  decision_events:
    enabled: true
    storage_path: "/logs/decision_events"
```

---

## 🧪 Testing Strategy

### Unit Tests
- Each component tested in isolation
- Mock dependencies
- 80%+ coverage requirement

### Integration Tests
- Test component interactions
- End-to-end training flow
- Brain Library integration

### Performance Tests
- 400-coin training load
- Distributed training efficiency
- Latency benchmarks

### Test Structure
```
tests/
├── test_orchestrator/
│   ├── test_distributed_trainer.py
│   └── test_training_coordinator.py
├── test_consensus/
│   ├── test_enhanced_consensus.py
│   └── test_reliability_tracker.py
└── test_integration/
    ├── test_full_training_flow.py
    └── test_brain_library_integration.py
```

---

## 🚀 Deployment

### RunPod GPU Setup
- Ray cluster on RunPod GPUs
- Automatic worker discovery
- GPU allocation per job

### Daily Retrain Job
- Scheduled at 02:00 UTC
- Runs distributed training
- Pushes best models at 03:00 UTC

### Monitoring
- Prometheus scraping every 15s
- Grafana dashboards
- Alerting on errors

---

## 📊 Success Metrics

### Training Metrics
- Models trained per day: 400+ (all coins × regimes × timeframes)
- Training time per coin: < 30 minutes
- Model storage: Versioned in Brain Library

### Execution Metrics
- Consensus latency: < 100ms
- Cost model accuracy: ±0.5 bps
- DecisionEvent logging: 100% coverage

### System Metrics
- Test coverage: > 80%
- Uptime: > 99.9%
- Error rate: < 0.1%

---

## 🔄 Migration Path

### Phase 1: Foundation (Week 1)
- Distributed training orchestrator
- Coin selection system
- Model versioning

### Phase 2: Consensus & Regime (Week 2)
- Enhanced consensus engine
- Regime gating
- Reliability tracking

### Phase 3: Costs & Risk (Week 3)
- Real-time cost model
- Risk presets
- Validation systems

### Phase 4: Observability (Week 4)
- DecisionEvent logging
- Prometheus metrics
- Grafana dashboards

### Phase 5: Testing & Deployment (Week 5)
- Comprehensive test suite
- RunPod deployment
- Production rollout

---

## 📚 References

- [Ray Documentation](https://docs.ray.io/)
- [Dask Documentation](https://docs.dask.org/)
- [Prometheus Metrics](https://prometheus.io/docs/concepts/metric_types/)
- [Grafana Dashboards](https://grafana.com/docs/grafana/latest/dashboards/)

---

**Last Updated:** 2025-01-XX  
**Maintained By:** Engine Architecture Team

