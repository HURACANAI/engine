# Scalable 400-Coin Engine - Quick Start Guide

**Version:** 3.0  
**Date:** 2025-01-XX

---

## 🚀 Quick Start

This guide will help you get started with the scalable 400-coin engine system.

---

## 📦 Prerequisites

1. **Python 3.11+**
2. **PostgreSQL** (for Brain Library)
3. **Ray or Dask** (for distributed training)
4. **Prometheus** (for metrics, optional)
5. **Grafana** (for dashboards, optional)

---

## 🔧 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install ray  # or dask[distributed]
pip install prometheus-client  # for metrics
```

### 2. Configure Database

Update `config/scalable_engine.yaml`:

```yaml
postgres:
  dsn: "postgresql://user:password@localhost:5432/huracan"
```

### 3. Configure Training

Update `config/scalable_engine.yaml`:

```yaml
training:
  distributed:
    backend: "ray"  # or "dask"
    num_workers: 8
    gpus_per_worker: 1
```

---

## 📝 Basic Usage

### 1. Coin Selection

```python
from src.cloud.training.services.coin_selector import (
    DynamicCoinSelector,
    CoinSelectionConfig,
)

config = CoinSelectionConfig(
    min_daily_volume_usd=1_000_000,
    max_spread_bps=8,
)

selector = DynamicCoinSelector(
    config=config,
    exchange_client=exchange_client,
    metadata_loader=metadata_loader,
)

coins = selector.select_coins()
print(f"Selected {len(coins)} coins: {coins[:10]}")
```

### 2. Distributed Training

```python
from src.cloud.training.orchestrator.distributed_trainer import (
    DistributedTrainer,
    DistributedTrainingConfig,
    TrainingBackend,
)

config = DistributedTrainingConfig(
    backend=TrainingBackend.RAY,
    num_workers=8,
    max_concurrent_jobs=16,
)

trainer = DistributedTrainer(
    config=config,
    model_storage_path=Path("/models/trained"),
    brain_library=brain_library,
)

# Add training jobs
jobs = [
    (coin, regime, timeframe)
    for coin in coins
    for regime in ["TREND", "RANGE", "PANIC"]
    for timeframe in ["1h", "4h", "1d"]
]

trainer.add_training_jobs(jobs)

# Run training
results = await trainer.run_training_loop()
```

### 3. Consensus Voting

```python
from src.cloud.training.consensus.enhanced_consensus import (
    EnhancedConsensusEngine,
    EngineVote,
    MarketRegime,
)

consensus = EnhancedConsensusEngine(
    num_engines=23,
    min_agreement_threshold=0.6,
)

votes = [
    EngineVote(
        engine_id=f"engine_{i}",
        signal=1 if i % 2 == 0 else -1,
        confidence=0.7 + (i * 0.01),
        raw_score=0.5,
    )
    for i in range(23)
]

result = consensus.generate_consensus(
    votes=votes,
    current_regime=MarketRegime.TREND,
)

print(f"Consensus: {result.consensus_signal}, Confidence: {result.consensus_confidence:.2f}")
```

### 4. Regime Gating

```python
from src.cloud.training.regime.regime_gate import RegimeGate, RegimeGateConfig

config = RegimeGateConfig(
    leaderboard_refresh_days=7,
    enable_soft_gates=True,
)

gate = RegimeGate(config=config, brain_library=brain_library)

approved = gate.get_approved_engines(MarketRegime.TREND)
print(f"Approved engines for TREND: {approved}")
```

### 5. Cost Model

```python
from src.cloud.training.costs.realtime_cost_model import (
    RealTimeCostModel,
    RealTimeCostConfig,
)

config = RealTimeCostConfig(
    update_interval_seconds=60,
    min_edge_after_cost_bps=3.0,
)

cost_model = RealTimeCostModel(config=config)
await cost_model.start()

breakdown = cost_model.calculate_costs(
    symbol="BTC/USDT",
    venue="binance",
    side="buy",
    size_usd=1000,
    expected_edge_bps=10.0,
)

print(f"Total cost: {breakdown.total_cost_bps:.2f} bps")
print(f"Net edge: {breakdown.net_edge_bps:.2f} bps")
print(f"Passes safety margin: {breakdown.passes_safety_margin}")
```

### 6. Risk Presets

```python
from src.cloud.training.risk.risk_presets import (
    RiskPresetManager,
    RiskPreset,
)

manager = RiskPresetManager(default_preset=RiskPreset.BALANCED)

allowed, reason = manager.check_trade_allowed(
    confidence=0.6,
    position_size_pct=15.0,
    daily_loss_pct=1.0,
    daily_trade_count=50,
    current_leverage=1.5,
)

if allowed:
    position_size = manager.calculate_position_size(
        equity=100000,
        stop_loss_pct=2.0,
    )
    print(f"Position size: ${position_size:.2f}")
else:
    print(f"Trade not allowed: {reason}")
```

### 7. Prometheus Metrics

```python
from src.cloud.training.observability.prometheus_metrics import get_metrics

metrics = get_metrics()
metrics.start_server()  # Starts HTTP server on port 9090

# Record metrics
metrics.record_engine_pnl("trend_v1", "BTC/USDT", "TREND", 100.0)
metrics.record_consensus_confidence("TREND", 0.75)
metrics.record_engine_latency("trend_v1", "predict", 50.0)
```

---

## 🔄 Daily Workflow

### Training Pipeline

1. **Coin Selection** (02:00 UTC)
   ```python
   coins = selector.select_coins()
   ```

2. **Distributed Training** (02:00-06:00 UTC)
   ```python
   results = await trainer.run_training_loop()
   ```

3. **Model Versioning** (06:00 UTC)
   ```python
   for coin, regime, timeframe in all_combinations:
       best_model = versioning.select_best_model(coin, regime, timeframe)
   ```

4. **Push to Brain Library** (03:00 UTC)
   ```python
   versioning.push_best_models_to_brain_library(coins)
   ```

### Execution Pipeline (Hamilton)

1. **Load Models** from Brain Library
2. **Run Consensus** with 23 engines
3. **Apply Regime Gates**
4. **Check Costs** with real-time cost model
5. **Enforce Risk Presets**
6. **Log Decisions** with DecisionEvent logger

---

## 📊 Monitoring

### Prometheus

Metrics are exposed at `http://localhost:9090/metrics`

Key metrics:
- `engine_pnl_usd{engine, symbol, regime}`
- `engine_latency_ms{engine, operation}`
- `consensus_confidence{regime}`
- `training_jobs_active`

### Grafana

Import dashboard from `observability/grafana/dashboards/engine_overview.json`

Panels:
- Total PnL
- PnL by Engine
- Engine Latency
- Error Rates
- Consensus Confidence
- Cost Breakdown
- Training Job Status

---

## 🧪 Testing

Run tests:

```bash
pytest tests/test_scalable_engine/ -v
pytest tests/test_scalable_engine/ --cov=src --cov-report=html
```

Example test:

```python
@pytest.mark.asyncio
async def test_consensus_voting():
    consensus = EnhancedConsensusEngine()
    votes = [EngineVote(...) for _ in range(23)]
    result = consensus.generate_consensus(votes, MarketRegime.TREND)
    assert result.consensus_confidence > 0.0
```

---

## 🔍 Troubleshooting

### Ray Not Starting

```python
# Check Ray status
import ray
ray.init(address="auto")  # Connect to existing cluster
```

### Database Connection Issues

```python
# Test connection
from src.cloud.training.brain.brain_library import BrainLibrary
brain = BrainLibrary(dsn="postgresql://...")
```

### GPU Allocation

```python
# Check GPU availability
trainer = DistributedTrainer(...)
print(trainer.available_gpus)
```

---

## 📚 Next Steps

1. **Read Architecture Plan**: `docs/architecture/SCALABLE_400_COIN_ARCHITECTURE.md`
2. **Review Configuration**: `config/scalable_engine.yaml`
3. **Explore Examples**: See usage examples above
4. **Run Tests**: `pytest tests/test_scalable_engine/`

---

## 🆘 Support

- **Architecture Docs**: `docs/architecture/`
- **Configuration**: `config/scalable_engine.yaml`
- **Implementation Summary**: `docs/architecture/SCALABLE_ENGINE_IMPLEMENTATION_SUMMARY.md`

---

**Last Updated:** 2025-01-XX

