# Advanced Trading Infrastructure

## Overview

This document describes the advanced trading infrastructure components implemented in the Huracan Engine, including statistical validation, risk management, latency monitoring, and feedback systems.

## Components

### 1. Statistical Validation Framework

#### Permutation Testing Module
**Location**: `src/cloud/training/validation/permutation_testing.py`

Validates whether a strategy's backtest performance is statistically significant or random.

**Features**:
- Randomly shuffle trade sequences 1,000+ times
- Compare actual Sharpe/return distribution to random permutations
- Calculate p-values for statistical significance
- Pass/fail models based on 99th percentile threshold
- Integration with Council voting system

**Usage**:
```python
from cloud.training.validation.permutation_testing import PermutationTester

tester = PermutationTester(num_permutations=1000)
result = tester.test_strategy(
    trades=[0.01, 0.02, -0.01, ...],  # Trade returns
    confidence_level=0.99
)

if result.is_significant:
    print("Strategy passed permutation test!")
    print(f"P-value: {result.p_value_sharpe}")
    print(f"Percentile rank: {result.percentile_rank_sharpe}")
```

#### Robustness Analyzer
**Location**: `src/cloud/training/validation/permutation_testing.py`

Analyzes model robustness using Monte Carlo and permutation testing.

**Features**:
- Sensitivity to randomization
- Noise injection testing
- Visualize robustness metrics
- Integration with Council voting

**Usage**:
```python
from cloud.training.validation.permutation_testing import RobustnessAnalyzer

analyzer = RobustnessAnalyzer()
results = analyzer.analyze_model_robustness(
    model_id="model_1",
    trades=[0.01, 0.02, ...],
    noise_levels=[0.01, 0.05, 0.10]
)

print(f"Robustness score: {results['robustness_score']}")
print(f"Recommendation: {results['recommendation']}")
```

#### Walk-Forward Testing
**Location**: `src/cloud/training/validation/walk_forward_testing.py`

Automates walk-forward testing with rolling in-sample and out-of-sample segments.

**Features**:
- Rolling window backtests
- In-sample / out-of-sample splitting
- Segment metrics logging
- Stability tracking
- Integration with Log Book

**Usage**:
```python
from cloud.training.validation.walk_forward_testing import WalkForwardTester

tester = WalkForwardTester(
    in_sample_days=180,
    out_of_sample_days=30,
    step_size_days=30
)

result = tester.test_model(
    model_id="model_1",
    data=train_data,
    train_fn=my_train_function,
    test_fn=my_test_function
)

if result.is_robust:
    print("Model passed walk-forward test!")
    print(f"Mean stability: {result.stability_metrics['mean_stability']}")
```

### 2. Trust Index System

**Location**: `src/cloud/training/models/trust_index.py`

Calculates a confidence weight blending model accuracy, drawdown duration, and recovery slope.

**Features**:
- Model accuracy tracking
- Drawdown duration monitoring
- Recovery slope calculation
- Trust index calculation (0-1)
- Capital allocation adjustment
- Integration with Council voting

**Usage**:
```python
from cloud.training.models.trust_index import TrustIndexCalculator

calculator = TrustIndexCalculator()
metrics = calculator.calculate_trust_index(
    model_id="model_1",
    accuracy=0.75,
    drawdown_duration_days=5,
    recovery_slope=0.02,
    recent_performance=0.15
)

print(f"Trust index: {metrics.trust_index}")
print(f"Capital allocation: {metrics.capital_allocation}")
print(f"Recommendation: {metrics.recommendation}")
```

### 3. Pre-Trade Risk Engine

**Location**: `src/cloud/training/risk/pre_trade_risk.py`

Fast risk validation layer between Hamilton's trade signal and final API send.

**Features**:
- Exposure limit checks
- Position size validation
- Daily loss limits
- Concentration limits
- Leverage limits
- Fast validation (< 1ms typical)

**Usage**:
```python
from cloud.training.risk.pre_trade_risk import PreTradeRiskEngine, RiskLimits

limits = RiskLimits(
    max_position_size_usd=10000.0,
    max_daily_loss_usd=1000.0,
    max_leverage=1.0
)

engine = PreTradeRiskEngine(limits=limits)
result = engine.validate_trade(
    symbol="BTCUSDT",
    direction="buy",
    size_usd=1000.0,
    current_price=50000.0,
    spread_bps=5.0,
    liquidity_score=0.8
)

if result.approved:
    execute_trade(size=result.recommended_size)
else:
    print(f"Trade rejected: {result.rejection_reason}")
```

### 4. Latency Monitoring System

**Location**: `src/cloud/training/monitoring/latency_monitor.py`

Tracks nanosecond-level metrics for every trade, module latency, and model inference duration.

**Features**:
- Nanosecond timestamps
- Tick-to-trade time tracking
- Module latency tracking
- Model inference duration
- Latency dashboards
- Slowdown detection

**Usage**:
```python
from cloud.training.monitoring.latency_monitor import LatencyMonitor

monitor = LatencyMonitor()

# Track module latency
with monitor.track("model_inference"):
    model.predict(data)

# Get metrics
metrics = monitor.get_metrics("model_inference")
print(f"Mean latency: {metrics['mean_ms']} ms")
print(f"P95 latency: {metrics['p95_ms']} ms")

# Record tick-to-trade
monitor.record_tick_to_trade(
    symbol="BTCUSDT",
    tick_timestamp_ns=tick_time,
    trade_timestamp_ns=trade_time,
    module_breakdown={"signal_gen": 1000000, "risk_check": 500000}
)

# Get tick-to-trade stats
stats = monitor.get_tick_to_trade_stats()
print(f"Mean tick-to-trade: {stats['mean_ms']} ms")
```

### 5. Trade Feedback System

**Location**: `src/cloud/training/feedback/trade_feedback.py`

Automates feedback capture for every fill, slippage, or rejection to update model training datasets.

**Features**:
- Fill feedback capture
- Slippage tracking
- Rejection tracking
- Order lifecycle tracking
- Training dataset updates
- Integration with model retraining

**Usage**:
```python
from cloud.training.feedback.trade_feedback import TradeFeedbackCollector

collector = TradeFeedbackCollector()

# Record fill
collector.record_fill(
    order_id="order_123",
    symbol="BTCUSDT",
    direction="buy",
    requested_price=50000.0,
    filled_price=50010.0,
    filled_size=0.1,
    market_price=50005.0,
    spread_bps=5.0,
    signal_confidence=0.85,
    outcome_bps=20.0
)

# Record rejection
collector.record_rejection(
    order_id="order_124",
    symbol="ETHUSDT",
    direction="sell",
    requested_price=3000.0,
    requested_size=1.0,
    rejection_reason="Insufficient funds",
    rejection_code="INSUFFICIENT_FUNDS"
)

# Get feedback for training
training_data = collector.get_feedback_for_training()
```

## Integration

### With Hamilton (Execution Layer)

```python
# In Hamilton execution layer
from cloud.training.risk.pre_trade_risk import PreTradeRiskEngine
from cloud.training.monitoring.latency_monitor import LatencyMonitor
from cloud.training.feedback.trade_feedback import TradeFeedbackCollector

# Initialize components
risk_engine = PreTradeRiskEngine()
latency_monitor = LatencyMonitor()
feedback_collector = TradeFeedbackCollector()

# Process trade signal
def execute_trade(signal):
    tick_time_ns = time.perf_counter_ns()
    
    # Risk check
    with latency_monitor.track("risk_check"):
        risk_result = risk_engine.validate_trade(
            symbol=signal.symbol,
            direction=signal.direction,
            size_usd=signal.size_usd,
            current_price=signal.price,
            spread_bps=signal.spread_bps
        )
    
    if not risk_result.approved:
        feedback_collector.record_rejection(
            order_id=signal.order_id,
            symbol=signal.symbol,
            direction=signal.direction,
            rejection_reason=risk_result.rejection_reason
        )
        return
    
    # Execute trade
    with latency_monitor.track("order_execution"):
        fill_result = exchange_client.place_order(
            symbol=signal.symbol,
            side=signal.direction,
            size=risk_result.recommended_size,
            price=signal.price
        )
    
    trade_time_ns = time.perf_counter_ns()
    
    # Record tick-to-trade
    latency_monitor.record_tick_to_trade(
        symbol=signal.symbol,
        tick_timestamp_ns=tick_time_ns,
        trade_timestamp_ns=trade_time_ns
    )
    
    # Record feedback
    if fill_result.filled:
        feedback_collector.record_fill(
            order_id=signal.order_id,
            symbol=signal.symbol,
            direction=signal.direction,
            requested_price=signal.price,
            filled_price=fill_result.fill_price,
            filled_size=fill_result.filled_size,
            market_price=signal.price,
            spread_bps=signal.spread_bps
        )
```

### With Council (Voting System)

```python
# In Council voting system
from cloud.training.validation.permutation_testing import PermutationTester
from cloud.training.models.trust_index import TrustIndexCalculator

# Test model significance
tester = PermutationTester()
permutation_result = tester.test_strategy(
    trades=model_trades,
    confidence_level=0.99
)

# Calculate trust index
trust_calculator = TrustIndexCalculator()
trust_metrics = trust_calculator.calculate_trust_index(
    model_id=model_id,
    accuracy=model_accuracy,
    drawdown_duration_days=drawdown_days,
    recovery_slope=recovery_slope,
    recent_performance=recent_sharpe
)

# Weight model vote by trust and significance
vote_weight = (
    trust_metrics.trust_index * 0.6 +
    (1.0 - permutation_result.p_value_sharpe) * 0.4
)

if vote_weight > 0.7 and permutation_result.is_significant:
    approve_model()
```

## Metrics and Monitoring

### Daily Metrics

The system tracks the following metrics daily:

1. **OOS Sharpe and its standard deviation across windows**
2. **Brier score of confidence per regime**
3. **Diversity score in consensus**
4. **Simulator slippage error vs realized costs**
5. **Exploration budget used vs cap**
6. **Early warning precision and recall**
7. **Drawdown days saved by risk advisories**

### Latency Dashboard

The latency monitoring system provides real-time dashboards showing:
- Tick-to-trade delay
- Event throughput
- Module latency breakdown
- Failover health
- Slowdown alerts

## Best Practices

1. **Always run permutation tests** on new models before live trading
2. **Monitor trust index** daily and adjust capital allocation accordingly
3. **Use pre-trade risk checks** for every trade signal
4. **Track latency** to detect slowdowns early
5. **Capture feedback** for all fills, rejections, and slippage
6. **Run walk-forward tests** periodically to validate model stability
7. **Integrate robustness analysis** into model approval process

## Future Enhancements

- [ ] Event-driven pipeline with async market data ingestion
- [ ] In-memory order book with replication
- [ ] Compiled inference layer (ONNX/TorchScript)
- [ ] Smart order router with liquidity-based routing
- [ ] OMS with order lifecycle tracking
- [ ] Continuous optimization from feedback

## References

- Permutation testing: Based on statistical validation principles
- Walk-forward testing: Standard practice in quantitative finance
- Trust index: Inspired by risk-adjusted performance metrics
- Pre-trade risk: Industry-standard risk management
- Latency monitoring: Best practices from HFT firms
- Feedback systems: Continuous learning from live trading

