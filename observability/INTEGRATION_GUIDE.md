# Observability Integration Guide

**How to integrate observability with your Engine simulation**

---

## 🎯 Quick Start - Minimal Integration

Add just 3 lines to your existing simulation:

```python
# At top of file
from observability.analytics.insight_aggregator import InsightAggregator

# After simulation completes
aggregator = InsightAggregator()
insights = aggregator.get_daily_insights(datetime.utcnow().strftime("%Y-%m-%d"))
print(insights['summary'])
```

---

## 📊 Full Integration Example

### Step 1: Initialize Observability

```python
import asyncio
from datetime import datetime
from observability.core.event_logger import EventLogger
from observability.core.io import HybridWriter
from observability.core.schemas import create_signal_event, MarketContext
from observability.analytics.insight_aggregator import InsightAggregator

# Initialize
logger = EventLogger(max_queue_size=10000, batch_size=1000)
writer = HybridWriter()
logger.writer = writer

# Start logger
await logger.start()
```

### Step 2: Log Signals

```python
# In your simulation loop, for each signal:
market_context = MarketContext(
    volatility_1h=0.34,
    spread_bps=4.2,
    liquidity_score=0.82,
    recent_trend_30m=0.008,
    volume_vs_avg=1.2
)

event = create_signal_event(
    symbol="ETH-USD",
    price=current_price,
    features={
        "confidence": signal.confidence,
        "predicted_return": signal.expected_return
    },
    regime=current_regime,  # "TREND", "RANGE", or "PANIC"
    market_context=market_context,
    tags=["simulation", f"step_{step}"]
)

await logger.log(event)
```

### Step 3: Track Shadow Trades

```python
from observability.analytics.trade_journal import TradeJournal

journal = TradeJournal()

# When gate passes (shadow trade executed)
journal.record_trade(
    trade_id=f"shadow_{step:06d}",
    ts_open=datetime.utcnow().isoformat(),
    symbol="ETH-USD",
    mode="scalp",  # or "runner"
    regime=current_regime,
    side="long",  # or "short"
    entry_price=entry_price,
    size_gbp=100.0,  # simulated
    size_asset=100.0 / entry_price,
    fee_entry_gbp=0.10,  # simulated
    model_id="simulation",
    tags=["shadow", "simulation"]
)

# When shadow trade closes
journal.close_trade(
    trade_id=f"shadow_{step:06d}",
    ts_close=datetime.utcnow().isoformat(),
    exit_price=exit_price,
    exit_reason="TP",  # or "SL", "timeout"
    fee_exit_gbp=0.10,
    slippage_bps=1.2
)
```

### Step 4: Get Daily Insights

```python
# After simulation
aggregator = InsightAggregator()
insights = aggregator.get_daily_insights(datetime.utcnow().strftime("%Y-%m-%d"))

print("\n" + "="*80)
print("OBSERVABILITY INSIGHTS")
print("="*80)
print(f"\n{insights['summary']}\n")

print("Recommendations:")
for rec in insights['recommendations']:
    print(f"  • {rec}")

# Stop logger
await logger.stop()
```

---

## 🔧 Integration with test_dual_mode_simulation.py

### Modify Your Simulation File

```python
# Add at top
import asyncio
from observability.core.event_logger import EventLogger
from observability.core.io import HybridWriter
from observability.core.schemas import create_signal_event, MarketContext
from observability.analytics.insight_aggregator import InsightAggregator

# Wrap main simulation function
async def run_simulation_with_observability():
    # Initialize observability
    event_logger = EventLogger(max_queue_size=10000, batch_size=1000)
    writer = HybridWriter(
        duckdb_path="observability/data/simulation_events.duckdb",
        parquet_path="observability/data/simulation_parquet"
    )
    event_logger.writer = writer
    await event_logger.start()

    # Run your existing simulation
    # (wrap your existing code here)

    # Log each signal
    for step in range(500):
        # Your existing signal generation...

        # LOG IT
        event = create_signal_event(
            symbol="ETH-USD",
            price=current_price,
            features={"confidence": 0.5},  # Replace with actual
            regime="TREND",  # Replace with actual
            market_context=MarketContext(
                volatility_1h=0.34,
                spread_bps=4.2,
                liquidity_score=0.82,
                recent_trend_30m=0.008,
                volume_vs_avg=1.2
            ),
            tags=["simulation", f"step_{step}"]
        )
        await event_logger.log(event)

    # After simulation, get insights
    await event_logger.stop()

    aggregator = InsightAggregator()
    insights = aggregator.get_daily_insights(datetime.utcnow().strftime("%Y-%m-%d"))

    print("\n" + "="*80)
    print("OBSERVABILITY INSIGHTS")
    print("="*80)
    print(f"\n{insights['summary']}\n")
    print("Recommendations:")
    for rec in insights['recommendations']:
        print(f"  • {rec}")

# Run it
if __name__ == '__main__':
    asyncio.run(run_simulation_with_observability())
```

---

## 📈 What You'll Get

After integration, you'll see:

```
======================================================================
SIMULATION RESULTS
======================================================================
OVERALL:
  Total Signals: 500
  Total Trades: 0
  Approval Rate: 0.0%

======================================================================
OBSERVABILITY INSIGHTS
======================================================================

⚠️ NO LEARNING TODAY - Engine received 500 signals but gates blocked all trades.
Model trend: → Stable.
Action needed: Loosen gate thresholds to enable shadow trading.

Recommendations:
  • ⚠️ NO SHADOW TRADES - Gates are too strict. Consider lowering meta_label threshold from 0.45 to 0.40 for scalp mode.
  • 🎓 No training sessions today - Schedule daily training at 00:00 UTC.
  • ⏳ Model not ready for Hamilton - Continue training to improve metrics.
```

---

## 🎯 Key Integration Points

### 1. **Signal Logging** (Every signal)
```python
await event_logger.log(create_signal_event(...))
```

### 2. **Gate Decisions** (When evaluating)
```python
from observability.core.schemas import create_gate_event, GateDecision

# After gate evaluation
gate_event = create_gate_event(
    symbol="ETH-USD",
    gate_decision=GateDecision(
        name="meta_label",
        decision="FAIL",  # or "PASS"
        inputs={"probability": 0.42},
        context={"threshold": 0.45},
        timing_ms=2.3
    ),
    mode="scalp",
    features=signal.features,
    tags=["simulation"]
)
await event_logger.log(gate_event)
```

### 3. **Shadow Trades** (When passed gates)
```python
from observability.analytics.trade_journal import TradeJournal

journal = TradeJournal()
journal.record_trade(...)  # Entry
journal.close_trade(...)   # Exit
```

### 4. **Daily Insights** (After simulation)
```python
insights = aggregator.get_daily_insights(date)
print(insights['summary'])
print(insights['recommendations'])
```

---

## 🔍 Debugging Current Issue

Your simulation shows:
- **500 signals received**
- **0 trades executed** (gates blocking 100%)

### Use Observability to Diagnose:

```python
from observability.analytics.gate_explainer import GateExplainer

explainer = GateExplainer()

# For each rejected signal:
explanation = explainer.explain_rejection(
    gate_name="meta_label",
    decision="FAIL",
    inputs={"probability": 0.42},
    context={"threshold": 0.45, "mode": "scalp"}
)

print(explanation.summary)
# ❌ Meta-label REJECTED: Predicted win probability 42.0% below threshold 45.0%

print(explanation.what_to_change)
# Need 3.0% higher predicted win rate

print(explanation.would_pass_if)
# ['Win probability increases by 3.0%', 'Model confidence improves', ...]
```

### Recommendation:
Lower meta_label threshold from 0.45 to 0.35-0.40 to increase shadow trading volume.

---

## 📊 Viewing Results

After integration, query your data:

```python
from observability.analytics.trade_journal import TradeJournal
from observability.analytics.learning_tracker import LearningTracker

# Shadow trade stats
journal = TradeJournal()
stats = journal.get_stats(mode="scalp", days=1)
print(f"Shadow trades: {stats['total_trades']}")
print(f"Win rate: {stats['win_rate']:.1%}")

# Learning progress
tracker = LearningTracker()
summary = tracker.get_daily_summary(date)
print(f"Training sessions: {summary['num_sessions']}")
print(f"Best AUC: {summary['best_metrics']['auc']:.3f}")

# Combined insights
from observability.analytics.insight_aggregator import InsightAggregator

aggregator = InsightAggregator()
insights = aggregator.get_daily_insights(date)
print(insights['summary'])
```

---

## 🚀 Next Steps

1. **Add minimal logging** to see what's happening
2. **Diagnose gate rejections** using gate_explainer
3. **Tune thresholds** based on recommendations
4. **Start shadow trading** to generate learning data
5. **Train models** on shadow trade outcomes
6. **Export to Hamilton** when ready

---

## 📝 Example Output

```
======================================================================
OBSERVABILITY INSIGHTS - 2025-11-06
======================================================================

📊 Shadow Trading:
  Total trades: 0
  Win rate: N/A
  Note: SIMULATED trades (paper only, no real money)

🎓 Learning Progress:
  Training sessions: 0
  Samples processed: 0

🤖 Models:
  Trend: → Stable
  Hamilton ready: False

💡 Recommendations:
  • ⚠️ NO SHADOW TRADES - Gates too strict. Lower meta_label threshold.
  • 🎓 Schedule daily training at 00:00 UTC
  • ⏳ Model not ready for Hamilton - need more training data

Summary:
⚠️ NO LEARNING TODAY - Engine received signals but gates blocked all trades.
Action needed: Loosen gate thresholds to enable shadow trading.
======================================================================
```

---

## ✅ Benefits

After integration you get:

1. **Complete visibility** - See every signal, gate decision, shadow trade
2. **Actionable insights** - "Lower threshold from X to Y"
3. **Learning tracking** - Is the Engine improving?
4. **Hamilton readiness** - When to export models?
5. **Performance debugging** - Which gate is blocking? Why?

**Integration time: ~30 minutes**
**Value: Infinite** (you'll finally know what's happening!)
