# Huracan Engine Architecture - Learning System

**Last Updated**: November 6, 2025

---

## 🎯 Core Concept

**The Engine = 100% Shadow Trading + Learning System**

- ❌ Does NOT make live trades
- ❌ Does NOT spend real money
- ✅ Only paper trades (shadow trades)
- ✅ Learns from market data
- ✅ Trains and improves models
- ✅ Exports trained models for Hamilton to use

**Hamilton = Live Trading System**

- ✅ Takes trained models FROM the Engine
- ✅ Makes real trades with real money
- ✅ Uses the models the Engine created
- ❌ Does NOT train models (just uses them)

---

## 📊 Data Flow

```
Market Data
    ↓
[Engine - Learning System]
    ├─ Receives signals
    ├─ Gates evaluate (meta-label, cost, confidence, regime, spread, volume)
    ├─ Would this pass? → SHADOW TRADE
    │   ├─ Track: What would have happened?
    │   ├─ Outcome: Win/Loss/Breakeven
    │   └─ Learn: Update models based on outcomes
    ├─ Train models on historical + shadow trade data
    ├─ Improve: Feature importance, calibration, regime detection
    └─ Export: Trained models (*.pkl) with SHA256 IDs
        ↓
[Hamilton - Live Trading System]
    ├─ Import: Latest trained models from Engine
    ├─ Receives signals
    ├─ Gates evaluate using Engine's models
    ├─ Pass? → REAL TRADE (with real money)
    └─ Reports: Trade outcomes back to Engine for learning
```

---

## 🔄 Shadow Trading Workflow

### 1. Signal Received (Engine)
```
Market: ETH-USD @ $2,045.50
Signal: LONG (confidence: 0.78)
Regime: TREND
```

### 2. Gates Evaluate
```
✓ meta_label_gate: PASS (prob 0.78 > threshold 0.45)
✓ cost_gate: PASS (expected return 15 bps > costs 8 bps + buffer 3 bps)
✓ confidence_gate: PASS (confidence 0.82 > min 0.55)
✓ regime_gate: PASS (TREND is allowed)
✓ spread_gate: PASS (spread 4.2 bps < max 5.0 bps)
✓ volume_gate: PASS (volume 1.2x > min 0.5x)
```

### 3. Shadow Trade Executed (Engine)
```
Shadow Trade ID: shadow_001
Type: PAPER TRADE (no real money)
Entry: $2,045.50
Size: £100 (simulated)
Fee: £0.10 (simulated)

→ Track this trade in shadow_trades table
```

### 4. Monitor Outcome (Engine)
```
Wait for exit condition:
  - Hit TP: $2,052.30 (+13 bps) ✓
  - Hit SL: $2,038.40 (-35 bps)
  - Timeout: 15 minutes

Actual Outcome: Hit TP after 8 minutes
Shadow P&L: +£0.13 (13 bps)
Result: WIN
```

### 5. Learn from Outcome (Engine)
```
Update meta-label model:
  - Feature vector at entry time
  - Actual outcome: WIN
  - Update training dataset

Update feature importance:
  - Which features predicted this correctly?
  - Adjust feature weights

Update calibration:
  - Predicted 78% win probability
  - Actual: WIN (100%)
  - Adjust calibration curve

Save shadow trade:
  - Log in shadow_trades table
  - Available for analysis
  - Used in next training session
```

### 6. Model Export (Engine → Hamilton)
```
After training session:
  - New model trained on 5,000 samples (historical + shadow trades)
  - Performance: AUC 0.72 → 0.74 (+0.02 improvement)
  - Save model with SHA256 ID: sha256:abc123...
  - Export to: models/meta_label_v42.pkl

Hamilton imports:
  - Reads: models/meta_label_v42.pkl
  - Validates: SHA256 checksum matches
  - Deploys: Uses for real trading decisions
```

---

## 🗂️ Database Schema

### Shadow Trades (Engine Only)

**shadow_trades** table:
```sql
CREATE TABLE shadow_trades (
    shadow_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,

    -- Signal info
    symbol TEXT NOT NULL,
    mode TEXT NOT NULL,  -- 'scalp' or 'runner'
    side TEXT NOT NULL,  -- 'long' or 'short'
    entry_price REAL NOT NULL,

    -- Why this was a shadow trade (not live)
    trade_type TEXT DEFAULT 'SHADOW',  -- Always 'SHADOW' in Engine

    -- Simulated execution
    simulated_size_gbp REAL,
    simulated_fee_gbp REAL,

    -- Outcome (what happened)
    exit_price REAL,
    exit_reason TEXT,  -- 'TP', 'SL', 'timeout'
    shadow_pnl_bps REAL,
    duration_sec REAL,

    -- Did model predict correctly?
    predicted_win_prob REAL,
    actual_outcome TEXT,  -- 'WIN', 'LOSS', 'BREAKEVEN'
    prediction_correct BOOLEAN,

    -- Used for learning
    features_json TEXT,  -- All features at entry time
    used_in_training BOOLEAN DEFAULT FALSE,

    -- Metadata
    model_id TEXT,  -- Which model made this prediction
    regime TEXT,
    tags TEXT
);
```

**trades** table (for Hamilton's live trades):
```sql
CREATE TABLE trades (
    trade_id TEXT PRIMARY KEY,
    ts_open TEXT NOT NULL,

    -- Trade info
    symbol TEXT NOT NULL,
    mode TEXT NOT NULL,
    side TEXT NOT NULL,

    -- Execution
    entry_price REAL NOT NULL,
    exit_price REAL,

    -- Real money
    size_gbp REAL NOT NULL,
    fee_entry_gbp REAL,
    fee_exit_gbp REAL,

    -- P&L (REAL MONEY)
    pnl_gbp REAL,
    return_bps REAL,

    -- Source
    trade_type TEXT DEFAULT 'LIVE',  -- Always 'LIVE' for Hamilton
    model_id TEXT,  -- Which Engine model was used

    -- Status
    status TEXT NOT NULL,  -- 'open', 'closed'

    -- Link back to Engine
    engine_shadow_id TEXT,  -- If this was based on a shadow trade

    -- Metadata
    tags TEXT
);
```

---

## 📈 Learning Pipeline

### Daily Learning Cycle (Engine)

**Morning (00:00 UTC)**:
1. Collect previous day's shadow trades
2. Collect Hamilton's live trade outcomes
3. Combine into training dataset
4. Train meta-label model
5. Train regime detector
6. Update feature importance
7. Calibrate probabilities
8. Export new models

**Throughout Day**:
1. Receive market signals
2. Evaluate with current models
3. Execute shadow trades (paper only)
4. Track outcomes
5. Log everything for tonight's training

**Evening (23:00 UTC)**:
1. Generate daily summary:
   - How many shadow trades?
   - Win rate (shadow trades)
   - What did we learn?
   - Feature importance changes
   - Model improvements
2. AI Council generates summary (Day 3 feature)
3. Send notification with summary

---

## 🎓 What Engine Tracks (Observability)

### 1. **Shadow Trading Activity**
- Number of signals received
- Number passed gates (would have traded)
- Number failed gates (blocked)
- Shadow trade outcomes (wins/losses)
- Shadow P&L (simulated)

### 2. **Learning Progress**
- Training sessions (daily)
- Model improvements (AUC, ECE, Brier)
- Feature importance evolution
- Calibration quality
- Data processed (how many samples)

### 3. **Gate Performance**
- Pass rates (too strict? too loose?)
- Gate accuracy (were blocks correct?)
- Shadow trades blocked:
  - Good blocks: Blocked a loser ✓
  - Bad blocks: Blocked a winner ✗
- Recommendations: Adjust thresholds?

### 4. **Model Readiness**
- Is model ready for Hamilton?
- Performance metrics meet targets?
- Properly calibrated?
- Sufficient training data?
- Model export history

### 5. **System Health**
- Event queue status
- Writer lag
- Error rates
- Data integrity

---

## 🔗 Engine ↔ Hamilton Integration

### Model Export (Engine → Hamilton)

```python
# Engine: After training
from observability.core.registry import ModelRegistry

registry = ModelRegistry()
model_id = registry.register_model(
    model=meta_label_model,
    code_git_sha="abc123",
    data_snapshot_id="snapshot_2025-11-06",
    metrics={"auc": 0.74, "ece": 0.055},
    notes="Trained on 5,000 samples (3,000 historical + 2,000 shadow trades)"
)

# Export for Hamilton
registry.export_model(
    model_id=model_id,
    export_path="/shared/models/meta_label_latest.pkl"
)

# Notify Hamilton (webhook or file signal)
notify_hamilton(model_id, metrics)
```

### Model Import (Hamilton ← Engine)

```python
# Hamilton: Import latest model
from observability.core.registry import ModelRegistry

registry = ModelRegistry(base_path="/shared/models")

# Load latest model
model_id, model = registry.load_latest_model("meta_label")

# Validate checksum
assert registry.verify_model(model_id), "Model checksum mismatch!"

# Use for live trading
gate = MetaLabelGate(model=model, threshold=0.45)
decision = gate.evaluate(features)
```

### Outcome Feedback (Hamilton → Engine)

```python
# Hamilton: After live trade closes
import requests

# Send outcome to Engine for learning
requests.post("http://engine:5000/api/trade_outcome", json={
    "trade_id": "live_001",
    "model_id": model_id,
    "features": features_at_entry,
    "outcome": "WIN",
    "pnl_bps": 13.2,
    "duration_sec": 480
})

# Engine receives this and adds to next training batch
```

---

## 📊 Key Metrics (Engine Only)

### Shadow Trading Metrics
```yaml
shadow_trades_daily:
  description: "Number of shadow trades executed per day"
  target: {min: 20, ideal: 50, max: 200}

shadow_win_rate:
  description: "Win rate of shadow trades (paper only)"
  target:
    scalp: {min: 0.70, ideal: 0.74, max: 0.78}
    runner: {min: 0.87, ideal: 0.90, max: 0.95}

shadow_pnl_daily:
  description: "Daily P&L from shadow trades (simulated)"
  target: {min: 50, ideal: 100, max: 200}  # bps, not GBP
  note: "This is SIMULATED - no real money"
```

### Learning Metrics
```yaml
training_frequency:
  description: "How often models are retrained"
  target: "Daily (00:00 UTC)"

model_improvement:
  description: "AUC improvement per training session"
  target: {min: 0.005, ideal: 0.01}  # +0.5% to +1%

training_data_size:
  description: "Number of samples used for training"
  target: {min: 1000, ideal: 5000, max: 50000}

feature_importance_stability:
  description: "How much feature importance changes"
  target: {max: 0.30}  # <30% change per update
```

### Model Export Metrics
```yaml
models_exported:
  description: "Models exported to Hamilton"
  target: "Daily"

model_validation:
  description: "All exported models pass validation"
  target: 100%

hamilton_adoption:
  description: "Is Hamilton using latest models?"
  target: "Within 1 day of export"
```

---

## 🎯 Observability Focus

### What We Track (Engine)

✅ **Shadow Trades**:
- Every signal that passes gates = shadow trade
- Track outcome: Did our prediction work?
- Learn: Update models based on outcomes

✅ **Learning Progress**:
- What did we learn today?
- How much better are our models?
- Which features became more/less important?

✅ **Model Readiness**:
- Is the model ready for Hamilton?
- Are metrics improving?
- Is calibration good?

✅ **Gate Tuning**:
- Are gates too strict? (blocking too many)
- Are gates too loose? (passing too many)
- Were rejections correct? (good/bad blocks)

### What We DON'T Track (Engine)

❌ **Live Trade Execution**: Hamilton handles this
❌ **Real Money P&L**: Hamilton tracks this
❌ **Exchange API Calls**: Hamilton handles this
❌ **Order Book Management**: Hamilton handles this

---

## 🔄 Day 2 Module Updates

With this clarification, our modules should focus on:

1. **metrics_computer.py**: Compute shadow trade metrics (not live trade metrics)
2. **model_evolution.py**: Track model improvements for Hamilton export
3. **insight_aggregator.py**: "What did we learn? Are models ready?"
4. **market_context_logger.py**: Market conditions during shadow trades
5. **live_feed.py**: Real-time learning activity (not live trading activity)

---

## 💡 Summary

**Engine = Learning Lab**
- Shadow trades (paper only)
- Model training
- Continuous improvement
- Model export

**Hamilton = Production Trading**
- Real trades (real money)
- Uses Engine's models
- Reports outcomes back to Engine

**Observability System = Track Learning**
- Shadow trade outcomes
- Model improvements
- Feature importance
- Readiness for Hamilton

**This is a Learning System, Not a Trading System.**
