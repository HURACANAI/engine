# Huracan V2 Data Pipeline

**Complete training data preparation system with component-specific windows, triple-barrier labeling, and realistic cost modeling.**

---

## Overview

The Huracan V2 data pipeline solves critical problems in ML-based trading:

1. **Lookahead Bias** → Triple-barrier labeling simulates real trade execution
2. **Cost Reality Gap** → Comprehensive TCA (fees + spread + slippage)
3. **Stale Data Problem** → Recency weighting with exponential decay
4. **Label Leakage** → Walk-forward validation with embargo periods
5. **One-Size-Fits-All Training** → Component-specific historical windows

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAW CANDLE DATA                              │
│                 (from CandleDataLoader)                         │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              1. DATA QUALITY PIPELINE                           │
│  • Deduplicate trades                                           │
│  • Fix timestamps (monotonic enforcement)                       │
│  • Remove outliers (flash crashes, bad prints)                  │
│  • Handle gaps (exchange outages)                               │
│  • Apply historical fees                                        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│           2. TRIPLE-BARRIER LABELING                            │
│  • TP/SL/Timeout barriers (no lookahead)                        │
│  • Realistic cost estimation (TCA)                              │
│  • Net P&L calculation                                          │
│  • Exit reason tracking                                         │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│               3. META-LABELING                                  │
│  • Filter to profitable-after-costs trades                      │
│  • Meta-label: 1 if net P&L > costs, 0 otherwise               │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              4. RECENCY WEIGHTING                               │
│  • Exponential decay: weight = exp(-λ × days_ago)              │
│  • Component-specific halflife:                                 │
│    - Scalp: 10 days                                             │
│    - Regime: 60 days                                            │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│           5. MULTI-WINDOW TRAINING                              │
│  • Scalp Core: 60 days (1m, 10d halflife)                      │
│  • Confirm Filter: 120 days (5m, 20d halflife)                 │
│  • Regime Classifier: 365 days (1m, 60d halflife)              │
│  • Risk Context: 730 days (1d, 120d halflife)                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│          6. WALK-FORWARD VALIDATION                             │
│  • Train on [D-N...D-1]                                         │
│  • Embargo period (prevent overlap)                             │
│  • Test on [D]                                                  │
│  • Deploy on [D+1]                                              │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│           TRAINED MODELS (Deployable Artifact)                  │
│  • All component models packaged                                │
│  • Metadata + validation metrics                                │
│  • Ready for Hamilton Pilot                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
engine/src/cloud/engine/
├── data_quality/
│   ├── __init__.py
│   ├── sanity_pipeline.py      # Main cleaning orchestrator
│   ├── fee_schedule.py          # Historical exchange fees
│   └── gap_handler.py           # Data gap detection/filling
│
├── labeling/
│   ├── __init__.py
│   ├── triple_barrier.py        # TP/SL/timeout labeling
│   ├── meta_labeler.py          # Profitable-after-costs filter
│   └── label_schemas.py         # Pydantic configs
│
├── costs/
│   ├── __init__.py
│   └── realistic_tca.py         # Transaction cost analysis
│
├── weighting/
│   ├── __init__.py
│   └── recency_weighter.py      # Exponential decay weighting
│
├── multi_window/
│   ├── __init__.py
│   ├── component_configs.py     # Component-specific configs
│   ├── window_manager.py        # Training window management
│   └── multi_window_trainer.py  # Main training orchestrator
│
├── walk_forward.py              # Walk-forward validation
├── example_pipeline.py          # Complete pipeline demo
└── example_multi_window_training.py  # Multi-window demo
```

---

## Quick Start

### 1. Run Complete Pipeline Example

```bash
python -m cloud.engine.example_pipeline
```

**What it does:**
- Loads/creates sample data
- Cleans data (dedup, outliers, gaps)
- Labels trades (triple-barrier)
- Applies meta-labeling
- Calculates recency weights
- Validates walk-forward
- Shows TCA report

### 2. Run Multi-Window Training Example

```bash
python -m cloud.engine.example_multi_window_training
```

**What it does:**
- Shows component configurations
- Prepares component-specific windows
- Trains all components
- Validates each component
- Packages deployable artifact

---

## Usage Patterns

### Pattern 1: Data Quality Pipeline

```python
from cloud.engine.data_quality import DataSanityPipeline, format_sanity_report

# Initialize pipeline
pipeline = DataSanityPipeline(
    exchange='binance',
    outlier_threshold_pct=0.10,  # Remove >10% moves
    max_gap_minutes=5
)

# Clean data
clean_df, report = pipeline.clean(raw_candles)

# Print report
print(format_sanity_report(report))
```

### Pattern 2: Triple-Barrier Labeling

```python
from cloud.engine.labeling import (
    ScalpLabelConfig,
    TripleBarrierLabeler,
    print_label_statistics
)
from cloud.engine.costs import CostEstimator

# Create config
config = ScalpLabelConfig(
    tp_bps=15.0,   # £1.50 on £1000
    sl_bps=10.0,
    timeout_minutes=30
)

# Initialize labeler with cost estimator
cost_estimator = CostEstimator(exchange='binance')
labeler = TripleBarrierLabeler(
    config=config,
    cost_estimator=cost_estimator
)

# Label trades
labeled_trades = labeler.label_dataframe(
    df=clean_df,
    symbol='BTC/USDT',
    max_labels=1000
)

# Print statistics
stats = labeler.get_statistics(labeled_trades)
print_label_statistics(stats)
```

### Pattern 3: Meta-Labeling

```python
from cloud.engine.labeling import MetaLabeler, print_label_distribution

# Initialize meta-labeler
meta_labeler = MetaLabeler(
    cost_threshold_bps=5.0,  # Must beat costs by 5 bps
    min_pnl_bps=0.0
)

# Apply meta-labels
labeled_trades = meta_labeler.apply(labeled_trades)

# Show distribution
distribution = meta_labeler.get_label_distribution(labeled_trades)
print_label_distribution(distribution)
```

### Pattern 4: Recency Weighting

```python
from cloud.engine.weighting import RecencyWeighter, create_mode_specific_weighter

# Option 1: Custom halflife
weighter = RecencyWeighter(halflife_days=10.0)
weights = weighter.calculate_weights(data)

# Option 2: Mode-specific (recommended)
weighter = create_mode_specific_weighter(mode='scalp')  # 10-day halflife
weights = weighter.calculate_weights_from_labels(labeled_trades)

# Show effective sample size
ess = weighter.get_effective_sample_size(weights)
print(f"Effective samples: {ess:.0f} (out of {len(weights)})")

# Visualize decay
weighter.plot_decay_curve()
```

### Pattern 5: Walk-Forward Validation

```python
from cloud.engine.walk_forward import WalkForwardValidator, print_walk_forward_results

# Initialize validator
validator = WalkForwardValidator(
    train_days=30,
    test_days=1,
    embargo_minutes=30  # Prevent label leakage
)

# Validate
results = validator.validate_with_labels(labeled_trades)
print_walk_forward_results(results)

# Check overfitting
overfit_check = validator.detect_overfitting(results)
print(overfit_check['recommendation'])
```

### Pattern 6: Multi-Window Training

```python
from cloud.engine.multi_window import (
    create_all_component_configs,
    MultiWindowTrainer,
    print_training_results
)

# Define training function
def train_component(window):
    X = window.data.drop(['timestamp', 'meta_label'])
    y = window.data['meta_label']

    model = XGBClassifier()
    model.fit(X, y, sample_weight=window.weights)

    return model

# Train all components
trainer = MultiWindowTrainer()
configs = create_all_component_configs()

results = trainer.train_all_components(
    data=labeled_df,
    train_fn=train_component,
    configs=configs
)

# Show results
print_training_results(results)

# Save artifact
artifact_path = trainer.save_artifact(
    results=results,
    output_dir='./models'
)
```

---

## Component Configurations

| Component | Lookback | Timeframe | Halflife | Why |
|-----------|----------|-----------|----------|-----|
| **Scalp Core** | 60 days | 1m | 10 days | Microstructure changes fast (new algos, fees) |
| **Confirm Filter** | 120 days | 5m | 20 days | Need regime variety (vol regimes, trending vs ranging) |
| **Regime Classifier** | 365 days | 1m | 60 days | Need full market cycles (bull/bear/sideways) |
| **Risk Context** | 730 days | 1d | 120 days | Long-term correlations are sticky |

**Philosophy:**
- Fast components → short windows (patterns fade quickly)
- Slow components → long windows (need full cycles)
- Each weighted by recency appropriate to its horizon

---

## Key Concepts

### 1. Triple-Barrier Labeling

**Problem:** Naive labeling ("did price go up?") creates lookahead bias.

**Solution:** Simulate actual trade execution with three barriers:
- **Take Profit (TP):** Exit at profit target
- **Stop Loss (SL):** Exit at loss limit
- **Timeout:** Exit after max hold time

**Example:**
```python
Entry: BTC @ $45,000
TP: +15 bps → $45,067.50
SL: -10 bps → $44,955.00
Timeout: 30 minutes

# Whichever hits first determines:
# - Exit price
# - Exit time
# - Exit reason
# - Gross P&L
```

### 2. Meta-Labeling

**Problem:** Standard labels ("will price go up?") ignore costs.

**Solution:** Meta-label = "Will trade be profitable AFTER costs?"

```python
Gross P&L: +12 bps
Costs: 8 bps (fees + spread + slippage)
Net P&L: +4 bps
Meta-label: 1 (profitable)
```

### 3. Transaction Cost Analysis (TCA)

**Components:**
1. **Exchange Fees**
   - Maker: 0.02% - 0.10%
   - Taker: 0.04% - 0.20%
   - Historical fee schedules (fees change!)

2. **Spread**
   - Bid-ask spread paid on entry/exit
   - Limit orders: pay ~half spread
   - Market orders: pay full spread

3. **Slippage**
   - Volatility-based: ATR * multiplier
   - Market impact: sqrt(order_size / volume)

**Example:**
```
Entry fee:    2.5 bps
Exit fee:     2.5 bps
Spread:       3.0 bps
Slippage:     1.5 bps
Conservative: +10% buffer
───────────────────────
TOTAL:        10.4 bps
```

**Your model must predict edge > 10.4 bps to profit.**

### 4. Recency Weighting

**Formula:** `weight = exp(-λ × days_ago)` where `λ = ln(2) / halflife`

**Example (10-day halflife):**
```
Today:        weight = 1.000 (100%)
10 days ago:  weight = 0.500 (50%)
20 days ago:  weight = 0.250 (25%)
30 days ago:  weight = 0.125 (12.5%)
60 days ago:  weight = 0.016 (1.6%)
```

**Why:**
- Market microstructure evolves (new algos, fee changes)
- Recent patterns more predictive
- But don't discard old data (regime memory)

### 5. Walk-Forward Validation

**Traditional (WRONG):**
```
[──────── All Data ────────]
  Split randomly into train/test
  → Future data leaks into training!
```

**Walk-Forward (CORRECT):**
```
Train: [D-30...D-1]
Embargo: [D-1 ... D-1 + 30min]  ← Prevent overlap
Test: [D]
Deploy: [D+1]
```

**Embargo prevents:**
- Overlapping trades contaminating test set
- Label leakage from long-hold trades

---

## Integration with Existing System

### Step 1: Replace Data Loading

**Before:**
```python
# Old approach
raw_data = candle_loader.load(symbol='BTC/USDT', days=90)
```

**After:**
```python
from cloud.engine.data_quality import DataSanityPipeline

# New approach
raw_data = candle_loader.load(symbol='BTC/USDT', days=90)

pipeline = DataSanityPipeline(exchange='binance')
clean_data, report = pipeline.clean(raw_data)

print(format_sanity_report(report))
```

### Step 2: Replace Labeling

**Before:**
```python
# Old approach (lookahead bias!)
labels = (df['close'].shift(-1) > df['close']).astype(int)
```

**After:**
```python
from cloud.engine.labeling import TripleBarrierLabeler, ScalpLabelConfig
from cloud.engine.costs import CostEstimator

# New approach (leak-proof)
config = ScalpLabelConfig(tp_bps=15.0, sl_bps=10.0, timeout_minutes=30)
cost_estimator = CostEstimator(exchange='binance')
labeler = TripleBarrierLabeler(config=config, cost_estimator=cost_estimator)

labeled_trades = labeler.label_dataframe(clean_data, symbol='BTC/USDT')
```

### Step 3: Add Recency Weighting

**Before:**
```python
# Old approach (all samples equal weight)
model.fit(X_train, y_train)
```

**After:**
```python
from cloud.engine.weighting import create_mode_specific_weighter

# New approach (recent data weighted higher)
weighter = create_mode_specific_weighter(mode='scalp')
weights = weighter.calculate_weights_from_labels(labeled_trades)

model.fit(X_train, y_train, sample_weight=weights)
```

### Step 4: Multi-Window Training

**Before:**
```python
# Old approach (one model, all data)
model = train_model(all_data)
```

**After:**
```python
from cloud.engine.multi_window import MultiWindowTrainer, create_all_component_configs

# New approach (component-specific windows)
trainer = MultiWindowTrainer()
configs = create_all_component_configs()

results = trainer.train_all_components(
    data=labeled_df,
    train_fn=your_train_function,
    configs=configs
)

scalp_model = results.components['scalp_core'].model
regime_model = results.components['regime_classifier'].model
```

---

## Performance Tips

### 1. Sampling for Speed

For large datasets, sample before labeling:

```python
# Option 1: Sample every Nth candle
df_sampled = df.sample_every_n(100)  # Every 100th candle

# Option 2: Max labels limit
labeled_trades = labeler.label_dataframe(
    df=clean_df,
    symbol='BTC/USDT',
    max_labels=10000  # Stop after 10k labels
)
```

### 2. Parallel Component Training

Train components in parallel:

```python
from concurrent.futures import ProcessPoolExecutor

def train_single(component_name):
    return trainer.train_single_component(
        data=df,
        component_name=component_name,
        train_fn=train_fn
    )

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(train_single, ['scalp_core', 'confirm_filter', 'regime_classifier', 'risk_context']))
```

### 3. Incremental Updates

For hourly retraining (Mechanic), only label new data:

```python
# Load previous labels
previous_labels = load_from_archive()

# Find new data
latest_timestamp = previous_labels[-1].exit_time
new_data = df.filter(pl.col('timestamp') > latest_timestamp)

# Label only new data
new_labels = labeler.label_dataframe(new_data, symbol='BTC/USDT')

# Combine
all_labels = previous_labels + new_labels

# Train with recency weighting (automatically downweights old)
weighter = RecencyWeighter(halflife_days=10)
weights = weighter.calculate_weights_from_labels(all_labels)
```

---

## Testing

### Run All Examples

```bash
# Basic pipeline
python -m cloud.engine.example_pipeline

# Multi-window training
python -m cloud.engine.example_multi_window_training

# Component configs
python -m cloud.engine.multi_window.component_configs
```

### Validate Your Data

```python
from cloud.engine.data_quality import DataSanityPipeline

pipeline = DataSanityPipeline()

# Validate schema
assert pipeline.validate_schema(your_df), "Invalid schema!"

# Check data quality
clean_df, report = pipeline.clean(your_df)
assert report.cleaned_rows > 0, "No data after cleaning!"

removal_rate = (report.original_rows - report.cleaned_rows) / report.original_rows
assert removal_rate < 0.05, f"High removal rate: {removal_rate:.1%}"
```

---

## Troubleshooting

### Problem: "No data available for component window"

**Cause:** Not enough historical data for component's lookback.

**Solution:**
```python
# Option 1: Reduce lookback
config = ScalpCoreConfig()
config.lookback_days = 30  # Reduce from 60

# Option 2: Load more data
df = load_data(days=800)  # Enough for all components
```

### Problem: "Insufficient samples" warning

**Cause:** Not enough labeled trades after filtering.

**Solution:**
```python
# Option 1: Increase max_labels
labeled_trades = labeler.label_dataframe(
    df=clean_df,
    max_labels=50000  # Increase limit
)

# Option 2: Relax meta-label threshold
meta_labeler = MetaLabeler(
    cost_threshold_bps=3.0,  # Reduce from 5.0
    min_pnl_bps=-2.0  # Allow small losses
)
```

### Problem: Walk-forward shows high overfitting

**Cause:** Model too complex or lookahead bias still present.

**Solution:**
```python
# 1. Check embargo is set
validator = WalkForwardValidator(
    embargo_minutes=30  # Must have embargo!
)

# 2. Simplify model
model = XGBClassifier(
    max_depth=3,  # Reduce from 6
    min_child_weight=10  # Increase regularization
)

# 3. Increase recency weighting
weighter = RecencyWeighter(halflife_days=5)  # More aggressive decay
```

---

## Next Steps

1. **Phase 1 (DONE):** Data pipeline + multi-window training
2. **Phase 2 (TODO):** Integration with existing RLTrainingPipeline
3. **Phase 3 (TODO):** Mechanic hourly retraining
4. **Phase 4 (TODO):** Drift detection and alerts
5. **Phase 5 (TODO):** Artifact deployment to Hamilton

---

## References

### Key Papers

1. **"Advances in Financial Machine Learning"** - Marcos López de Prado
   - Triple-barrier labeling (Chapter 3)
   - Meta-labeling (Chapter 3)
   - Sample weighting (Chapter 4)
   - Purged k-fold CV (Chapter 7)

2. **"Machine Learning for Asset Managers"** - Marcos López de Prado
   - Feature importance (Chapter 6)
   - Bet sizing (Chapter 5)

### Documentation

- [Triple-Barrier Method](https://quantdare.com/triple-barrier-method/)
- [Walk-Forward Validation](https://www.quantstart.com/articles/walk-forward-analysis/)
- [Transaction Cost Analysis](https://www.investopedia.com/terms/t/tca.asp)

---

## Support

For questions or issues:
1. Check examples: `example_pipeline.py`, `example_multi_window_training.py`
2. Review docstrings in each module
3. Run with `structlog` debug logging enabled

**Happy training! 🚀**
