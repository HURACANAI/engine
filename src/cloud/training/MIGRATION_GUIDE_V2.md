

# Migration Guide: V1 → V2 Data Pipeline

**Upgrade your training pipeline from naive labeling to production-grade data preparation.**

---

## Overview

### What's Changing

| Component | V1 (Current) | V2 (New) | Why Upgrade |
|-----------|--------------|----------|-------------|
| **Labeling** | Naive forward-looking | Triple-barrier (TP/SL/timeout) | Eliminates lookahead bias |
| **Costs** | Static assumption | Realistic TCA (fees+spread+slippage) | Matches live trading costs |
| **Labels** | "Will price go up?" | "Profitable after costs?" | Focuses on profitability |
| **Weighting** | Equal weights | Exponential recency decay | Recent data weighted higher |
| **Data Quality** | Minimal | Comprehensive cleaning | Removes bad data |
| **Validation** | Single-fold | Walk-forward with embargo | Prevents label leakage |

### Expected Improvements

Based on V2 design principles:

- **Reduced overfitting**: No lookahead bias → better live performance
- **Cost-aware models**: Labels include realistic costs → profitable trades
- **Better generalization**: Recency weighting → adapts to changing markets
- **Cleaner data**: Sanity pipeline → fewer bad trades from bad data

---

## Migration Paths

### Path 1: Drop-In Replacement ⚡ **RECOMMENDED**

**Who:** Anyone wanting immediate V2 benefits with minimal code changes

**Effort:** 5 minutes

**Steps:**

1. **Change import:**
   ```python
   # OLD
   from cloud.training.pipelines.rl_training_pipeline import RLTrainingPipeline

   # NEW
   from cloud.training.pipelines.rl_training_pipeline_v2 import RLTrainingPipelineV2
   ```

2. **Update initialization:**
   ```python
   # OLD
   pipeline = RLTrainingPipeline(settings=settings, dsn=dsn)

   # NEW
   pipeline = RLTrainingPipelineV2(
       settings=settings,
       dsn=dsn,
       use_v2_pipeline=True,  # Enable V2 features
       trading_mode='scalp'    # 'scalp' or 'runner'
   )
   ```

3. **Everything else stays the same:**
   ```python
   # Same API
   metrics = pipeline.train_on_symbol(
       symbol='BTC/USDT',
       exchange_client=exchange,
       lookback_days=365
   )

   # Metrics now include V2 stats
   print(f"V2 profitable labels: {metrics['v2_profitable_labels']}")
   print(f"V2 avg net P&L: {metrics['v2_avg_net_pnl_bps']} bps")
   ```

**Benefits:**
- ✅ 5-minute migration
- ✅ All V2 features enabled
- ✅ Backward compatible (same API)
- ✅ Can toggle V2 on/off (`use_v2_pipeline=False`)

**Risks:**
- ⚠️ Different training data (triple-barrier vs naive)
- ⚠️ Fewer labels (meta-labeling filters unprofitable)
- ⚠️ Need to retrain models from scratch

---

### Path 2: Adapter Integration 🛠️

**Who:** Custom training workflows, non-RL models, gradual rollout

**Effort:** 30 minutes

**Steps:**

1. **Create adapter:**
   ```python
   from cloud.training.integrations.v2_pipeline_adapter import create_v2_scalp_adapter

   adapter = create_v2_scalp_adapter(exchange='binance')
   ```

2. **Process data:**
   ```python
   # Load candles (existing code)
   candles = loader.load(query)

   # Process through V2 pipeline
   labeled_trades, weights = adapter.process(
       data=candles,
       symbol='BTC/USDT'
   )
   ```

3. **Use in your training loop:**
   ```python
   for i, trade in enumerate(labeled_trades):
       weight = weights[i] if weights else 1.0

       # Your custom training logic
       model.add_sample(
           features=trade_to_features(trade),
           label=trade.meta_label,
           weight=weight
       )

   model.fit()
   ```

**Benefits:**
- ✅ Maximum flexibility
- ✅ Works with any ML framework
- ✅ Can use V2 components selectively
- ✅ Easy to A/B test V1 vs V2

**Use Cases:**
- XGBoost/LightGBM training
- Custom RL implementations
- Research experiments
- A/B testing

---

### Path 3: Gradual Migration 🐌

**Who:** Production systems, risk-averse teams, need stability

**Effort:** 1-2 weeks

**Steps:**

#### **Week 1: Data Quality Only**

```python
from cloud.training.integrations.v2_pipeline_adapter import V2PipelineAdapter, V2PipelineConfig

config = V2PipelineConfig(
    enable_data_quality=True,      # ← Enable cleaning
    enable_meta_labeling=False,     # Keep legacy labeling
    enable_recency_weighting=False  # Keep equal weights
)

adapter = V2PipelineAdapter(config=config)
labeled_trades, _ = adapter.process(data, symbol='BTC/USDT')
```

**Validate:**
- Compare # of candles before/after cleaning
- Check # of duplicates/outliers removed
- Verify no major performance changes

#### **Week 2: Triple-Barrier Labeling**

```python
config = V2PipelineConfig(
    enable_data_quality=True,
    enable_meta_labeling=False,     # Not yet
    enable_recency_weighting=False,
    # Triple-barrier parameters
    scalp_tp_bps=15.0,
    scalp_sl_bps=10.0,
    scalp_timeout_minutes=30
)

adapter = V2PipelineAdapter(config=config)
labeled_trades, _ = adapter.process(data, symbol='BTC/USDT')
```

**Validate:**
- Compare # of labels (may be different)
- Check exit reasons (tp/sl/timeout distribution)
- Verify costs are realistic

#### **Week 3: Meta-Labeling**

```python
config = V2PipelineConfig(
    enable_data_quality=True,
    enable_meta_labeling=True,      # ← Filter to profitable
    enable_recency_weighting=False,
    meta_cost_threshold_bps=5.0
)

adapter = V2PipelineAdapter(config=config)
labeled_trades, _ = adapter.process(data, symbol='BTC/USDT')
```

**Validate:**
- Check profitable % (should be 45-55%)
- Verify avg net P&L > 0
- Compare training metrics

#### **Week 4: Recency Weighting (Full V2)**

```python
config = V2PipelineConfig(
    enable_data_quality=True,
    enable_meta_labeling=True,
    enable_recency_weighting=True,  # ← Full V2
    recency_halflife_days=10.0
)

adapter = V2PipelineAdapter(config=config)
labeled_trades, weights = adapter.process(data, symbol='BTC/USDT')
```

**Validate:**
- Check effective sample size
- Verify recent data has higher weights
- Compare model performance

**Benefits:**
- ✅ Low risk (incremental changes)
- ✅ Easy to rollback
- ✅ Validate each step
- ✅ Learn V2 gradually

**Risks:**
- ⚠️ Slower migration
- ⚠️ More testing required

---

## Configuration Reference

### V2PipelineConfig

```python
@dataclass
class V2PipelineConfig:
    # Data quality
    enable_data_quality: bool = True
    outlier_threshold_pct: float = 0.10  # Remove >10% moves
    max_gap_minutes: int = 5             # Gap detection

    # Trading mode
    trading_mode: str = 'scalp'  # 'scalp' or 'runner'

    # Scalp mode (quick in/out)
    scalp_tp_bps: float = 15.0           # Target £1.50 on £1000
    scalp_sl_bps: float = 10.0
    scalp_timeout_minutes: int = 30

    # Runner mode (let winners run)
    runner_tp_bps: float = 80.0
    runner_sl_bps: float = 40.0
    runner_timeout_minutes: int = 10080  # 7 days

    # Meta-labeling
    enable_meta_labeling: bool = True
    meta_cost_threshold_bps: float = 5.0  # Must beat costs by 5 bps

    # Recency weighting
    enable_recency_weighting: bool = True
    recency_halflife_days: float = 10.0   # Scalp: 10, Runner: 20

    # Walk-forward validation (optional)
    enable_walk_forward: bool = False
    walk_forward_train_days: int = 30
    walk_forward_test_days: int = 1
    embargo_minutes: int = 30

    # Performance
    max_labels: int = 10000  # Limit for speed
```

### Presets

#### Scalp Mode
```python
from cloud.training.integrations.v2_pipeline_adapter import create_v2_scalp_adapter

adapter = create_v2_scalp_adapter(exchange='binance')
# TP: 15 bps, SL: 10 bps, Timeout: 30 min, Halflife: 10 days
```

#### Runner Mode
```python
from cloud.training.integrations.v2_pipeline_adapter import create_v2_runner_adapter

adapter = create_v2_runner_adapter(exchange='binance')
# TP: 80 bps, SL: 40 bps, Timeout: 7 days, Halflife: 20 days
```

---

## Testing & Validation

### Pre-Migration Checklist

- [ ] Baseline V1 metrics recorded
- [ ] Test dataset prepared (BTC/USDT, 90+ days)
- [ ] Rollback plan documented
- [ ] Team trained on V2 concepts

### Post-Migration Validation

#### 1. Data Quality Metrics

```python
# Check sanity report
clean_df, report = pipeline.sanity_pipeline.clean(raw_data)

print(f"Duplicates removed: {report.duplicates_removed}")
print(f"Outliers removed: {report.outliers_removed}")
print(f"Gaps filled: {report.gaps_filled}")

# Should see <5% removal rate
removal_rate = (report.original_rows - report.cleaned_rows) / report.original_rows
assert removal_rate < 0.05, "High removal rate!"
```

#### 2. Labeling Metrics

```python
# Check label distribution
labeled_trades, _ = adapter.process(data, 'BTC/USDT')

profitable = sum(1 for t in labeled_trades if t.meta_label == 1)
profitable_pct = profitable / len(labeled_trades) * 100

print(f"Total labels: {len(labeled_trades)}")
print(f"Profitable: {profitable} ({profitable_pct:.1f}%)")

# Should see 45-55% profitable
assert 45 <= profitable_pct <= 55, "Label distribution off!"
```

#### 3. Cost Realism

```python
# Check average costs
avg_costs = sum(t.costs_bps for t in labeled_trades) / len(labeled_trades)

print(f"Avg costs: {avg_costs:.2f} bps")

# Binance scalp costs should be 8-12 bps
assert 8 <= avg_costs <= 12, "Costs unrealistic!"
```

#### 4. Recency Weighting

```python
# Check effective sample size
weights = adapter.weighter.calculate_weights_from_labels(labeled_trades)
ess = adapter.weighter.get_effective_sample_size(weights)
efficiency = ess / len(weights) * 100

print(f"Effective samples: {ess:.0f} (out of {len(weights)})")
print(f"Sample efficiency: {efficiency:.1f}%")

# 10-day halflife should give ~60-70% efficiency on 90-day window
assert 50 <= efficiency <= 80, "Weighting too aggressive!"
```

#### 5. Model Performance

```python
# Compare V1 vs V2 training
v1_metrics = train_with_v1(data)
v2_metrics = train_with_v2(data)

print(f"V1 train win rate: {v1_metrics['train_win_rate']:.1%}")
print(f"V2 train win rate: {v2_metrics['train_win_rate']:.1%}")
print(f"V1 test win rate: {v1_metrics['test_win_rate']:.1%}")
print(f"V2 test win rate: {v2_metrics['test_win_rate']:.1%}")

# V2 should have better test performance (less overfitting)
assert v2_metrics['test_win_rate'] > v1_metrics['test_win_rate'], "V2 not better!"
```

---

## Common Issues & Solutions

### Issue 1: Fewer Labels Than Expected

**Symptom:**
```
V1: 10,000 labels
V2: 3,500 labels
```

**Cause:** Meta-labeling filters unprofitable trades

**Solution:**
```python
# Option 1: Relax meta-label threshold
config.meta_cost_threshold_bps = 3.0  # Reduce from 5.0

# Option 2: Disable meta-labeling temporarily
config.enable_meta_labeling = False

# Option 3: Increase max_labels
config.max_labels = 20000
```

### Issue 2: High Data Removal Rate

**Symptom:**
```
Original rows: 10,000
Cleaned rows: 7,000 (30% removed!)
```

**Cause:** Outlier threshold too aggressive or bad data source

**Solution:**
```python
# Option 1: Relax outlier threshold
config.outlier_threshold_pct = 0.20  # Allow 20% moves

# Option 2: Check data source
# Bad data → high removal is GOOD (V2 protecting you)

# Option 3: Disable specific checks
sanity_pipeline.remove_outliers = False
```

### Issue 3: Costs Too High

**Symptom:**
```
Avg costs: 25 bps (expected 8-12 bps)
```

**Cause:** Wrong exchange fees or mode

**Solution:**
```python
# Check exchange
adapter = V2PipelineAdapter(exchange='binance')  # Not 'kraken'

# Check mode
config.trading_mode = 'scalp'  # Maker fees, not taker

# Verify fee schedule
from cloud.engine.data_quality import HistoricalFeeManager
fee_mgr = HistoricalFeeManager()
fee = fee_mgr.get_fee_for_date('binance', datetime.now(), is_maker=True)
print(f"Current Binance maker fee: {fee} bps")
```

### Issue 4: Walk-Forward Validation Fails

**Symptom:**
```
Train Sharpe: 2.5
Test Sharpe: 0.3 (overfitting!)
```

**Cause:** Lookahead bias still present or model too complex

**Solution:**
```python
# Check embargo is set
validator = WalkForwardValidator(embargo_minutes=30)  # Must have embargo!

# Simplify model
model = XGBClassifier(
    max_depth=3,  # Reduce from 6
    min_child_weight=10  # Increase regularization
)

# Increase recency weighting
config.recency_halflife_days = 5.0  # More aggressive
```

---

## Rollback Plan

If V2 causes issues, you can rollback quickly:

### Quick Rollback (5 minutes)

```python
# In RLTrainingPipelineV2
pipeline = RLTrainingPipelineV2(
    settings=settings,
    dsn=dsn,
    use_v2_pipeline=False  # ← Disable V2, use legacy
)
```

### Full Rollback (10 minutes)

```python
# Switch back to V1
from cloud.training.pipelines.rl_training_pipeline import RLTrainingPipeline

pipeline = RLTrainingPipeline(settings=settings, dsn=dsn)
```

### Data Preservation

V2 doesn't delete V1 data:
- V1 labels still in database
- V1 models still loadable
- V1 metrics still tracked

You can run V1 and V2 in parallel for A/B testing.

---

## Performance Considerations

### Speed

**V2 is slower than V1** (but worth it for quality):

| Component | V1 Speed | V2 Speed | Reason |
|-----------|----------|----------|--------|
| Data loading | 1x | 1x | Same |
| Data quality | - | 0.9x | Cleaning overhead |
| Labeling | 1x | 0.5x | Triple-barrier scan |
| Training | 1x | 1x | Same |
| **Overall** | **1x** | **~0.6x** | **40% slower** |

**Mitigation:**

```python
# 1. Limit labels
config.max_labels = 5000  # Faster labeling

# 2. Sample candles
df_sampled = df.sample_fraction(0.5)  # Use 50% of data

# 3. Parallelize (future)
# Will add multiprocessing support
```

### Memory

V2 uses same memory as V1 (Polars is efficient).

---

## Next Steps After Migration

### 1. Monitor V2 Metrics

```python
# Track V2-specific metrics
metrics = pipeline.train_on_symbol(...)

print(f"V2 profitable %: {metrics['v2_profitable_pct']:.1f}%")
print(f"V2 avg net P&L: {metrics['v2_avg_net_pnl_bps']:+.2f} bps")
print(f"V2 avg costs: {metrics['v2_avg_costs_bps']:.2f} bps")
print(f"V2 effective samples: {metrics.get('v2_effective_samples', 'N/A')}")
```

### 2. Compare Live Performance

- Deploy V2-trained models to Hamilton
- Track live win rate, P&L, costs
- Compare to V1 baseline
- Iterate on V2 config based on results

### 3. Tune V2 Parameters

```python
# Experiment with:
- TP/SL levels (scalp_tp_bps, scalp_sl_bps)
- Recency halflife (faster decay for volatile markets)
- Meta-label threshold (tighter for higher quality)
- Timeouts (shorter for scalp, longer for runner)
```

### 4. Enable Walk-Forward Validation

```python
config.enable_walk_forward = True
config.walk_forward_train_days = 30
config.embargo_minutes = 30

# Catch overfitting before deployment
validation_results = adapter.validate_trades(labeled_trades)
if validation_results['overfitting_detected']:
    print("⚠️ Overfitting detected! Simplify model.")
```

---

## Support & Resources

### Documentation

- **V2 Pipeline README**: `cloud/engine/README_V2_PIPELINE.md`
- **Implementation Summary**: `cloud/engine/IMPLEMENTATION_SUMMARY.md`
- **Integration Example**: `cloud/training/examples/v2_integration_example.py`

### Examples

```bash
# Run complete V2 pipeline demo
python -m cloud.engine.example_pipeline

# Run multi-window training demo
python -m cloud.engine.example_multi_window_training

# Run integration examples
python -m cloud.training.examples.v2_integration_example
```

### Key Files

- **RLTrainingPipelineV2**: `cloud/training/pipelines/rl_training_pipeline_v2.py`
- **V2PipelineAdapter**: `cloud/training/integrations/v2_pipeline_adapter.py`
- **Data Quality**: `cloud/engine/data_quality/`
- **Labeling**: `cloud/engine/labeling/`
- **Costs**: `cloud/engine/costs/`
- **Weighting**: `cloud/engine/weighting/`

---

## Summary

### Migration Checklist

- [ ] Choose migration path (drop-in / adapter / gradual)
- [ ] Record baseline V1 metrics
- [ ] Test V2 on historical data (BTC/USDT, 90 days)
- [ ] Validate data quality metrics
- [ ] Validate labeling distribution
- [ ] Validate cost realism
- [ ] Compare V1 vs V2 model performance
- [ ] Update production training code
- [ ] Monitor V2 metrics in production
- [ ] Compare live trading performance
- [ ] Iterate and tune V2 parameters

### Expected Timeline

- **Drop-in replacement**: 1 day (test + deploy)
- **Adapter integration**: 3 days (integrate + test + deploy)
- **Gradual migration**: 2-3 weeks (incremental rollout)

### Success Criteria

- ✅ V2 data quality: <5% removal rate
- ✅ V2 labels: 45-55% profitable after costs
- ✅ V2 costs: 8-12 bps for Binance scalp
- ✅ V2 test performance: Better than V1 (less overfitting)
- ✅ V2 live performance: Matches backtest (no lookahead bias)

---

**Ready to migrate? Start with the integration example!**

```bash
python -m cloud.training.examples.v2_integration_example
```

Good luck! 🚀
