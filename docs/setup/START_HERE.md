# 🚀 START HERE - Brain Library Quick Start

## What is Brain Library?

Brain Library is a comprehensive ML enhancement system that automatically:
- ✅ Analyzes feature importance after training
- ✅ Compares and selects best models
- ✅ Tracks model versions with automatic rollback
- ✅ Monitors data quality
- ✅ Selects models based on volatility regime

## 🎯 Quick Start (5 Minutes)

### Step 1: Configure Database

Edit `config/base.yaml`:
```yaml
postgres:
  dsn: "postgresql://user:password@localhost:5432/huracan"
```

Or set environment variable:
```bash
export DATABASE_DSN="postgresql://user:password@localhost:5432/huracan"
```

### Step 2: Verify Installation

```bash
python scripts/verify_brain_library_integration.py
```

Expected output:
- ✅ All imports successful
- ✅ Brain Library initialized
- ✅ Database schema created
- ✅ All services available

### Step 3: Run Engine Training

Brain Library integrates **automatically** - just run Engine training:

```bash
python -m src.cloud.training.pipelines.daily_retrain
```

**That's it!** Brain Library will:
- ✅ Analyze feature importance after training
- ✅ Store model metrics
- ✅ Track model versions
- ✅ Automatically rollback if performance degrades

## ✅ Verify It's Working

### Check Logs
```bash
grep "brain_library" logs/*.log
```

You should see:
- `brain_library_initialized`
- `brain_integration_complete`
- `feature_importance_analyzed`
- `model_version_registered`

### Check Database
```sql
-- Check feature importance
SELECT * FROM feature_importance ORDER BY analysis_date DESC LIMIT 10;

-- Check model metrics
SELECT * FROM model_metrics ORDER BY evaluation_date DESC LIMIT 10;

-- Check model registry
SELECT * FROM model_registry WHERE is_active = TRUE;
```

## 📊 What You Get

### Automatic Feature Importance
- Analyzes features after each training run
- Stores top 20 features per symbol
- Tracks feature importance trends

### Model Comparison
- Compares multiple model types
- Selects best model per symbol
- Tracks historical performance

### Model Versioning
- Tracks model versions automatically
- Stores hyperparameters and feature sets
- Automatic rollback if performance degrades >5%

### Comprehensive Metrics
- Sharpe ratio, Sortino ratio
- Hit ratio, Profit factor
- Max drawdown, Calmar ratio
- VaR, CVaR, Skewness, Kurtosis

## 🎓 Usage Examples

### Get Top Features
```python
from src.cloud.training.brain.brain_library import BrainLibrary

brain = BrainLibrary(dsn=dsn)
top_features = brain.get_top_features("BTC/USDT", top_n=20)
print(top_features)
```

### Get Best Model
```python
best_model = brain.get_best_model("BTC/USDT")
print(f"Best model: {best_model['model_type']}")
```

### Select Model by Regime
```python
from src.cloud.training.services.model_selector import ModelSelector

selector = ModelSelector(brain)
model = selector.select_model_for_symbol("BTC/USDT", volatility_regime="high")
```

## 🔍 Troubleshooting

### Brain Library Not Working?

1. **Check database connection:**
   ```bash
   psql $DATABASE_DSN -c "SELECT 1"
   ```

2. **Check logs:**
   ```bash
   grep "brain_library" logs/*.log
   ```

3. **Run verification:**
   ```bash
   python scripts/verify_brain_library_integration.py
   ```

### No Database?

Brain Library will gracefully degrade - Engine continues without it. To enable:
1. Set up PostgreSQL database
2. Configure DSN in settings
3. Run Engine training again

## 📚 Documentation

- **Quick Start**: `QUICK_START.md`
- **Usage Guide**: `docs/BRAIN_LIBRARY_USAGE_GUIDE.md`
- **Action Plan**: `NEXT_STEPS_ACTION_PLAN.md`
- **Deployment**: `DEPLOYMENT_CHECKLIST.md`
- **Architecture**: `HURACAN_ML_ENHANCEMENTS.md`

## 🎉 That's It!

Brain Library is now integrated and ready to use. Just run Engine training and it will automatically:
- Analyze feature importance
- Store model metrics
- Track model versions
- Select best models

**No additional configuration needed!**

---

**Status**: ✅ **READY TO USE**

**Last Updated**: 2025-01-08

